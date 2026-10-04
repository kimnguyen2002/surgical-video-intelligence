"""
Regression tests for the released SurgVU v2 label schema.

The parser was originally written against an assumed schema and could not read
the actual files: `tasks.csv` raised on every case, and `tools.csv` resolved
every row to part 0. Both failures were silent in the sense that matters —
training would still run, the loss would still fall, and the model would be
learning from frames labelled by a timestamp lookup that never matched.

These pin the real column names and value formats.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai.training.labels import (
    build_index,
    case_id_from_path,
    load_label_rows,
    normalise_label,
    parse_case_part,
    parse_time,
    resolve_intervals,
)

TOOLS_HEADER = (
    "index,install_case_part,install_case_time,uninstall_case_part,"
    "uninstall_case_time,arm,commercial_toolname,groundtruth_toolname\n"
)
TASKS_HEADER = "index,start_part,start_time,stop_part,stop_time,groundtruth_taskname\n"


def _case(tmp_path: Path, name: str, tools: str = "", tasks: str = "") -> Path:
    d = tmp_path / name
    d.mkdir()
    if tools:
        (d / "tools.csv").write_text(TOOLS_HEADER + tools, encoding="utf-8")
    if tasks:
        (d / "tasks.csv").write_text(TASKS_HEADER + tasks, encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------


def test_tasks_csv_uses_start_stop_columns(tmp_path):
    """
    `tasks.csv` is `start_part/start_time/stop_part/stop_time`.

    The original alias table had `start_case_part` and `stop_case_part` but not
    these, so every task label in all 155 cases failed to parse.
    """
    d = _case(tmp_path, "case_002", tasks="0,1,30.5,1,162.25,Suturing\n")
    rows = load_label_rows(d / "tasks.csv", "task")
    assert len(rows) == 1
    assert rows[0]["label"] == "suturing"
    assert rows[0]["start_time"] == pytest.approx(30.5)
    assert rows[0]["end_time"] == pytest.approx(162.25)


def test_tools_csv_columns_still_parse(tmp_path):
    d = _case(
        tmp_path,
        "case_002",
        tools="0,1.0,00:43:36.978000,1.0,01:23:09.478000,USM1,Large Needle Driver,needle driver\n",
    )
    rows = load_label_rows(d / "tools.csv", "tool")
    assert len(rows) == 1
    assert rows[0]["label"] == "needle_driver"
    assert rows[0]["start_time"] == pytest.approx(43 * 60 + 36.978)


# ---------------------------------------------------------------------------
# Part / case resolution
# ---------------------------------------------------------------------------


def test_bare_part_number_is_a_part_not_a_case():
    """
    The part column holds a bare number; the case comes from the directory.

    Reading `1.0` as a case id (and defaulting the part to 0) is what made
    every timestamp lookup miss.
    """
    assert parse_case_part("1.0", default_case="002") == ("002", 1)
    assert parse_case_part("1", default_case="002") == ("002", 1)
    assert parse_case_part("3.0", default_case="147") == ("147", 3)


def test_composite_form_from_other_releases_still_works():
    assert parse_case_part("case_042_video_part_003") == ("042", 3)


def test_case_id_comes_from_the_directory(tmp_path):
    assert case_id_from_path(Path("labels/case_002/tools.csv")) == "002"
    d = _case(tmp_path, "case_017", tools="0,2.0,00:00:10,2.0,00:00:20,USM1,x,stapler\n")
    rows = load_label_rows(d / "tools.csv", "tool")
    assert rows[0]["case"] == "017"
    assert rows[0]["start_part"] == 2


# ---------------------------------------------------------------------------
# Time formats — the two files differ
# ---------------------------------------------------------------------------


def test_both_time_formats_parse():
    assert parse_time("00:43:36.978000") == pytest.approx(43 * 60 + 36.978)
    assert parse_time("3226.251309") == pytest.approx(3226.251309)
    assert parse_time("00:00:00") == 0.0


# ---------------------------------------------------------------------------
# Non-instrument rows
# ---------------------------------------------------------------------------


def test_camera_rows_are_dropped_from_tools(tmp_path):
    """
    `nan(camera in)` is the endoscope, not an instrument — 1,449 rows in the
    release. Kept, it becomes a phantom thirteenth class present in most frames.
    """
    d = _case(
        tmp_path,
        "case_002",
        tools=(
            "0,1.0,00:39:45.278000,1.0,00:46:28.478000,USM2,0 Endoscope,nan(camera in)\n"
            "1,1.0,00:43:29.678000,1.0,00:59:38.178000,USM4,ProGrasp,prograsp forceps\n"
            "2,1.0,00:44:00.000000,1.0,00:45:00.000000,USM3,,\n"
        ),
    )
    rows = load_label_rows(d / "tools.csv", "tool")
    assert [r["label"] for r in rows] == ["prograsp_forceps"]


def test_other_is_kept_as_a_task_label(tmp_path):
    """`other` is a real task class, so the tool drop-list must not touch it."""
    d = _case(tmp_path, "case_002", tasks="0,1,10,1,20,other\n")
    rows = load_label_rows(d / "tasks.csv", "task")
    assert [r["label"] for r in rows] == ["other"]


def test_trailing_whitespace_in_label_is_normalised():
    """The release ships 909 rows of `'clip applier '` with a trailing space."""
    assert normalise_label("clip applier ") == "clip_applier"


def test_cat1_singular_scissors_matches_surgvu_plural():
    assert normalise_label("monopolar curved scissor") == "monopolar_curved_scissors"
    assert normalise_label("Monopolar Curved Scissors") == "monopolar_curved_scissors"


# ---------------------------------------------------------------------------
# Cross-part intervals — the case that silently mislabels whole video parts
# ---------------------------------------------------------------------------


def test_interval_spanning_parts_covers_the_middle_part(tmp_path):
    """
    A tool installed in part 1 and removed in part 3 is present for all of
    part 2. Dropping the middle part labels those frames 'no tool' — the loss
    still falls and the result is meaningless.
    """
    d = _case(
        tmp_path,
        "case_002",
        tools="0,1.0,01:00:00,3.0,00:10:00,USM2,x,needle driver\n",
    )
    rows = load_label_rows(d / "tools.csv", "tool")
    durations = {("002", 1): 7200.0, ("002", 2): 7200.0, ("002", 3): 7200.0}
    intervals = resolve_intervals(rows, durations)

    by_part = {i.part: i for i in intervals}
    assert set(by_part) == {1, 2, 3}
    assert by_part[1].start == pytest.approx(3600.0)
    assert by_part[1].end == pytest.approx(7200.0)   # to the end of part 1
    assert by_part[2].start == 0.0                    # all of part 2
    assert by_part[2].end == pytest.approx(7200.0)
    assert by_part[3].start == 0.0
    assert by_part[3].end == pytest.approx(600.0)


def test_timestamp_lookup_is_multilabel_across_arms(tmp_path):
    """Several arms carry tools at once — presence is a union, never a softmax."""
    d = _case(
        tmp_path,
        "case_002",
        tools=(
            "0,1.0,00:00:00,1.0,01:00:00,USM1,x,needle driver\n"
            "1,1.0,00:30:00,1.0,01:30:00,USM3,x,bipolar forceps\n"
        ),
    )
    index = build_index(d / "tools.csv", "tool", {("002", 1): 7200.0})
    assert index.labels_at("002", 1, 60.0) == frozenset({"needle_driver"})
    assert index.labels_at("002", 1, 2400.0) == frozenset(
        {"needle_driver", "bipolar_forceps"}
    )
    assert index.labels_at("002", 1, 4200.0) == frozenset({"bipolar_forceps"})
    assert index.labels_at("002", 1, 6000.0) == frozenset()
