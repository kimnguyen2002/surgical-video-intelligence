"""
Experiment tracking.

A research workspace is only useful if its results are reproducible, so every
run records the *whole* context it depended on: model, configuration, dataset
identity, code version, and the resolved capability tier. Two runs are only
comparable if those match, and :meth:`ExperimentTracker.compare` says so
explicitly instead of quietly ranking incomparable numbers against each other.

Stored as one JSON file per run — no tracking server required, and runs can be
committed alongside the code that produced them.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ai.common import capability_report, settings

logger = logging.getLogger("charlie.experiments")


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "not-a-git-repo"


@dataclass
class Experiment:
    id: str
    name: str
    task: str  # tool_detection | step_recognition | retrieval | ablation | custom
    model: str
    config: dict = field(default_factory=dict)
    dataset: str = ""
    metrics: dict = field(default_factory=dict)
    #: Per-epoch or per-step history, for plotting learning curves.
    history: list[dict] = field(default_factory=list)
    notes: str = ""
    status: str = "running"  # running | complete | failed
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    environment: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def duration(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return round(self.completed_at - self.created_at, 2)

    def reproducibility_key(self) -> tuple:
        """What must match for two runs to be legitimately comparable."""
        return (
            self.task,
            self.dataset,
            self.environment.get("git_revision"),
            self.environment.get("capability_tier"),
        )


class ExperimentTracker:
    """Records, compares, and exports experiment runs."""

    def __init__(self, directory: Optional[Path] = None):
        self.directory = Path(directory or settings.experiments_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._experiments: dict[str, Experiment] = {}
        self._load_all()

    # -- lifecycle ---------------------------------------------------------
    def start(
        self,
        name: str,
        task: str,
        model: str,
        config: Optional[dict] = None,
        dataset: str = "",
        notes: str = "",
    ) -> Experiment:
        capabilities = capability_report()
        experiment = Experiment(
            id=uuid.uuid4().hex[:12],
            name=name,
            task=task,
            model=model,
            config=config or {},
            dataset=dataset,
            notes=notes,
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "git_revision": _git_revision(),
                "capability_tier": capabilities["tier"],
                "gpu": capabilities["gpu"],
            },
        )
        with self._lock:
            self._experiments[experiment.id] = experiment
            self._persist(experiment)
        logger.info("Started experiment %s (%s / %s)", experiment.id, task, model)
        return experiment

    def log(self, experiment_id: str, step: int, **metrics) -> Optional[Experiment]:
        """Append one step/epoch of metrics."""
        with self._lock:
            experiment = self._experiments.get(experiment_id)
            if experiment is None:
                return None
            experiment.history.append({"step": step, "timestamp": time.time(), **metrics})
            self._persist(experiment)
            return experiment

    def finish(
        self,
        experiment_id: str,
        metrics: Optional[dict] = None,
        status: str = "complete",
        artifacts: Optional[dict] = None,
    ) -> Optional[Experiment]:
        with self._lock:
            experiment = self._experiments.get(experiment_id)
            if experiment is None:
                return None
            experiment.metrics.update(metrics or {})
            experiment.artifacts.update(artifacts or {})
            experiment.status = status
            experiment.completed_at = time.time()
            self._persist(experiment)
        logger.info("Finished experiment %s (%s)", experiment_id, status)
        return experiment

    def delete(self, experiment_id: str) -> bool:
        with self._lock:
            if self._experiments.pop(experiment_id, None) is None:
                return False
            (self.directory / f"{experiment_id}.json").unlink(missing_ok=True)
        return True

    # -- queries -----------------------------------------------------------
    def get(self, experiment_id: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_id)

    def list(self, task: Optional[str] = None, limit: int = 100) -> list[dict]:
        items = list(self._experiments.values())
        if task:
            items = [e for e in items if e.task == task]
        items.sort(key=lambda e: e.created_at, reverse=True)
        return [
            {**e.to_dict(), "duration": e.duration, "history": len(e.history)}
            for e in items[:limit]
        ]

    def compare(self, experiment_ids: list[str], metric: Optional[str] = None) -> dict:
        """
        Compare runs side by side, and say whether they are comparable at all.
        """
        experiments = [self._experiments[i] for i in experiment_ids if i in self._experiments]
        if len(experiments) < 2:
            return {"error": "Need at least two existing experiments to compare."}

        keys = {e.reproducibility_key() for e in experiments}
        comparable = len(keys) == 1

        warnings: list[str] = []
        if not comparable:
            if len({e.dataset for e in experiments}) > 1:
                warnings.append(
                    "Runs used different datasets — metrics are not directly comparable."
                )
            if len({e.environment.get("git_revision") for e in experiments}) > 1:
                warnings.append(
                    "Runs came from different code revisions — a difference may be a code change, not a model change."
                )
            if len({e.environment.get("capability_tier") for e in experiments}) > 1:
                warnings.append(
                    "Runs executed on different capability tiers (e.g. GPU vs pure-Python) — latency figures are not comparable."
                )
            if len({e.task for e in experiments}) > 1:
                warnings.append("Runs are for different tasks.")

        metric_names = sorted({m for e in experiments for m in e.metrics})
        table = []
        for experiment in experiments:
            table.append(
                {
                    "id": experiment.id,
                    "name": experiment.name,
                    "model": experiment.model,
                    "task": experiment.task,
                    "dataset": experiment.dataset,
                    "status": experiment.status,
                    "duration": experiment.duration,
                    "git_revision": experiment.environment.get("git_revision"),
                    "metrics": {m: experiment.metrics.get(m) for m in metric_names},
                }
            )

        best = None
        if metric and comparable:
            scored = [
                (e.metrics.get(metric), e.id) for e in experiments if metric in e.metrics
            ]
            if scored:
                best = max(scored, key=lambda pair: pair[0])[1]

        return {
            "comparable": comparable,
            "warnings": warnings,
            "metrics": metric_names,
            "experiments": table,
            "best": best,
            "ranking_metric": metric,
        }

    def stats(self) -> dict:
        by_task: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for experiment in self._experiments.values():
            by_task[experiment.task] = by_task.get(experiment.task, 0) + 1
            by_status[experiment.status] = by_status.get(experiment.status, 0) + 1
        return {
            "total": len(self._experiments),
            "by_task": by_task,
            "by_status": by_status,
        }

    # -- persistence -------------------------------------------------------
    def _persist(self, experiment: Experiment) -> None:
        try:
            (self.directory / f"{experiment.id}.json").write_text(
                json.dumps(experiment.to_dict(), indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.error("Could not persist experiment %s: %s", experiment.id, exc)

    def _load_all(self) -> None:
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._experiments[data["id"]] = Experiment(**data)
            except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.warning("Skipping unreadable experiment %s: %s", path.name, exc)
        if self._experiments:
            logger.info("Restored %d experiments", len(self._experiments))


experiment_tracker = ExperimentTracker()
