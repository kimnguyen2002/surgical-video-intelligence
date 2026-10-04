"""
Render the clip manifest as a typed TypeScript module.

Kept separate from `build_clips.py` because it is pure text generation with no
ffmpeg dependency: the manifest can be re-rendered without re-encoding 33 MB of
video, which matters while iterating on the interface.

A runtime `fetch('/clips/clips.json')` would also work and is rejected on
purpose. It puts the case library behind a network round trip that can fail,
and it gives the compiler nothing to check — a renamed field would surface as
`undefined` in the player instead of as a build error.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLIPS_JSON = REPO / "public" / "clips" / "clips.json"
CLIPS_TS = REPO / "src" / "data" / "surgvuClips.ts"

HEADER = '''/**
 * GENERATED FILE - do not edit by hand.
 *
 * Produced by scripts/build_clips.py. Regenerate with: npm run build:clips
 *
 * One short excerpt per SurgVU part, bundled with the app so the player is not
 * empty for a visitor who does not have the 8.9 GB release on disk. Each window
 * was chosen by how much *labelled* activity it contains rather than taken from
 * the start of the recording - see the module docstring in build_clips.py.
 */

export interface SurgvuClip {
  caseId: string;
  /** The part of the original recording this excerpt was cut from. */
  part: number;
  /** Path under `public/`, served at the site root. */
  file: string;
  sourceFilename: string;
  /**
   * Offset of the excerpt within the original part, in seconds.
   *
   * Every ground-truth lookup adds this to the player's `currentTime`. Without
   * it the app would read annotations from the opening minutes of the case
   * while the viewer watches its middle - confident, precisely-timed and
   * entirely wrong.
   */
  sourceStartSeconds: number;
  durationSeconds: number;
  sizeBytes: number;
  /** Instruments the release records as installed during this window. */
  tools: string[];
  /** Tasks the release labels during this window. */
  tasks: string[];
}

export const SURGVU_CLIPS: SurgvuClip[] = [
'''

FOOTER = '''];

export function clipFor(caseId: string, part: number): SurgvuClip | undefined {
  return SURGVU_CLIPS.find((c) => c.caseId === caseId && c.part === part);
}

/** Total bundled video, so the About page can state the real figure. */
export const SURGVU_CLIPS_TOTAL_BYTES = SURGVU_CLIPS.reduce(
  (sum, c) => sum + c.sizeBytes,
  0
);
'''


def render(manifest: list[dict]) -> str:
    rows = []
    for m in manifest:
        selection = m.get("selection", {})
        rows.append(
            "  {\n"
            f"    caseId: {json.dumps(m['caseId'])},\n"
            f"    part: {m['part']},\n"
            f"    file: {json.dumps(m['file'])},\n"
            f"    sourceFilename: {json.dumps(m['sourceFilename'])},\n"
            f"    sourceStartSeconds: {m['sourceStartSeconds']},\n"
            f"    durationSeconds: {m['durationSeconds']},\n"
            f"    sizeBytes: {m.get('sizeBytes') or 0},\n"
            f"    tools: {json.dumps(selection.get('tools', []))},\n"
            f"    tasks: {json.dumps(selection.get('tasks', []))},\n"
            "  }"
        )
    return HEADER + ",\n".join(rows) + ",\n" + FOOTER


def main() -> int:
    manifest = json.loads(CLIPS_JSON.read_text())
    CLIPS_TS.write_text(render(manifest))
    print(f"wrote {CLIPS_TS.relative_to(REPO)} ({len(manifest)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
