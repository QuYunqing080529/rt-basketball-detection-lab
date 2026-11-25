"""Generate dataset-curated/ for ONE video source, reusing lab.py's exact logic.

Why this exists: `python lab.py prepare` walks sources in annotations.csv order
(plos01 first). On this machine the PLOS/Jackson videos are unreachable, so
prepare dies before it ever reaches `rt`. This helper runs the same code path
for a single source, so the RT split can be produced offline.

Usage (from the repo root, .venv active):
    python tools/prepare_subset.py rt

Output is byte-for-byte what `lab.py prepare` would write for that source:
  dataset-curated/images/<split>/<source>-<frame:03>.jpg   (JPEG q=92)
  dataset-curated/labels/<split>/<source>-<frame:03>.txt   (YOLO, normalised)

Re-running is safe; a later full `lab.py prepare` overwrites with identical bytes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import lab  # noqa: E402

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "rt"

rows = [row for row in lab.read_annotations() if row["source"] == SOURCE]
if not rows:
    raise SystemExit(f"annotations.csv 中没有来源 {SOURCE}")

meta = lab.json.loads((lab.ROOT / "sources.json").read_text(encoding="utf-8"))[SOURCE]
path = lab.ROOT / meta["file"]
lab.get_asset(path, meta["url"], meta["sha256"])

frames = {row["frame"]: row for row in rows}
reader = cv2.VideoCapture(str(path))
try:
    for index in range(max(frames) + 1):
        ok, image = reader.read()
        if not ok:
            raise RuntimeError(f"无法解码: {SOURCE} 第 {index} 帧")
        if index in frames:
            lab.write_sample(image, frames[index])
finally:
    reader.release()

for split in lab.SPLITS:
    (lab.DATA / "labels" / (split + ".cache")).unlink(missing_ok=True)

splits = {}
for row in rows:
    splits.setdefault(row["split"], []).append(row["stem"])
print(f"{SOURCE}: 写出 {len(rows)} 帧 -> {lab.DATA}")
for split, stems in sorted(splits.items()):
    print(f"  {split}: {len(stems)} 帧  {' '.join(sorted(stems))}")
