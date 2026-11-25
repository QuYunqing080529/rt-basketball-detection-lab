"""Task 02 — restore exported YOLO boxes onto the two original images.

Reads the CVAT export (Ultralytics YOLO Detection format), converts each
normalised (cx, cy, w, h) back to pixel xyxy using THAT image's own size,
draws it on the original image and writes restore-check.json.

Usage (from the repo root, .venv active):
    python submission/task02/restore_boxes.py [path/to/extracted_export]

Default export path: /mnt/c/Users/admin/Downloads/_labeling_extract
(the contents of the CVAT ZIP, extracted).
"""
import json
import sys
from pathlib import Path

import cv2

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/admin/Downloads/_labeling_extract")
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
IMG = ROOT / "dataset-curated" / "images"

# (stem, image path, repo annotation xyxy — used only for self-check, never drawn)
TARGETS = [
    ("rt-150", IMG / "val" / "rt-150.jpg", (0, 247, 31, 344)),
    ("plos02-055", IMG / "train" / "plos02-055.jpg", (1072, 563, 1141, 632)),
]

GREEN, WHITE = (0, 200, 0), (255, 255, 255)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main():
    report = []
    for stem, path, truth in TARGETS:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"读不到原图: {path}")
        height, width = image.shape[:2]

        label_file = SRC / "labels" / "train" / f"{stem}.txt"
        if not label_file.exists():
            raise SystemExit(f"导出包里缺少 {stem}.txt")
        fields = label_file.read_text().split()
        cls, cx, cy, bw, bh = int(fields[0]), *map(float, fields[1:5])

        box = [round((cx - bw / 2) * width, 2), round((cy - bh / 2) * height, 2),
               round((cx + bw / 2) * width, 2), round((cy + bh / 2) * height, 2)]

        canvas = image.copy()
        band = canvas.copy()
        cv2.rectangle(band, (0, 0), (width, 88), (0, 0, 0), -1)
        cv2.addWeighted(band, 0.45, canvas, 0.55, 0, canvas)
        cv2.rectangle(canvas, (round(box[0]), round(box[1])),
                      (round(box[2]), round(box[3])), GREEN, 2)

        lines = [
            f"{stem}.jpg   original size {width} x {height}",
            f"xyxy = ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f})",
        ]
        if width > 1000:
            lines.append(f"class 0 basketball   exported norm = {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        scale = 0.68 if width > 1000 else 0.5
        for index, text in enumerate(lines):
            cv2.putText(canvas, text, (10, 30 + index * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, WHITE, 2 if index < 2 else 1, cv2.LINE_AA)

        cv2.imwrite(str(OUT / f"restore-{stem}.png"), canvas)
        entry = {"file": f"{stem}.jpg", "size": f"{width}x{height}", "class": cls,
                 "student_box": box, "repo_annotation": list(truth),
                 "iou_student_vs_repo": round(iou(box, truth), 4)}
        report.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    (OUT / "restore-check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("written: restore-check.json")


if __name__ == "__main__":
    main()
