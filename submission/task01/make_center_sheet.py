"""Build the rt-225 centre-point comparison sheet (original vs horizontal flip).

Inputs : dataset-curated/images/val/rt-225.jpg  +  submission/task01/rt-225-centers.json
Output : submission/task01/rt-225-centers-compare.png

Run from the repo root with the repo virtualenv:
    ./.venv/bin/python submission/task01/make_center_sheet.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
VAL = ROOT / "dataset-curated" / "images" / "val"

GREEN, RED, WHITE, GREY, INK = (0, 200, 0), (0, 0, 255), (255, 255, 255), (170, 170, 170), (20, 20, 20)
HEADER, GAP, FOOTER = 46, 20, 34


def annotate(img, box, center, conf, tag):
    canvas = img.copy()
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), GREEN, 2)
    cx, cy = int(round(center[0])), int(round(center[1]))
    cv2.drawMarker(canvas, (cx, cy), RED, cv2.MARKER_CROSS, 26, 2)
    cv2.circle(canvas, (cx, cy), 4, RED, -1)
    cv2.putText(canvas, f"{tag}  conf {conf:.4f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"center ({center[0]:.2f}, {center[1]:.2f})", (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"x={cx}px", (min(max(cx - 18, 2), 296), min(cy + 36, 628)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, RED, 1, cv2.LINE_AA)
    return canvas


def main():
    centers = json.loads((OUT / "rt-225-centers.json").read_text(encoding="utf-8"))
    original = cv2.imread(str(VAL / "rt-225.jpg"))
    if original is None:
        raise SystemExit("先运行 detect_cases.py 生成 rt-225.jpg 相关结果")
    flipped = cv2.flip(original, 1)

    left = annotate(original, centers["original"]["bbox_xyxy"],
                    centers["original"]["center_xy"], centers["original"]["confidence"], "ORIGINAL")
    right = annotate(flipped, centers["flipped"]["bbox_xyxy"],
                     centers["flipped"]["center_xy"], centers["flipped"]["confidence"], "FLIP (mirrored)")

    height, width = original.shape[:2]
    canvas = numpy.full((HEADER + height + FOOTER, width * 2 + GAP, 3), 255, numpy.uint8)
    canvas[HEADER:HEADER + height, 0:width] = left
    canvas[HEADER:HEADER + height, width + GAP:width * 2 + GAP] = right
    cv2.line(canvas, (width + GAP // 2, HEADER), (width + GAP // 2, HEADER + height), GREY, 1)

    cv2.putText(canvas, "rt-225 centre-point comparison | imgsz=640, conf=0.10, sports ball",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, INK, 2, cv2.LINE_AA)
    ox, fx = centers["original"]["center_xy"][0], centers["flipped"]["center_xy"][0]
    cv2.putText(canvas,
                f"centre x: {ox:.2f} -> {fx:.2f}   mirror expected {width - ox:.2f}   "
                f"error {abs(fx - (width - ox)):.2f} px",
                (8, HEADER + height + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)

    target = OUT / "rt-225-centers-compare.png"
    cv2.imwrite(str(target), canvas)
    print(f"written: {target} {canvas.shape}")


if __name__ == "__main__":
    main()
