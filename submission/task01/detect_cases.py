"""Single-image detections and a horizontal-flip comparison for task 01.

Run from the repo root with the repo virtualenv:
    ./.venv/bin/python submission/task01/detect_cases.py
"""
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "yolo11n.pt"
VAL = ROOT / "dataset-curated" / "images" / "val"
TARGET_NAME = "sports ball"
DEVICE = "cpu"
IMGSZ = 640
RERUN = "./.venv/bin/python submission/task01/detect_cases.py"


def load_model():
    model = YOLO(str(WEIGHTS))
    index = next(key for key, name in model.names.items() if name == TARGET_NAME)
    return model, index


def infer(model, index, image, conf):
    """Same call shape as lab.py: classes=[target], imgsz, conf, device, verbose=False."""
    result = model.predict(
        image, classes=[index], imgsz=IMGSZ, conf=conf, device=DEVICE, verbose=False,
    )[0]
    return [
        {"bbox_xyxy": [round(float(value), 4) for value in box],
         "confidence": round(float(score), 6)}
        for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
    ]


def draw(image, detections):
    canvas = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection["bbox_xyxy"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"{detection['confidence']:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def best(detections):
    return max(detections, key=lambda item: item["confidence"], default=None)


def describe(detection):
    if detection is None:
        return {"empty": True, "center_xy": None, "confidence": None, "bbox_xyxy": None}
    x1, y1, x2, y2 = detection["bbox_xyxy"]
    return {
        "empty": False,
        "center_xy": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
        "confidence": detection["confidence"],
        "bbox_xyxy": detection["bbox_xyxy"],
    }


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    model, index = load_model()

    # (a) rt-100, conf=0.25, one inference
    image_100 = cv2.imread(str(VAL / "rt-100.jpg"))
    detections_100 = infer(model, index, image_100, 0.25)
    cv2.imwrite(str(OUT / "rt-100-boxed.jpg"), draw(image_100, detections_100))
    dump(OUT / "rt-100-first-inference.json", {
        "weights": str(WEIGHTS),
        "device": DEVICE,
        "imgsz": IMGSZ,
        "conf": 0.25,
        "class_name": TARGET_NAME,
        "class_index": index,
        "image": str(VAL / "rt-100.jpg"),
        "boxes": detections_100,
        "rerun": RERUN,
    })

    # (b) rt-225, conf=0.10
    image_225 = cv2.imread(str(VAL / "rt-225.jpg"))
    detections_225 = infer(model, index, image_225, 0.10)
    cv2.imwrite(str(OUT / "rt-225-boxed.jpg"), draw(image_225, detections_225))

    # (c) horizontal flip of rt-225, PNG first, then inference on the flipped frame
    flipped = cv2.flip(image_225, 1)
    cv2.imwrite(str(OUT / "rt-225-flip.png"), flipped)
    detections_flip = infer(model, index, flipped, 0.10)
    cv2.imwrite(str(OUT / "rt-225-flip-boxed.jpg"), draw(flipped, detections_flip))

    # (d) highest-score box per image, centres in pixel coordinates
    original = describe(best(detections_225))
    mirrored = describe(best(detections_flip))
    dump(OUT / "rt-225-centers.json", {
        "image": str(VAL / "rt-225.jpg"),
        "conf": 0.10,
        "any_side_empty": original["empty"] or mirrored["empty"],
        "original": original,
        "flipped": mirrored,
        "note": "检测为空的一侧 center_xy 为 null（empty=true）",
    })

    print(json.dumps({
        "rt100": detections_100,
        "rt225": detections_225,
        "rt225_flip": detections_flip,
        "centers": {"original": original, "flipped": mirrored},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
