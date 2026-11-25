# Task 02 — CVAT 标注与坐标还原

仓库：`/home/QYQ/work/rt-basketball-detection-lab`（分支 `learn/neural-basics`），解释器 `./.venv/bin/python`。

## 素材（4 张）

| 文件 | 原图宽高 | 画面 |
|---|---|---|
| `plos02-055.jpg` | **1920×1080** | 另一机位，球在球员手边，尺寸很小 |
| `rt-125.jpg` | **360×640** | 球完整可见 |
| `rt-150.jpg` | **360×640** | 球部分出画，贴画面左边缘 |
| `rt-200.jpg` | **360×640** | 无球帧 |

## 标注方式

CVAT Online，任务内只建一个类别 `basketball`；四张图一次性上传（1 个 Job、4 帧，帧序按文件名：`plos02-055` → `rt-125` → `rt-150` → `rt-200`）。导出格式选 **Ultralytics YOLO Detection 1.0**，保持 **Save images 关闭**（只导出标注，原图沿用本地）。

## 导出包结构

```
cvat-yolo-export.zip
├── data.yaml                 # names: {0: basketball}, path: ., train: train.txt
├── train.txt                 # 4 张图片清单（含无球的 rt-200）
└── labels/train/
    ├── plos02-055.txt
    ├── rt-125.txt
    └── rt-150.txt            # rt-200 无框，故无对应 txt
```

`train.txt` 共 4 行：

```
data/images/train/plos02-055.jpg
data/images/train/rt-125.jpg
data/images/train/rt-150.jpg
data/images/train/rt-200.jpg
```

## 四张图标注核对

| 文件名 | 原图宽高 | 有无标签 | 说明 |
|---|---|---|---|
| `plos02-055.jpg` | 1920×1080 | **有** | 1 个 `basketball` 框 |
| `rt-125.jpg` | 360×640 | **有** | 1 个框，球完整 |
| `rt-150.jpg` | 360×640 | **有** | 1 个框，球出画，框左边界落在 0 附近 |
| `rt-200.jpg` | 360×640 | **无** | 无球帧，保留在 `train.txt` 中，无对应标签文件 |

## 坐标还原

把导出标签里的归一化 `(cx, cy, w, h)` 按**各自原图的宽高**换回像素 `xyxy`，再画回原图：

| 文件 | 原图宽高 | 还原框 `xyxy`（px） |
|---|---|---|
| `rt-150.jpg` | 360×640 | `(0.24, 243.06, 35.40, 350.98)` |
| `plos02-055.jpg` | 1920×1080 | `(1080.70, 573.66, 1156.31, 646.98)` |

还原图：`restore-rt-150.png`、`restore-plos02-055.png`（图中直接标注文件名、原图尺寸与像素坐标）。

## 自查（与仓库 `annotations.csv` 的人工框对比，不随交付提交）

| 文件 | 本任务框 | 仓库人工框 | IoU | 判断 |
|---|---|---|---|---|
| `rt-150.jpg` | `(0.24, 243.06, 35.40, 350.98)` | `(0, 247, 31, 344)` | 0.78 | 本任务框略宽、略高，可再收紧 |
| `plos02-055.jpg` | `(1080.70, 573.66, 1156.31, 646.98)` | `(1072, 563, 1141, 632)` | 0.52 | 5× 放大比对后，本任务框比仓库框更贴合球体；仓库框整体偏左上约 10 px |

> 说明：`plos02-055` 的球只有约 75×73 px，10 px 的偏移就足以让 IoU 掉到 0.5 附近。**IoU 低不等于画错了**，要放大看框与球的相对位置。

## 复跑

```bash
cd /home/QYQ/work/rt-basketball-detection-lab
# 1) 把 CVAT 导出的 ZIP 解压到任意目录（默认按 /mnt/c/Users/admin/Downloads/_labeling_extract）
# 2) 生成两张还原图与 restore-check.json
./.venv/bin/python submission/task02/restore_boxes.py
# 如需指定解压目录：
./.venv/bin/python submission/task02/restore_boxes.py /path/to/extracted_export
```

## 文件清单

| 文件 | 内容 |
|---|---|
| `cvat-yolo-export.zip` | CVAT 导出的 Ultralytics YOLO Detection 标注包（原样，未改动） |
| `restore-rt-150.png` | rt-150 原图 + 还原框（360×640） |
| `restore-plos02-055.png` | plos02-055 原图 + 还原框（1920×1080） |
| `restore-check.json` | 还原框数值与自查 IoU |
| `restore_boxes.py` | 还原图与 `restore-check.json` 的复跑入口 |
| `README.md` | 本文件 |
