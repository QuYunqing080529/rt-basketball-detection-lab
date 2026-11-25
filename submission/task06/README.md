# Task 06 — 固定 416×416 ONNX 与双后端对照

仓库：`/home/QYQ/work/rt-basketball-detection-lab`（分支 `learn/neural-basics`），解释器 `./.venv/bin/python`。

## 一、固定导出与验收口径（导出前写定）

见 `export-protocol.md`。写入时间 `2026-11-25 22:35:48`，**早于** ONNX 文件（`22:36:20`）与对照结果（`22:36:50`）。

| 口径项 | 写定的值 |
|---|---|
| 权重来源 | 预训练 `yolo11n.pt`（SHA-256 `0ebbc80d…7644ee1`） |
| 模型格式 | ONNX，固定输入、`dynamic=False` |
| 类别映射 | `sports ball`（COCO 32），两端同一映射 |
| 输入尺寸 | **416 × 416**（letterbox + 补边） |
| 置信度阈值 | **0.10** |
| 匹配门槛 | IoU ≥ 0.50；**切换判据为配对框 IoU ≥ 0.95** |
| 测试图片 | `rt-100`（有球）、`rt-150`（出画）、`rt-200`（无球） |
| 需保存 | 原图像素坐标的框、类别、置信度、空结果、导出日志与对照 JSON |
| 计时口径 | 预热 5 次、重复 20 次、分预处理/推理/后处理三段；**本题不测**，留作选做板卡项 |

## 二、导出配置与产物

```bash
./.venv/bin/yolo export model=yolo11n.pt format=onnx imgsz=416 dynamic=False simplify=False
```

导出日志（`export-onnx/export.log`）记录的实际环境与结果：

| 项 | 值 |
|---|---|
| Ultralytics | 8.3.232 |
| Python / PyTorch | 3.12.3 / **2.6.0+cpu** |
| 设备 | CPU（11th Gen Intel Core i7-1165G7 @ 2.80GHz） |
| 导出输入形状 | `(1, 3, 416, 416)` BCHW |
| 输出形状 | `(1, 84, 3549)` |
| ONNX 版本 / opset | onnx 1.19.1 / **opset 19** |
| 导出耗时 | 3.6 s |
| 产物 | `export-onnx/yolo11n-416.onnx`，10,604,541 字节（约 10.1 MB） |

**输入尺寸确认**（`onnx-input-shape.txt`，由 ONNX Runtime 读取）：

```
name: images
shape: [1, 3, 416, 416]
type: tensor(float)
```

无动态维度，是固定形状。

## 三、双后端逐图对照

入口：`./.venv/bin/python compare_onnx.py --onnx … --output …`（仓库提供的双后端尺寸核对入口）。

两侧实际输入张量（由 PyTorch forward pre-hook 与 ONNX Runtime 分别记录）：

| 侧 | 记录 |
|---|---|
| PyTorch | `[[1,3,416,416], [1,3,416,416], [1,3,416,416]]` |
| ONNX Runtime | `[1, 3, 416, 416]` |

逐图结果：

| 帧 | 人工情况 | PyTorch 框（原图坐标） | ONNX 框（原图坐标） | 类别 | 置信度 PT / ONNX | 置信度差 | 配对 IoU | 空结果一致 |
|---|---|---|---|---|---|---|---|---|
| **100** | 有球，完整可见 | `(140.13, 391.16, 260.04, 516.37)` | `(140.13, 391.16, 260.04, 516.37)` | sports ball / sports ball | 0.638959 / 0.638959 | **1.19e-07** | **1.000000** | — |
| **150** | 有球，贴上边缘出画 | 空 | 空 | — | — | — | — | ✅ 两侧都为空 |
| **200** | 无球 | 空 | 空 | — | — | — | — | ✅ 两侧都为空 |

## 四、切换条件判定

| 判据（题面） | 实测 | 是否满足 |
|---|---|---|
| 两边框按同类一一配对 | 第 100 帧 1↔1，类别同为 `sports ball`；150/200 帧双方均为 0 | ✅ |
| 配对框 IoU ≥ 0.95 | **1.000000** | ✅ |
| 置信度差 ≤ 0.01 | **1.19e-07** | ✅ |
| 空结果一致 | 150、200 两侧都是空列表 | ✅ |

**结论：达到切换后端的条件。** 在开发机、416 固定输入、0.10 阈值、同一 letterbox 预处理下，ONNX Runtime 与 PyTorch 的框坐标**逐位相同**，置信度差异在 1e-7 量级，空结果完全一致。

## 五、差异解释（题面要求：能解释任何增加、消失或偏移的框）

**本次没有出现增加、消失或偏移的框。**

- **没有偏移**：第 100 帧两边的 `xyxy` 四个数值完全相同（到浮点打印精度），IoU = 1.0。
- **没有消失/增加**：三张图两侧的框数量一一相等（1↔1、0↔0、0↔0）。
- **为什么差异这么小**：ONNX 只是把同一张 PyTorch 计算图静态化后交给另一个执行引擎，权重与算子一致；唯一的差别来自浮点累加顺序与个别算子的实现细节，因此只有 1e-7 量级的置信度漂移，不足以改变 NMS 结果。
- **若将来出现差异，排查顺序**应为：输入缩放与补边是否一致 → 坐标还原（除以缩放比、减去 padding）是否正确 → 数值精度（FP16/INT8 量化） → 后处理（NMS 的 IoU 阈值与 `max_det`）。

## 六、本次对照的信息量（诚实说明）

三张图里**只有第 100 帧产生了框**，另两张都是空结果。因此这次对照实际验证了两件事：

1. 有框时，两边框能精确配对（IoU = 1.0、置信度差 1e-7）；
2. 无框时，两边都为空、不会一侧多出一侧少一（`rt-150` 是边缘球、`rt-200` 是无球帧）。

但**"多目标、多个框、部分重叠"的场景没有被覆盖**——`rt-100` 只有 1 个球。这是本批素材的局限，若要更充分验证，应补充多球或密集场景的图片。

## 七、复跑

```bash
cd /home/QYQ/work/rt-basketball-detection-lab

# 1) 导出固定 416 的 ONNX
./.venv/bin/yolo export model=yolo11n.pt format=onnx imgsz=416 dynamic=False simplify=False \
  2>&1 | tee submission/task06/export-onnx/export.log
mv yolo11n.onnx submission/task06/export-onnx/yolo11n-416.onnx

# 2) 记录输入形状
./.venv/bin/python -c "import onnxruntime as ort; i=ort.InferenceSession('submission/task06/export-onnx/yolo11n-416.onnx').get_inputs()[0]; print(i.name, i.shape, i.type)" \
  | tee submission/task06/onnx-input-shape.txt

# 3) 双后端对照
./.venv/bin/python compare_onnx.py --onnx submission/task06/export-onnx/yolo11n-416.onnx \
  --output submission/task06/backend-comparison.json
```

## 八、文件清单

| 文件 | 内容 |
|---|---|
| `export-protocol.md` | 导出前写定的口径（4 条） |
| `export-onnx/export.log` | 导出日志（版本、输入形状、opset、耗时） |
| `export-onnx/yolo11n-416.onnx` | 固定 416×416 的 ONNX 模型（`.gitignore` 忽略 `*.onnx`，不随分支提交） |
| `onnx-input-shape.txt` | ONNX 实际输入形状记录 |
| `backend-comparison.json` | 双后端逐图对照原始结果 |
| `README.md` | 本文件 |
