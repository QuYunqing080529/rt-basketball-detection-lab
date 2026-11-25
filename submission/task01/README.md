# Task 01 — 回放、单图检测与水平翻转对照

仓库：`/home/QYQ/work/rt-basketball-detection-lab`（分支 `learn/neural-basics`），解释器 `./.venv/bin/python`（Python 3.12.3，torch 2.6.0+cpu，torchvision 0.21.0+cpu，Ultralytics 8.3.232，OpenCV 4.11.0，NumPy 1.26.4，ONNX 1.19.1，ONNX Runtime 1.32.0）。

## 复跑命令

```bash
cd /home/QYQ/work/rt-basketball-detection-lab

# 0) 数据准备（本机离线方案，原因见「数据来源」）
./.venv/bin/python tools/prepare_subset.py rt

# 1) 回放 rt.mp4 的 0–225 帧（imgsz=640, conf=0.10），逐帧写 jsonl
./.venv/bin/python lab.py replay --source data-src/rt.mp4 --limit 226 \
  --imgsz 640 --conf 0.10 --output submission/task01/replay-000-225-conf010.jsonl

# 2) 单图检测 + 水平翻转对照（一次产出 rt-100 与 rt-225 的全部结果）
./.venv/bin/python submission/task01/detect_cases.py

# 3) 由上一步的 rt-225-centers.json 生成中心点对照合图
./.venv/bin/python submission/task01/make_center_sheet.py
```

`detect_cases.py` 的推理调用与 `lab.py` 完全同形：

```python
model.predict(image, classes=[index], imgsz=640, conf=..., device="cpu", verbose=False)
```

`index` 由模型 `names` 中值为 `"sports ball"` 的键**动态取得**（COCO 中为 32），不写死。画框用 `cv2.rectangle`，框旁标注置信度。

## 数据来源

`dataset-curated/` 由 `./.venv/bin/python tools/prepare_subset.py rt` 生成，而不是由 `python lab.py prepare` 生成。

原因：`lab.py prepare` 按 `annotations.csv` 的行序遍历视频来源，第一条就是 `plos01`；本机无法访问 `upload.wikimedia.org`——**IPv4 直连被重置（`curl: (35) Recv failure: Connection was reset`），IPv6 返回 `429 Too many requests`**，两条路都不通，`prepare` 在这个来源上即中断，排在最后面的 `rt` 一帧都不会被写出（`dataset-curated/` 甚至不会被创建）。`tools/prepare_subset.py rt` 复用 `lab.py` 的 `read_annotations` / `get_asset` / `write_sample`，只处理 `rt` 一个来源，输出与完整 `prepare` 对 `rt` 的结果字节一致，因此 14 张 val 帧可离线生成。

权重 `yolo11n.pt` 与视频 `data-src/rt.mp4` 均按 `sources.json` 的 SHA-256 校验通过（`0ebbc80d…7644ee1` / `f244ae13…aff171a2`）。

## 结果

### rt-100 首次推理（imgsz=640, conf=0.25, sports ball）

| 项 | 值 |
|---|---|
| 预测框 `xyxy` | `[139.41, 390.48, 260.47, 516.74]` |
| 置信度 | `0.512907` |
| 人工标注框 `xyxy` | `[136, 395, 255, 515]` |
| 与人工框 IoU | ≈ **0.88**（≥ 0.50，视为匹配） |

带框图：`rt-100-boxed.jpg`；完整记录：`rt-100-first-inference.json`。

### 回放第 200 / 225 帧（imgsz=640, conf=0.10）

| 帧号 | time_s | 宽×高 | 检测数 | 框与置信度 |
|---|---|---|---|---|
| 200 | 3.3325 | 360×640 | 0 | —（`detections` 为空列表，与题面预期一致） |
| 225 | 3.7491 | 360×640 | 1 | `[24.64, 156.35, 132.91, 265.46]` @ `0.1937` |

完整逐帧记录：`replay-000-225-conf010.jsonl`（226 行，帧号 0–225 连续无缺）。

### rt-225 原图 vs 水平翻转图（imgsz=640, conf=0.10）

| 输入 | 最高分框中心点 (x, y)/px | 置信度 | 框 `xyxy` |
|---|---|---|---|
| 原图 `rt-225.jpg` | (78.69, 210.83) | `0.251331` | `[24.39, 156.21, 133.00, 265.45]` |
| 水平翻转 `cv2.flip(img, 1)` | (281.92, 211.28) | `0.280921` | `[227.55, 156.48, 336.28, 266.08]` |

镜像期望中心：`360 − 78.69 = 281.31`，实测 `281.92`，**误差 0.61 px**；球从画面左侧（x≈79）换到右侧（x≈282）。翻转图以 PNG 保存：`rt-225-flip.png`（360×640 RGB）。数据取自 `rt-225-centers.json`。

中心点对照合图见 `rt-225-centers-compare.png`（左＝原图、右＝翻转图，红色十字为最高分框中心，底部标注镜像期望值与误差）。

## 思考题

### a) 同一卷积核怎样在新位置提取局部线索

卷积层每个输出只读取输入的一个小窗口（**局部连接**），整张图共用同一组权重（**权重共享**，这组权重就是卷积核）。球从画面一侧移到另一侧时，窗口内“球与背景交界”的像素排列没有变，变的只是这个窗口在图上的坐标，所以同一个核在新位置给出同样的响应，各位置的响应排成特征图，后续层再把这些局部线索组合起来定位目标。

需要补一句边界：上述结论只对**平移**成立，不能推广到镜像。镜像会改变窗口内像素的左右排列，而卷积核是有方向性的，理论上响应会变。但本次实测显示，rt-225 水平翻转后仍然检出篮球，置信度 `0.280921` 还略高于原图的 `0.251331`。原因是篮球接近**各向同性**——圆形轮廓、左右近似对称的黑色纹路，镜像后局部特征几乎没变，核照样响应；若换成文字、球衣号码、箭头这类强方向性目标，镜像后分数会明显下降。所以准确的说法是：**卷积核对镜像的敏感程度取决于目标本身有没有方向性纹理**，不能由“方向敏感”直接外推成“无法识别镜像”。

### b) 两张有球图的置信度是否一样

**不一样，但差异很小。** 原图最高分框置信度 `0.251331`（中心 (78.69, 210.83)），水平翻转图 `0.280921`（中心 (281.92, 211.28)），翻转侧高 `0.0296`，相对高约 `11.8%`。两图中心点满足水平镜像关系（期望 281.31，实测 281.92，差 0.61 px），说明两次检测落在同一个球上。

**怎么读这个差异**：0.03 的浮动与 JPEG 重编码、letterbox 缩放插值、候选框排序带来的正常抖动同量级，不足以说明镜像让检测变难或变易；只能说“镜像没有明显破坏检出”。真正会显著移动分数的是输入管线——同一帧 225 经 `lab.py replay`（视频解码）得 `0.1937`，经 `rt-225.jpg`（q92 JPEG）得 `0.251331`，相差 `0.058`。这也是题面要求单图对照统一使用 `rt-225.jpg` 的原因：**对照实验里输入管线必须锁死，一次只改一个变量。**

## 口径提醒

`replay` 那条记录读的是视频解码帧，`detect_cases.py` 读的是 q92 JPEG，两条管线不同，因此**第 225 帧在回放里是 `0.1937`、在单图对照里是 `0.251331`，两者不可直接相比较**。本任务的翻转对照严格固定在 `rt-225.jpg` 这一条管线上完成。

## 提交文件

| 文件 | 内容 |
|---|---|
| `env-check.txt` | 环境核验（解释器路径、8 项依赖版本、cv2 路径、**PyTorch 使用 CPU 的核验**、yolo 路径与版本） |
| `rt-100-boxed.jpg` | rt-100 带框图 |
| `rt-100-first-inference.json` | 首次推理记录（权重、设备、参数、框、复跑命令） |
| `replay-000-225-conf010.jsonl` | 0–225 帧逐帧回放记录 |
| `replay-200-225.md` | 200 / 225 两帧的人可读摘要 |
| `rt-225-flip.png` | 水平翻转图（PNG） |
| `rt-225-boxed.jpg` / `rt-225-flip-boxed.jpg` | 原图 / 翻转图带框图 |
| `rt-225-centers.json` | 两图最高分框中心点与置信度 |
| `rt-225-centers-compare.png` | 原图 / 翻转图中心点对照合图 |
| `detect_cases.py` | 单图与翻转对照的复跑入口 |
| `make_center_sheet.py` | 中心点对照合图的复跑入口 |
| `README.md` | 本文件 |

## 问卷对应（任务一）

| 问卷题 | 交什么 |
|---|---|
| 3 环境核验记录 | `env-check.txt`（或同内容的终端截图） |
| 4 第一次 YOLO11n 推理 | `rt-100-boxed.jpg` + `rt-100-first-inference.json` |
| 5 第 225 帧中心点对照图 | `rt-225-centers-compare.png` |
| 6 第 200、225 帧回放记录 | `replay-200-225.md`（可另附 `replay-000-225-conf010.jsonl`） |
| 7 卷积与置信度观察 | 本文件「思考题」a、b 两段全文 |
