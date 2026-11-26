# Task 04 — 冻结前 10 层 vs 全量微调

仓库：`/home/QYQ/work/rt-basketball-detection-lab`（分支 `learn/neural-basics`），解释器 `./.venv/bin/python`。

## 一、数据与口径

| 项 | 内容 |
|---|---|
| 训练集 | `dataset-curated/images/train` **33 张**（PLOS 六段视频抽帧，`plos01/02/03/06/07/09`） |
| 验证集 | `dataset-curated/images/val` **14 张**（RT 视频抽帧，12 有球 + 2 无球） |
| 起点权重 | `yolo11n.pt`（两组**都是**这一份，第二组不是从第一组的 `best.pt` 起） |
| 训练设置 | CPU、imgsz 416、5 轮、batch 8、seed 42、deterministic、AdamW lr0=0.001、mosaic/mixup 0、scale 0.20 |
| 最终评测口径 | CPU、**imgsz 640**、**conf 0.25**、IoU ≥ 0.50、`split=val` |

> 说明：`lab.py train` 把 `epochs=15` 写死且没有 `freeze` 参数，无法满足“两组各 5 轮 + 冻结前 10 层”。因此本题按第三章示范的参数形式直接调用 Ultralytics CLI，未修改仓库脚本。

## 二、两组配置（`args.yaml` 差异）

`diff full-ft/args.yaml freeze10-ft/args.yaml` 的结果只有三行：

```
name: full-ft            → name: freeze10-ft
freeze: null             → freeze: 10
save_dir: …/full-ft      → save_dir: …/freeze10-ft
```

其余参数（`model`、`data`、`epochs`、`imgsz`、`batch`、`optimizer`、`lr0`、`seed`、`mosaic`、`mixup`、`scale`…）**逐项相同**。题目要求的“改动集中在冻结范围”成立。

## 三、冻结是否真的生效

### 3.1 训练日志里的证据

| 组 | 日志内容 |
|---|---|
| 两组共同 | `Overriding model.yaml nc=80 with nc=1` —— 检测头按本任务的单类别重建，**这是参数量口径的前提** |
| 两组共同 | `Freezing layer 'model.23.dfl.conv.weight'` —— Ultralytics 对 DFL 末层的内置冻结，**只涉及 16 个参数**，与本题的 `freeze` 无关 |
| 冻结前 10 层 | 共 **100 行** `Freezing layer 'model.0.…'` 到 `'model.9.…'`，覆盖 backbone 前 10 层 |
| 全量微调 | 只有上面那一行 `model.23.dfl.conv.weight`，前 10 层未被冻结 |

### 3.2 与预训练权重的逐张量比对

预训练权重是 80 类，本任务模型是 1 类，检测头形状不同（51 个张量无法逐位比较）；其余 **448 个张量形状一致**：

| 组 | 同形状未变 | 同形状但已变 | 检测头（80→1 类，形状不同） |
|---|---|---|---|
| 全量微调 | 97 | **351** | 51 |
| 冻结前 10 层 | **255** | 193 | 51 |

按层看：**冻结组 `model.0`–`model.9` 的权重全部未变（合计 255 个张量，变化数 0）**，`model.10` 之后发生变化；全量组 `model.0`–`model.9` 全部发生变化。**这是冻结生效最硬的证据。**

### 3.3 可训练参数量（本次训练口径）

本次模型（`nc=1`）共 **2,590,035** 个参数。预训练 `yolo11n.pt` 是 `nc=80`、2,624,080 个参数，两者之差来自检测头重建，**不能拿预训练模型的参数量当作本次训练的口径**。

| 组 | 算术 | 可训练参数 | 占总参数 |
|---|---|---|---|
| 全量微调 | 2,590,035 − 16（两组共有的固定 DFL 参数 `model.23.dfl.conv.weight`） | **2,590,019** | 99.999% |
| 冻结前 10 层 | 2,590,019 − 1,115,744（`model.0`–`model.9`） | **1,474,275** | 56.9% |

冻结组被冻结的参数为 16 + 1,115,744 = **1,115,760**（占 43.1%）。

## 四、训练期间指标（`results.csv`，5 轮）

两组都是 5 行记录，`metrics/mAP50(B)` 与 `metrics/mAP50-95(B)` **第 1–5 轮全部为 0**。

| 组 | 末轮 train/cls_loss | 末轮 val/cls_loss | 末轮 mAP50 | 末轮 mAP50-95 |
|---|---|---|---|---|
| 全量微调 | 1.59166 | 4.62927 | 0 | 0 |
| 冻结前 10 层 | 1.57900 | 4.52537 | 0 | 0 |

**这两列数值是训练期在 416 输入、训练器自带验证流程下算出来的，不能与第五节的帧级匹配混谈。** 训练损失在下降、验证损失在上升，但 mAP 始终为 0，说明 5 轮内没有学出可用的检测结果。

## 五、最终评测（640 / 0.25，IoU ≥ 0.50）

| 组 | 有球帧匹配 | 边缘球帧匹配 | 无球帧误报 | 14 帧内检测总数 |
|---|---|---|---|---|
| 全量微调 | **0 / 12** | 0 / 2 | 0 / 2 | **0** |
| 冻结前 10 层 | **0 / 12** | 0 / 2 | 0 / 2 | **0** |

两组在 14 帧上**一个框都没有输出**（`max_conf = 0`），因此也不存在“有框但 IoU 不够”的情况。

### 与预训练基线对比

| 权重 | 有球匹配 | 边缘球 | 无球误报 |
|---|---|---|---|
| 预训练 `yolo11n.pt`（640/0.25） | **3 / 12** | 0 / 2 | 0 / 2 |
| 全量微调 5 轮 | 0 / 12 | 0 / 2 | 0 / 2 |
| 冻结前 10 层 5 轮 | 0 / 12 | 0 / 2 | 0 / 2 |

## 六、文字说明

**① 哪些参数根据梯度更新。**
两次训练都从同一个 `yolo11n.pt` 起步，检测头按本任务重建为 `nc=1`，因此**本次模型共 2,590,035 个参数**（不是预训练模型的 2,624,080）。其中 `model.23.dfl.conv.weight` 的 **16 个参数**由 Ultralytics 固定冻结、两组都不训练。

- **全量微调**：可训练参数 **2,590,019** 个。实测与预训练权重相比，同形状的 448 个张量中有 **351 个发生变化**，且 `model.0`–`model.9` 全部变化。
- **冻结前 10 层**：在 2,590,019 的基础上再扣除 `model.0`–`model.9` 的 **1,115,744** 个，剩 **1,474,275** 个可训练参数（占 56.9%），被冻结 1,115,760 个（43.1%）。实测这 10 层与预训练权重**逐张量完全相同（255 个张量、变化数 0）**，`model.10` 之后才发生变化。

**② 训练集与验证集各自的作用。**
训练集（33 张，PLOS 六段视频）用来计算损失并反向传播更新参数；验证集（14 张 RT 帧）在训练过程中只用于计算 mAP、挑选权重，**不参与梯度更新**。需要注意训练期指标（416 输入下的 mAP）与最终评测（640 输入、IoU ≥ 0.5 的帧级匹配）是两套口径，本文件分表呈现，未混用。

**③ 现有结果支持继续尝试哪一组。**
**暂时无法判断，且两组都劣于预训练基线。** 两组在最终评测里都是 0/12、14 帧零检测，完全一致，没有任何指标能区分它们；而预训练基线是 3/12。因此现有结果既不能区分两组，也不支持用任一微调权重复代预训练权重。

## 七、复跑

```bash
cd /home/QYQ/work/rt-basketball-detection-lab

# 第 1 组：全量微调
./.venv/bin/yolo detect train model=yolo11n.pt data=data.yaml \
  device=cpu imgsz=416 epochs=5 batch=8 workers=0 seed=42 \
  deterministic=True optimizer=AdamW lr0=0.001 mosaic=0.0 mixup=0.0 \
  scale=0.20 plots=False project=submission/task04 name=full-ft exist_ok=True

# 第 2 组：冻结前 10 层（唯一区别是 freeze=10）
./.venv/bin/yolo detect train model=yolo11n.pt data=data.yaml \
  device=cpu imgsz=416 epochs=5 batch=8 workers=0 seed=42 freeze=10 \
  deterministic=True optimizer=AdamW lr0=0.001 mosaic=0.0 mixup=0.0 \
  scale=0.20 plots=False project=submission/task04 name=freeze10-ft exist_ok=True

# 两组评测（只换 --weights）
./.venv/bin/python lab.py eval --split val --imgsz 640 --conf 0.25 \
  --weights submission/task04/full-ft/weights/best.pt \
  --output submission/task04/eval-full-640-conf025.jsonl
./.venv/bin/python lab.py eval --split val --imgsz 640 --conf 0.25 \
  --weights submission/task04/freeze10-ft/weights/best.pt \
  --output submission/task04/eval-freeze10-640-conf025.jsonl
```

## 八、文件清单

| 文件 | 内容 |
|---|---|
| `full-ft/args.yaml` / `freeze10-ft/args.yaml` | 两组训练配置（唯一差异为 `freeze`） |
| `full-ft/results.csv` / `freeze10-ft/results.csv` | 每轮记录（各 5 轮） |
| `full-ft-train.log` / `freeze10-ft-train.log` | 训练日志（含 `nc=80 → nc=1` 与冻结层打印） |
| `eval-full-640-conf025.jsonl` / `.summary.json` | 全量微调组的最终评测 |
| `eval-freeze10-640-conf025.jsonl` / `.summary.json` | 冻结组的最终评测 |
| `full-ft/weights/`、`freeze10-ft/weights/` | 权重文件（`.gitignore` 忽略 `*.pt`，不随分支提交） |
| `README.md` | 本文件 |

## 九、复现性复核（额外验证，非题面要求）

用完全相同的参数把两组重训到临时目录，与已提交的产物比较：

| 检查 | 结果 |
|---|---|
| `results.csv` | 除 `time`（墙钟耗时）一列外**逐列完全相同** |
| 权重 | 499/499 个张量**完全一致**，最大元素差 `0.000e+00` |

结论：训练可复现，两组 0/12 的评测结果**不是偶发波动**，而是该设置下的稳定结果。

## 十、关于“为什么两组都未命中”的说明（待验证的解释）

第 5 题那样，这里**只把观察到的现象列出来，不把它们当成已经确定的因果结论**。

**观察到的现象**：

- 训练期 mAP 全程为 0，末轮 train/cls_loss 降到 1.58 左右，val/cls_loss 升到 4.5–4.6（呈过拟合迹象）；
- 最终评测两组都是 0/12、14 帧零检测；
- 用 `conf=1e-6` 探测，最高原始置信度只有 4e-4 量级，说明不是被 0.25 的阈值筛掉。

**一个待验证的解释**：训练素材（PLOS，1920×1080 远景，球折算到 416 输入后约 15 px）与验证素材（RT，360×640 近景，球约 116 px）在**画面尺度与场景**上差异很大，33 张样本可能不足以让模型跨过这个差异。

**为什么现有证据还不足以确认它**：

1. 本次改动同时涉及多个因素——检测头由 `sports ball` 改为单类 `basketball`（原类别被移除）、样本量仅 33 张、关闭了 `mosaic`、`scale` 只允许 ±20%、训练与验证的分辨率和机位都不同；**没有做单变量对照**，无法把退化归到其中任何一个；
2. 补充训练轮数的观察（15 轮、40 轮）只说明“加大轮数不能解决”，并不指向具体原因；
3. “40 轮模型能在训练图上检出、在验证图上检不出”只说明**跨域失败**这一现象存在，但没有排除“类别重映射导致原检测能力被覆盖”等其他解释。

**要确认这个解释，需要**：固定其他条件、一次只改一项做对照，例如①补采与验证域同机位同尺度的训练素材；②保持类别映射不变、只改数据量；③保持数据不变、只开 `mosaic`/`scale` 增加尺度变化。在这些对照完成之前，“画面差异”只能写作**待验证的解释**。

## 十一、补充观察：不同训练轮数（不足以确定原因）

题面要求 5 轮，文档第三章示范为 15 轮。为排除“轮数太少”这一解释，另跑 15 轮与 40 轮（仅改 `epochs`，其余同参数）：

| 轮数 | train/cls_loss | val/cls_loss | 训练期 mAP50 最大值 | 训练图检出 | 验证图检出 | 最终评测 |
|---|---|---|---|---|---|---|
| **5 轮（题面）** | 5.021 → 1.592 | 3.687 → 4.629 | 0 | 0 | 0 | **0 / 12** |
| 15 轮（文档示范） | 5.021 → 1.160 | 3.687 → 5.129 | 0 | 0 | 0 | **0 / 12** |
| 40 轮 | 5.021 → 0.819 | 3.687 → 5.015 | 0 | 1 个（conf 0.64） | 0 | **0 / 12** |

这组观察只能说明：**加大轮数不能解决问题**，40 个 epoch 里训练期 mAP 一次都没有大于 0。它**不能**用来证明“画面差异就是原因”——那需要第十节列出的单变量对照。

> 对照产物在 `/tmp/probe15`、`/tmp/probe40`，不随分支提交；命令见第七节，改 `epochs` 即可复跑。
