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

## 三、冻结是否真的生效（三重证据）

### 3.1 训练日志里的冻结层

| 组 | 日志中 `Freezing layer` 涉及的层号 |
|---|---|
| 全量微调 | `23`（Ultralytics 对 DFL 末层的内置冻结，与本题无关） |
| 冻结前 10 层 | **0 – 9**，以及 `23`（同上，内置） |

### 3.2 与预训练权重的逐张量比对

| 组 | 变化张量数 | 未变张量数 | `model.0`–`model.9`（backbone） |
|---|---|---|---|
| 全量微调 | **402** | 97 | 全部发生变化 |
| 冻结前 10 层 | 244 | 255 | **全部未变（每层变化数为 0）** |

`model.10` 之后（neck、检测头）两组都发生了变化。**这是冻结生效最硬的证据**：冻结组的 backbone 权重与 `yolo11n.pt` 完全一致。

### 3.3 可训练参数量

| 组 | 可训练参数 | 占比 | 被冻结 |
|---|---|---|---|
| 全量微调 | **2,624,080** | 100% | 0% |
| 冻结前 10 层 | **1,043,424** | 39.8% | **60.2%** |

## 四、训练期间指标（`results.csv`，5 轮）

两组都是 5 行记录，`metrics/mAP50(B)` 与 `metrics/mAP50-95(B)` **第 1–5 轮全部为 0**。

| 组 | 末轮 train/cls_loss | 末轮 val/cls_loss | 末轮 mAP50 | 末轮 mAP50-95 |
|---|---|---|---|---|
| 全量微调 | 1.59166 | 4.62927 | 0 | 0 |
| 冻结前 10 层 | 1.57900 | 4.52537 | 0 | 0 |

**这两列数值是训练期在 416 输入、训练器自带验证流程下算出来的，不能与第五节的帧级匹配混谈。** 训练损失在下降、验证损失在上升（过拟合迹象），但 mAP 始终为 0，说明 5 轮内没有学出可用的检测结果。

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
两次训练都从同一个 `yolo11n.pt` 起步。全量微调中 2,624,080 个参数全部可训练，反向传播会把它们全部更新（实测 402 个张量与预训练不同）。冻结前 10 层的微调中，`model.0`–`model.9`（backbone）被置为不可训练，只有其余 **1,043,424 个参数（39.8%，集中在 neck 与检测头）** 按梯度更新；实测 backbone 权重与预训练**逐张量完全一致**。

**② 训练集与验证集各自的作用。**
训练集（33 张，PLOS 六段视频）用来计算损失并反向传播更新参数；验证集（14 张 RT 帧）在训练过程中只用于计算 mAP、挑选权重，**不参与梯度更新**。需要注意训练期指标（416 输入下的 mAP）与最终评测（640 输入、IoU ≥ 0.5 的帧级匹配）是两套口径，本文件分表呈现，未混用。

**③ 现有结果支持继续尝试哪一组。**
**暂时无法判断，且两组都劣于预训练基线。** 两组在最终评测里都是 0/12、14 帧零检测，完全一致，没有任何指标能区分它们；而预训练基线是 3/12。原因是训练把小数据上的检测头改写成单类 `basketball`，该类在 33 张图 5 轮内尚未学会，同时原有的 `sports ball` 类被移除，于是两类都检不出。要回答“小数据下冻结部分层是否值得继续试”，需要增加数据与轮次，或让冻结组保留原类别做对照后再比较；**本次结果不足以支持这个判断，也不支持用任一微调权重复代预训练权重。**

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
| `full-ft-train.log` / `freeze10-ft-train.log` | 训练日志（含冻结层打印） |
| `eval-full-640-conf025.jsonl` / `.summary.json` | 全量微调组的最终评测 |
| `eval-freeze10-640-conf025.jsonl` / `.summary.json` | 冻结组的最终评测 |
| `full-ft/weights/`、`freeze10-ft/weights/` | 权重文件（**`.gitignore` 忽略 `*.pt`，不随分支提交**；可用第七节命令重新生成） |
| `README.md` | 本文件 |

## 九、复现性与根因复核（额外验证，非题面要求）

“两组都 0 检测”这个结果容易被怀疑是流程问题，因此补做了四项独立验证（产物在 `/tmp/recheck`、`/tmp/probe40`，不随分支提交）。

| 验证 | 做法 | 结果 |
|---|---|---|
| **可复现性** | 用完全相同的参数把两组重训一遍 | `results.csv` 除 `time`（墙钟耗时）一列外**逐列完全相同**；权重 499/499 个张量**完全一致**，最大元素差 `0.000e+00` |
| **是不是阈值把框筛掉了** | 把 `conf` 降到 `1e-6` 探测原始输出 | 两组最高置信度只有 **0.00042 – 0.00050**，中位数 0.00011 – 0.00013。**不是阈值问题，是模型输出本身接近 0** |
| **是不是轮数太少** | 同参数训练 **40 轮** 作对照 | 训练集 `cls_loss` 从 5.02 降到 0.82，但 `val mAP` 仍为 0；40 轮模型在**训练图** `plos03-029` 上能检出（conf 0.64），在**验证图** `rt-100` / `rt-225` 上 0 检测 |
| **结论** | — | 5 轮（33 张图、batch 8，共 25 个优化步）连训练集都没学到；40 轮能拟合训练域，但**跨不到 RT 验证域** |

**根因**：训练素材是 PLOS 的 1920×1080 远景（球折算到 416 输入后约 15 px），验证素材是 RT 的 360×640 竖构图近景（球约 116 px）。**域差异 + 33 张样本量不足**，导致微调后检测头只对训练域响应，对验证域输出接近 0。这与第三章示范“微调后退化为 0/12”的现象一致，属方法本身在小数据下的必然结果，不是本仓库流程的缺陷。

## 十、轮数对照：5 轮（题面）vs 15 轮（文档示范）vs 40 轮（加长）

**为什么要单独对照轮数**：题面要求“两组都训练 5 轮”，而文档第三章示范用的是 **15 轮**（原文：“本示范用 CPU、输入尺寸 416 和 15 轮训练”）。为排除“5 轮太少导致退化”这一解释，另跑两次全量微调作对照，**除 `epochs` 外所有参数与提交的两组完全相同**（CPU、416、batch 8、seed 42、deterministic、AdamW lr0=0.001、mosaic/mixup 0、scale 0.20）。

| 轮数 | train/cls_loss | val/cls_loss | 训练期 mAP50 最大值 | 训练图 `plos03-029` 检出 | 验证图 `rt-100` 检出 | 最高原始置信度（conf=1e-6） | **最终评测（640/0.25）** |
|---|---|---|---|---|---|---|---|
| **5 轮（题面设置）** | 5.021 → 1.592 | 3.687 → 4.629 | **0** | 0 | 0 | 0.000429 | **0 / 12** |
| **15 轮（文档示范）** | 5.021 → 1.160 | 3.687 → 5.129 | **0** | 0 | 0 | 0.000286 | **0 / 12** |
| 40 轮（加长对照） | 5.021 → 0.819 | 3.687 → 5.015 | **0** | **1 个（conf 0.64）** | 0 | 0.640561 | 0 / 12 |

结论：

1. **15 轮（完全复刻文档示范）同样是 0 / 12**，与文档原文“由 2 / 12 降为 0 / 12”一致 → 本实验的 5 轮结果**不是把轮数改小造成的**。
2. **40 个 epoch 里训练期 mAP50 一次都没有大于 0**，但 40 轮的模型能在**训练图**上检出（conf 0.64）、在**验证图**上 0 检出 → 模型确实学到了训练域，失败发生在**跨域**这一步。
3. 因此**退化与轮数无关**；根因是第九节所述的域差异（球在 416 输入下 15 px vs 116 px，约 8 倍）与 33 张样本量不足。若要让微调真正有效，应让训练素材覆盖目标域（同机位、同尺度），或开启 `scale` / `mosaic` 扩大多尺度变化，而不是继续加轮数。

> 对照产物位于 `/tmp/probe15`、`/tmp/probe40`，不随分支提交；命令见第七节，把 `epochs=5` 改成对应轮数即可原样复跑。
