# 语义 ID 碰撞消歧的生成式推荐

本项目在 Amazon Reviews 2014 Beauty 数据集上研究四位语义 ID（SID）的最后一位分配。固定前三位和第四级码本后，比较怎样给共享前缀的商品分配唯一的第四位。主实验使用 `256 / 256 / 256 / 64` 码本和每组 **20,000 步**训练预算。

[英文课程报告（PDF）](report/paper.pdf) · [LaTeX 源文件和编译说明](report/README.md)

## 方法简介

1. 按全局时间切分交互数据，使用冻结的 `all-MiniLM-L6-v2` 编码商品标题与描述。
2. 用残差 K-means 生成固定的前三位 SID，并拟合 64 个候选第四级质心。共享前三位的商品组成一个碰撞组。
3. 从固定前缀 ZCR 的一对一最小失真分配出发，在组内用文本向量建立 Top-`k` 近邻图。第四位分配同时考虑商品残差到获配质心的距离，以及近邻商品之间获配质心的加权距离。权重 `λ` 控制两项的取舍。通过换到空码或交换两件商品的码来降低代价，完整 SID 保持唯一。
4. 将固定 SID 用作用户历史输入和下一商品预测目标。Transformer 预测四位 SID，通过目录约束 beam search 返回商品，在全量和碰撞子集上评估 NDCG、Hit Rate。

| 对照方案 | 第四位分配规则 |
| --- | --- |
| `sequential` | 按商品 ID 在组内顺序编号 |
| `global_pop` | 按训练期商品频次映射到全目录分位桶，再在组内分配最近的可用码 |
| `zcr` | 固定前三位，求组内残差失真之和最小的一对一匹配 |
| `neighbors_zero` | 近邻权重置零，分配结果与 ZCR 一致 |
| `neighbors` | 在 ZCR 上加入组内文本近邻项，参考配置为 `λ=0.5, k=3` |

在三个随机种子上，`λ=1.0, k=3` 的碰撞子集测试 NDCG@10 为 **0.02218**，相比 ZCR 的 **0.01822** 提升 **21.7%**。顺序编号的碰撞 Hit@10 和两项 @20 指标最高。

## 安装和冒烟测试

在仓库根目录运行。启动脚本需要 Linux、Bash、`curl` 和 `flock`。以下以 Python 3.10 为例。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

首次运行会从 Stanford SNAP 下载 Beauty 交互与商品元数据，并从 Hugging Face 下载固定版本的 MiniLM 权重。先用 CPU 执行一次 4 步训练，检查下载、数据处理、训练和评估入口。

```bash
DEVICE=cpu THREADS=2 bash scripts/run_experiments.sh smoke sequential 17
```

## 复现主实验

以下以 8 张 GPU 为例。将 `GPUS` 改成实际可用的物理卡编号即可。单卡串行运行时，用 `DEVICE=cuda` 代替 `GPUS=...`。正式实验默认运行种子 `17`、`42`、`73`。

```bash
# 五种方案各训练到 10,000 步。
GPUS=0,1,2,3,4,5,6,7 THREADS=2 bash scripts/run_experiments.sh course_256

# 从各自的 10,000 步状态继续训练到 20,000 步。
GPUS=0,1,2,3,4,5,6,7 THREADS=2 bash scripts/run_experiments.sh course_256_long

# 基于验证集选出的检查点评估 @5、@10、@20。
python -m sidrec.evaluate course_256_long --device cuda --threads 2

# 补跑三个近邻参数设置，并生成调参汇总。
GPUS=0,1,2,3,4,5,6,7 THREADS=2 bash scripts/run_neighbor_sweep.sh
```

训练每 500 步验证一次，优先按碰撞子集 NDCG@10 选择检查点，平局按全量 NDCG@10 决定。调参脚本复用 ZCR 和参考近邻配置，另跑 `(λ, k) = (0.25, 3)`、`(1.0, 3)`、`(0.5, 1)`。

只跑一个主实验方案或种子时，在预设后加上方案名和种子。续训时使用相同的方案与种子。

```bash
DEVICE=cuda THREADS=2 bash scripts/run_experiments.sh course_256 neighbors 17
DEVICE=cuda THREADS=2 bash scripts/run_experiments.sh course_256_long neighbors 17
```

## 输出位置

| 路径 | 内容 |
| --- | --- |
| `data/processed/beauty_256/` | 时间切分、文本向量、码本与各方案的 SID 表 |
| `runs/course_256_long/<方案>/<种子>/` | 检查点、训练 loss、验证过程与测试指标 |
| `runs/course_256_long/ranking_at_k_256.csv` | 全量和碰撞子集的 @5、@10、@20 结果及逐种子分数 |
| `runs/course_256_long/validation_curve.csv` | 按训练步数汇总的验证曲线和训练 loss |
| `runs/neighbor_sweep/ranking_at_k.csv` | 近邻参数对照的验证与测试结果 |
| `runs/neighbor_sweep/assignments.csv` | 相比 ZCR 发生换码的商品数及近邻质心距离 |

数据、权重和运行产物会保存在当前仓库下的 `data/` 和 `runs/`。研究背景见[课程提案](research-proposal.md)。
