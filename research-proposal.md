# 固定码长下，前缀内的语义邻近性会改善碰撞商品的生成式推荐吗？

## 研究问题

基于语义 ID 的生成式推荐将商品表示成离散 token 序列，由 Transformer 根据用户的交互历史生成下一件商品的 ID。多个商品映射到相同的前三位时，需要用第四位区分。本文拟检验，在固定前三位、第四位词表大小和生成器的条件下，利用前缀内的文本近邻分配第四位，能否改善碰撞商品的推荐排序。

TIGER 提供整体框架。后续研究已经比较末位分配策略，也研究了按前缀自适应分支。本项目的拟做贡献是隔离并验证**前缀内语义邻近性对末位分配的影响**，重点分析效果随碰撞组大小和商品频次如何变化。方法的新颖性以与 ZCR 及其他近邻方案的进一步比较为准。

## 参考论文

| 论文 | 与本项目的关系 |
| --- | --- |
| [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)，NeurIPS 2023 | TIGER 原论文。用 RQ-VAE 得到前三位语义码，以第四位消歧，使用约 13M 参数的 encoder–decoder Transformer 生成 ID。 |
| [Don’t Waste the Last Token: A Quantizer-Agnostic Quality Boost from Semantic ID Collision Resolution](https://recsys.acm.org/recsys26/contributions/)，RecSys 2026 | 固定码长下比较多种消歧策略。可复用作者的 [sid-collisions 实现](https://github.com/sb-ai-lab/sid-collisions)。 |
| [Faithful Evaluation of Semantic-ID Tokenizers for Generative Recommendation](https://arxiv.org/abs/2605.25330)，CIKM 2026 | 给出碰撞感知的商品级评估和最小代价末位重分配 ZCR，是评估与算法基线。 |
| [Not All Branches Are Equal: Adaptive Semantic ID Construction for Generative Recommendation](https://arxiv.org/abs/2611.11549)，RecSys 2026 | 使用自适应分支构造语义 ID，帮助界定本项目与重建整棵编码树的研究差异。 |

## 数据与实现

- 主数据集用 [Amazon Reviews 2014](https://jmcauley.ucsd.edu/data/amazon/) 的 Beauty。TIGER 附录 C 报告 22,363 个用户、12,101 件商品，按平均序列长度估算约 19.8 万次交互。最终记录去重、过滤和切分后的实际规模。
- 首选 `sid-collisions` 处理数据、编码和训练。冻结 MiniLM，以标题和描述生成商品向量，复用前三级残差 K-means，并固定同一份全局时间切分。商品频次和分配策略的调参只使用相应的训练或验证数据。
- [NonameUntitled/tiger](https://github.com/NonameUntitled/tiger) 自带 Beauty 的预处理数据、向量和训练入口，可用于快速跑通 TIGER 风格基线。该第三方复现约 5M 参数，和原论文配置有差异。对应的 [Zenodo 数据包](https://zenodo.org/records/17351848) 为 2.1 GB，也可用于补充 Sports 和 Toys。主实验优先沿用同一套生成器和评估实现。

## 实验设计

1. **数据诊断。** 统计前三位相同的商品组数量、大小分布、码字容量和测试目标覆盖率。若最大组超出末位词表容量，先统一调整词表或前三位量化配置，再锁定设定进行比较。
2. **方案与对照。** 对比顺序编号、训练期全局流行度分配、ZCR，以及拟做的前缀内近邻分配。候选实现为在组内建立文本近邻图，联合考虑商品对末位码字的匹配代价与近邻分配的一致性，同时保证组内一对一。详细目标函数与消融在实现后确定。固定前三位后重新分配的 ZCR 应标明为受控版本，并记录与作者实现的配置差异。
3. **控制变量。** 各组共享数据切分、前三位码字、末位词表、生成器结构、训练预算和解码配置。每组单独训练并用三个随机种子报告均值和标准差。近邻项权重设为零可控制该项自身的贡献。
4. **评估。** 将生成结果映射回唯一商品 ID，报告全量及碰撞商品子集的 NDCG@10、Hit@10，再按碰撞组大小和训练期商品频次分组。比较有效码字利用率和完整验证集解码耗时。固定前三位时，前三位碰撞率用于描述样本，末位唯一性用于验证实现。
5. **实验顺序。** 先在 Beauty 跑通顺序编号和全局流行度基线，并确认碰撞商品样本量。完成主实验后，若时间允许，在 Toys 上复验最强基线和拟做方法。

## 资源与工期预算

| 项目 | 规划值 |
| --- | --- |
| 存储与主存 | 单类目初始预留 20 GB 磁盘与 16 GB 主存。约 1.2 万个 384 维 FP32 文本向量本身约 18 MiB。 |
| 预处理 | 下载、依赖安装、清洗和首次文本编码预留 0.5–2 小时墙钟时间。各组共享缓存。 |
| 单次训练 | 轻量配置先按 2–5 万步、验证早停估算。若每步 0.1–0.3 秒，纯训练约 0.6–4.2 A800 卡时，含验证和 beam search 暂按 1–6 卡时。 |
| 主实验 | 4 个配置 × 3 个随机种子，共 12 次训练，暂按 12–72 A800 卡时。近邻项消融和第二类目另计。 |
| GPU 使用 | 单张 40 GB A800 起步。可用 2–4 张卡独立并行不同配置或种子。 |

以上是排期估算，尚未在 A800 实测。`sid-collisions` 的公开配置为 8 层、隐藏维度 512，实际耗时需单独标定。正式批跑前，分别计时 500–1,000 个训练步和一次完整验证集 beam search，并用实测更新卡时。TIGER 原论文对 Beauty 使用 batch size 256 和 20 万训练步，可作为复现规格参考。

## 课程交付

提交 4–8 页电子技术报告，可采用课件推荐的 CVPR 格式。实验代码随报告提交，附 `readme.txt` 说明环境、数据准备、参数和复现实验命令。
