# PPL 复现实验结果报告

生成时间：2026-06-05，环境：`pess-pl-legacy`。

## 复现成功判定

结论：**目前不能在统计显著性意义上声称已经成功复现论文结果。**

原因不是代码无法运行，而是统计判定所需条件尚未满足：

1. 除 MAB 外，Figure 5-10 尚未完成论文规模 full 运行；quick 结果只用于烟测，样本数和数据集规模被刻意缩小，不能用于正式复现判定。
2. 对“我们的结果是否与论文结果一致”做显著性判断，需要论文中的数值均值/标准误，或从图中 digitize 得到的数值，并预先设定等效性容忍界 `delta`。仅凭论文图片和当前自动报告，不能做严格的等效性检验。
3. 已完成的 full MAB 只能支持**部分、定性复现**：小样本下 PPL 相比 GPL 有改善，方向与论文一致；但在 `T=20000` 处，PPL 相对 GPL 的提升很小，Suboptimal 的单侧检验 p 值约 0.221，Uniform 约 0.432，不能说在最终点显著优于 GPL。

更严谨的成功标准应是：对每个论文图中的曲线点，比较 `ours_mean - paper_mean`，并做双单侧等效性检验（TOST）。若绝大多数关键点在预设误差界 `[-delta, delta]` 内等效，且论文主要方向性结论（例如 PPL > GPL 的场景）也在 full 结果中成立，才可以说“统计意义上成功复现”。本仓库现在新增了 `experiments/compare_to_paper.py`，用于在拿到论文数值参考后执行这个检验。

## 运行范围

本次修复后，代码已按论文第 6、7 节重建核心算法与实验入口，并完成以下运行：

- `full/mab`：按论文第 7.1.1 的 N=1000、T 网格运行五臂 bandit 实验，生成 Figure 4、Figure 11、Figure 12 对应数据与图。
- `quick/all`：运行 MAB、Figure 5 非自适应上下文实验、Figure 6-9 TS 合成实验和一个真实数据烟测子集，验证完整代码链路。

未完成的论文全量部分：Figure 5-10 的 full 网格没有在本次交互中全部跑完。原因是 full 设置包含大量 R `policytree` 拟合：非自适应树实验约十万级树搜索，TS 合成实验和 33 个 OpenML 数据集的 5-fold CV 也会进一步放大计算量。一次真实数据 quick 原始完整子集运行 20 分钟仍未结束，因此本报告将 full 结论限制在 MAB，其他部分只解释为功能性烟测和趋势核对。

另一个真实数据注意点：当前 OpenML 中 `houses` 的默认目标 `median_house_value` 有 3842 个唯一值，不符合论文“分类数据集、每个类别作为一个 arm”的假设。代码现在会在真实数据 full 模式中跳过超过 50 个类别的数据集，并把原因写到 `real_skipped.csv`，避免静默构造几千臂 bandit。

## 主要资产

- full MAB 数据：`artifacts/reproduction/full/data/mab_results.csv`
- full MAB 图：`artifacts/reproduction/full/figures/figure4_mab.png`、`figure11_mab_clip.png`、`figure12_mab_frequency.png`
- quick 全链路数据：`artifacts/reproduction/quick/data/*.csv`
- quick 全链路图：`artifacts/reproduction/quick/figures/*.png`
- 自动统计表：`artifacts/reproduction/full/report_zh.md`、`artifacts/reproduction/quick/report_zh.md`

## 与论文的对应关系

### MAB full

MAB 使用论文指定的 3 个行为策略、`mu/sqrt(T)` 缩放、`sigma=0.1`、N=1000。整体趋势与论文一致：在小样本下，PPL 的方差惩罚能降低 rescaled suboptimality；均匀重叠场景中，PPL 与 GPL 接近。

在 `T=100` 时：

| setting | GPL mean | best PPL | best PPL mean | improvement |
| --- | ---: | --- | ---: | ---: |
| Optimal | 0.03959 | pess_0.2 | 0.03125 | 0.00834 |
| Suboptimal | 0.03414 | pess_0.1 | 0.02743 | 0.00671 |
| Uniform | 0.03102 | pess_0.1 | 0.03099 | 0.00003 |

在 `T=20000` 时，三种 setting 的差异都很小；对每个重复实验做 GPL-bestPPL 的单侧 t 检验，Suboptimal 的 p 值约 0.221，Uniform 约 0.432，不能认为最终点有显著提升。这和论文图中“主要收益出现在缺乏均匀 overlap 的有限样本区间”并不冲突，但本次 full MAB 的最终点优势弱于论文视觉叙述。

### Figure 5 quick

非自适应上下文实验 quick 只跑 N=1 的缩小网格，因此只作为代码链路检查。结果趋势：

- Setting 1：GPL 平均 1.4107，best PPL 1.4260，PPL 优于 GPL；linear PEVI 低于两者。
- Setting 2：GPL 平均 1.2311，best PPL 1.2444，PPL 小幅优于 GPL。
- Setting 3：GPL 平均 0.9709，best PPL 0.9668，linear PEVI 1.0403；这与论文中 worst overlap 下线性方法更稳定的叙述一致。

### Figure 6-9 quick

TS 合成实验 quick 同样只用于烟测。三个 setting 中，best PPL 均高于 GPL：

| setting | GPL mean | best PPL mean |
| --- | ---: | ---: |
| 1 | 1.3010 | 1.3256 |
| 2 | 1.2720 | 1.2930 |
| 3 | 1.2308 | 1.2339 |

CV quick 中，`CV_pess` 平均 1.2761，GPL 平均 1.2574，方向与论文 Figure 9 一致。但样本数只有 8 个组合，不足以做论文级显著性声明。

### Real data quick

真实数据 quick 使用 `cmc` 一个数据集、batch size 10、两种探索设置、短训练/评估切片。PPL-GPL 平均提升为 0.00583；由于只有 2 个点，单侧 t 检验 p 值约 0.25，统计上不显著。它只证明真实数据入口、OpenML 缓存、TS 收集、Algorithm 2 CV 和评估代码可以跑通。

## 统计说明

自动报告中所有均值均给出经验标准差、样本量和正态近似 95% 置信区间。真实数据报告额外给出 `PPL - GPL > 0` 的单侧 t 检验 p 值。对 quick 模式，置信区间和 p 值不应被解释为论文结论，因为 quick 模式有意缩小了重复次数、样本量或数据集数量。

## 结论

代码层面已经修复到可以按论文设定运行，且 quick 全链路和 full MAB 均成功生成数据、图表和报告。当前结果对论文主要结论给出部分支持：MAB 小样本和 TS/上下文 quick 中 PPL 通常优于 GPL，worst overlap 下线性方法更稳定。但这仍然只是**部分定性复现**，不是正式统计复现。正式结论需要完成 Figure 5-10 的 full 运行，并用论文数值参考做等效性检验；同时当前 OpenML 的 `houses` 元数据与论文分类假设不一致，需要在真实数据复现中明确处理或替换。
