# 复现实验报告

本报告由 `experiments/reproduce.py` 生成。`mean/std/count` 分别是重复实验均值、经验标准差和样本数；`ci95_low/ci95_high` 使用正态近似给出均值的 95% 置信区间。真实数据部分的 `p_value_gt0` 是对 `PPL - GPL > 0` 的单侧 t 检验 p 值。快速模式样本数很小，只用于烟测；论文级结论应以 `--mode full` 的 N=200/N=1000 设置为准。

### figure4_equivalence_margin_0.001.csv

rows: 168, columns: setting_name, T, method, ours_mean, ours_sd, ours_count, ours_se, paper_mean, paper_se, paper_count, digitization_resolution, source, diff, combined_se, z, ci95_low, ci95_high, p_gt_minus_margin, p_lt_plus_margin, tost_p, equivalent, consistent_by_ci




### figure4_equivalence_margin_0.0025.csv

rows: 168, columns: setting_name, T, method, ours_mean, ours_sd, ours_count, ours_se, paper_mean, paper_se, paper_count, digitization_resolution, source, diff, combined_se, z, ci95_low, ci95_high, p_gt_minus_margin, p_lt_plus_margin, tost_p, equivalent, consistent_by_ci




### figure4_equivalence_margin_0.005.csv

rows: 168, columns: setting_name, T, method, ours_mean, ours_sd, ours_count, ours_se, paper_mean, paper_se, paper_count, digitization_resolution, source, diff, combined_se, z, ci95_low, ci95_high, p_gt_minus_margin, p_lt_plus_margin, tost_p, equivalent, consistent_by_ci




### mab_results.csv

rows: 357000, columns: experiment, setting, setting_name, T, rep, method, action, reward_abs_mu, rescaled_subopt, correct


| setting_name   | method   |   mean |    std |   count |     se |   ci95_low |   ci95_high |
|:---------------|:---------|-------:|-------:|--------:|-------:|-----------:|------------:|
| Optimal        | clip_0.1 | 0.0292 | 0.0249 |    7000 | 0.0003 |     0.0286 |      0.0298 |
| Optimal        | clip_0.2 | 0.0297 | 0.0249 |    7000 | 0.0003 |     0.0291 |      0.0303 |
| Optimal        | clip_0.5 | 0.0314 | 0.0246 |    7000 | 0.0003 |     0.0309 |      0.0320 |
| Optimal        | clip_1   | 0.0348 | 0.0235 |    7000 | 0.0003 |     0.0342 |      0.0353 |
| Optimal        | clip_10  | 0.0495 | 0.0082 |    7000 | 0.0001 |     0.0493 |      0.0497 |
| Optimal        | clip_15  | 0.0495 | 0.0082 |    7000 | 0.0001 |     0.0493 |      0.0497 |
| Optimal        | clip_2   | 0.0421 | 0.0189 |    7000 | 0.0002 |     0.0416 |      0.0425 |
| Optimal        | clip_5   | 0.0490 | 0.0092 |    7000 | 0.0001 |     0.0488 |      0.0492 |
| Optimal        | greedy   | 0.0458 | 0.0150 |    7000 | 0.0002 |     0.0454 |      0.0461 |
| Optimal        | pess_0.1 | 0.0294 | 0.0248 |    7000 | 0.0003 |     0.0288 |      0.0300 |
| Optimal        | pess_0.2 | 0.0134 | 0.0221 |    7000 | 0.0003 |     0.0129 |      0.0139 |
| Optimal        | pess_0.5 | 0.0055 | 0.0153 |    7000 | 0.0002 |     0.0051 |      0.0058 |
| Optimal        | pess_1   | 0.0052 | 0.0149 |    7000 | 0.0002 |     0.0048 |      0.0055 |
| Optimal        | pess_10  | 0.0051 | 0.0148 |    7000 | 0.0002 |     0.0047 |      0.0054 |
| Optimal        | pess_15  | 0.0051 | 0.0148 |    7000 | 0.0002 |     0.0047 |      0.0054 |
| Optimal        | pess_2   | 0.0051 | 0.0148 |    7000 | 0.0002 |     0.0047 |      0.0054 |
| Optimal        | pess_5   | 0.0051 | 0.0148 |    7000 | 0.0002 |     0.0047 |      0.0054 |
| Suboptimal     | clip_0.1 | 0.0228 | 0.0225 |    7000 | 0.0003 |     0.0223 |      0.0233 |
| Suboptimal     | clip_0.2 | 0.0235 | 0.0227 |    7000 | 0.0003 |     0.0230 |      0.0240 |
| Suboptimal     | clip_0.5 | 0.0261 | 0.0232 |    7000 | 0.0003 |     0.0255 |      0.0266 |
| Suboptimal     | clip_1   | 0.0315 | 0.0235 |    7000 | 0.0003 |     0.0309 |      0.0320 |
| Suboptimal     | clip_10  | 0.0571 | 0.0073 |    7000 | 0.0001 |     0.0570 |      0.0573 |
| Suboptimal     | clip_15  | 0.0574 | 0.0071 |    7000 | 0.0001 |     0.0573 |      0.0576 |
| Suboptimal     | clip_2   | 0.0418 | 0.0214 |    7000 | 0.0003 |     0.0413 |      0.0423 |
| Suboptimal     | clip_5   | 0.0551 | 0.0101 |    7000 | 0.0001 |     0.0548 |      0.0553 |
| Suboptimal     | greedy   | 0.0399 | 0.0218 |    7000 | 0.0003 |     0.0394 |      0.0404 |
| Suboptimal     | pess_0.1 | 0.0271 | 0.0228 |    7000 | 0.0003 |     0.0266 |      0.0276 |
| Suboptimal     | pess_0.2 | 0.0178 | 0.0180 |    7000 | 0.0002 |     0.0174 |      0.0182 |
| Suboptimal     | pess_0.5 | 0.0131 | 0.0115 |    7000 | 0.0001 |     0.0128 |      0.0133 |
| Suboptimal     | pess_1   | 0.0129 | 0.0112 |    7000 | 0.0001 |     0.0126 |      0.0132 |
| Suboptimal     | pess_10  | 0.0129 | 0.0112 |    7000 | 0.0001 |     0.0126 |      0.0131 |
| Suboptimal     | pess_15  | 0.0129 | 0.0112 |    7000 | 0.0001 |     0.0126 |      0.0131 |
| Suboptimal     | pess_2   | 0.0129 | 0.0112 |    7000 | 0.0001 |     0.0126 |      0.0131 |
| Suboptimal     | pess_5   | 0.0129 | 0.0112 |    7000 | 0.0001 |     0.0126 |      0.0131 |
| Uniform        | clip_0.1 | 0.0308 | 0.0220 |    7000 | 0.0003 |     0.0303 |      0.0313 |
| Uniform        | clip_0.2 | 0.0308 | 0.0220 |    7000 | 0.0003 |     0.0302 |      0.0313 |
| Uniform        | clip_0.5 | 0.0308 | 0.0220 |    7000 | 0.0003 |     0.0303 |      0.0313 |
| Uniform        | clip_1   | 0.0308 | 0.0219 |    7000 | 0.0003 |     0.0303 |      0.0313 |
| Uniform        | clip_10  | 0.0313 | 0.0219 |    7000 | 0.0003 |     0.0307 |      0.0318 |
| Uniform        | clip_15  | 0.0315 | 0.0220 |    7000 | 0.0003 |     0.0310 |      0.0320 |
| Uniform        | clip_2   | 0.0309 | 0.0220 |    7000 | 0.0003 |     0.0304 |      0.0314 |
| Uniform        | clip_5   | 0.0310 | 0.0219 |    7000 | 0.0003 |     0.0305 |      0.0315 |
| Uniform        | greedy   | 0.0307 | 0.0220 |    7000 | 0.0003 |     0.0302 |      0.0312 |
| Uniform        | pess_0.1 | 0.0307 | 0.0219 |    7000 | 0.0003 |     0.0302 |      0.0313 |
| Uniform        | pess_0.2 | 0.0308 | 0.0219 |    7000 | 0.0003 |     0.0303 |      0.0313 |
| Uniform        | pess_0.5 | 0.0308 | 0.0219 |    7000 | 0.0003 |     0.0302 |      0.0313 |
| Uniform        | pess_1   | 0.0308 | 0.0219 |    7000 | 0.0003 |     0.0302 |      0.0313 |
| Uniform        | pess_10  | 0.0325 | 0.0220 |    7000 | 0.0003 |     0.0320 |      0.0330 |
| Uniform        | pess_15  | 0.0328 | 0.0219 |    7000 | 0.0003 |     0.0322 |      0.0333 |
| Uniform        | pess_2   | 0.0312 | 0.0219 |    7000 | 0.0003 |     0.0307 |      0.0317 |
| Uniform        | pess_5   | 0.0319 | 0.0220 |    7000 | 0.0003 |     0.0314 |      0.0325 |

