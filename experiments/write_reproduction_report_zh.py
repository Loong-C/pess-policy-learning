from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_MARGINS = {
    4: 0.0025,
    5: 0.05,
    6: 0.05,
    7: 0.05,
    8: 0.05,
    9: 0.05,
    10: 0.02,
}


def _as_int(value) -> str:
    if pd.isna(value):
        return "-"
    return str(int(value))


def _as_percent(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{100.0 * float(value):.1f}%"


def _primary_rows(summary: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for figure, margin in PRIMARY_MARGINS.items():
        rows = summary[
            (summary["figure"] == figure)
            & np.isclose(summary["margin"], margin)
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one primary row for Figure {figure}, got {len(rows)}."
            )
        selected.append(rows.iloc[0])
    return pd.DataFrame(selected)


def _figure_verdict(row: pd.Series) -> str:
    figure = int(row["figure"])
    if figure == 10:
        if bool(row.get("paired_mean_equivalent", False)):
            return "数据集聚类后的平均差异等价"
        return "未证明数据集聚类后的平均差异等价"
    if bool(row["all_cells_equivalent"]):
        return "全部匹配单元等价"
    equivalent = int(row["equivalent_cells"])
    different = int(row["different_cells"])
    inconclusive = int(row["inconclusive_cells"])
    if different > 0:
        return (
            f"未成功：{different} 个单元明确不同，"
            f"{inconclusive} 个证据不足"
        )
    return f"未完全证明：{equivalent} 个等价，{inconclusive} 个证据不足"


def _overall_verdict(primary: pd.DataFrame) -> str:
    synthetic = primary[primary["figure"].between(4, 9)]
    all_synthetic = bool(synthetic["all_cells_equivalent"].fillna(False).all())
    real = primary[primary["figure"] == 10].iloc[0]
    real_equivalent = bool(real.get("paired_mean_equivalent", False))
    if all_synthetic and real_equivalent:
        return (
            "在预先声明的等价界和显著性水平下，所有主要图均通过等价性判定，"
            "可认为本次复现成功。"
        )
    failed = [
        str(int(row["figure"]))
        for _, row in primary.iterrows()
        if (
            int(row["figure"]) < 10
            and not bool(row["all_cells_equivalent"])
        )
        or (
            int(row["figure"]) == 10
            and not bool(row.get("paired_mean_equivalent", False))
        )
    ]
    return (
        "不能在预先声明的标准下把整组实验认定为成功复现。"
        f"未通过完整等价性判定的图为 Figure {', '.join(failed)}。"
    )


def _results_table(primary: pd.DataFrame) -> str:
    lines = [
        "| 图 | 等价界 | 匹配/论文单元 | 等价 | 明确不同 | 证据不足 | MAE | 判定 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in primary.iterrows():
        figure = int(row["figure"])
        if figure == 10:
            equivalent = (
                "是" if bool(row.get("paired_mean_equivalent", False)) else "否"
            )
            cluster_count = _as_int(row.get("paired_cluster_count", np.nan))
            status = (
                f"聚类均值等价={equivalent}，独立数据集数={cluster_count}"
            )
            eq = diff = inconclusive = "-"
        else:
            status = _figure_verdict(row)
            eq = _as_int(row["equivalent_cells"])
            diff = _as_int(row["different_cells"])
            inconclusive = _as_int(row["inconclusive_cells"])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(figure),
                    f"{float(row['margin']):g}",
                    (
                        f"{_as_int(row['matched_cells'])}/"
                        f"{_as_int(row['reference_cells'])}"
                    ),
                    eq,
                    diff,
                    inconclusive,
                    f"{float(row['mae']):.5f}",
                    status,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _sensitivity_table(summary: pd.DataFrame) -> str:
    lines = [
        "| 图 | 等价界 | 等价比例 | 明确不同 | 证据不足 | 最大绝对差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary[summary["figure"].between(4, 9)].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["figure"])),
                    f"{float(row['margin']):g}",
                    _as_percent(row["equivalent_rate"]),
                    _as_int(row["different_cells"]),
                    _as_int(row["inconclusive_cells"]),
                    f"{float(row['max_abs_diff']):.5f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(
    root: Path,
    reference_root: Path,
    output: Path,
) -> None:
    summary_path = root / "data" / "equivalence_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}; run analyze_full_reproduction.py first."
        )
    summary = pd.read_csv(summary_path).sort_values(["figure", "margin"])
    primary = _primary_rows(summary)
    verdict = _overall_verdict(primary)

    real_skipped = root / "data" / "real_skipped.csv"
    skipped_text = "无。"
    if real_skipped.exists():
        skipped = pd.read_csv(real_skipped)
        if not skipped.empty:
            details = "; ".join(
                f"{row.dataset}: {row.reason}"
                for row in skipped.itertuples(index=False)
            )
            skipped_text = details

    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    report = f"""# Pessimistic Policy Learning 第 7 节复现报告

生成时间：{generated}

## 最终结论

**{verdict}**

这里的“相同”不是传统差异检验中“未拒绝零差异”，而是双单侧等价性检验
（TOST）主动证明差异落在预先声明的容忍范围内。显著性水平固定为
`alpha=0.05`；主要等价界为 Figure 4 的 `0.0025`、Figure 5-9 的
`0.05`、Figure 10 的 `0.02`。更宽或更窄界限只作为敏感性分析，
不用于事后替换主要结论。

## 主要结果

{_results_table(primary)}

Figure 4-9 的判定单位是论文图中的方法/参数/样本量/实验设置单元。
“等价”要求差异的 90% 置信区间完全位于等价界内；“明确不同”要求
差异的 95% 置信区间完全位于等价界外，其余归为“证据不足”。
Figure 10 的多个面板共享同一数据集，故总体 TOST 先在数据集内平均，
再以数据集作为独立聚类单位，避免把相关面板点误当成独立样本。

## 敏感性分析

{_sensitivity_table(summary)}

## 实验范围

- Figure 4：5 臂 MAB，论文规模 `N=1000`。
- Figure 5：3 个情景、5 个 overlap 衰减设置、4 个样本量、论文/发布资产中的全部可见方法。
- Figure 6-8：3 个 TS 情景、2 个 batch size、4 个探索下界设置、4 个样本量、每单元 `N=200`。
- Figure 9：Setting 2、batch size 10、固定 beta 与 5-fold CV，`N=200`。
- Figure 10：发布图的 32 数据集顺序、2 个 batch size、4 个探索设置。
- 论文数值参考来自 arXiv 矢量图提取，目录为 `{reference_root.as_posix()}`。

## 论文与公开代码的关键不一致

1. 论文第 7 节写明合成实验噪声标准差为 `0.1`，发布 Figure 5-9
   脚本却依赖 DGP 默认值 `sigma=1`。本报告的 `published` 协议使用
   发布路径的 `sigma=1`；`paper-spec` 协议保留论文文字的 `0.1`。
2. 论文展示的 Algorithm 1/2 与发布 MM/CV 实现不完全相同，两条路径
   已在代码中分离，避免把“忠实论文公式”和“忠实发布数值实现”混称。
3. 发布 TS 收集器没有把首批确定性动作写回返回向量，留下未初始化内存。
   维护版修复该 bug；强制使用上游模块、仅补这一行的 50 次哨兵与维护版
   一致，但 Figure 6 仍存在残余数值差异。
4. Figure 5 图注写 6 个 PPL beta，矢量图和发布脚本实际包含 9 个。
5. Figure 10 论文文字列 33 个数据集，发布脚本和图实际使用 32 个并遗漏
   `skin-segmentation`。
6. 图注把阴影称为标准差，但矢量宽度与发布聚合尺度更接近标准误。本报告
   将提取到的带宽按发布数值资产的标准误尺度用于比较，并在结论中保留这一
   来源不确定性。

## 数据集异常

{skipped_text}

## 结果资产

- 原始汇总数据：`{(root / 'data').as_posix()}`
- 复现图：`{(root / 'figures').as_posix()}`
- 等价性汇总：`{summary_path.as_posix()}`
- 逐单元比较：`{(root / 'data').as_posix()}/figure*_comparison_*.csv`
- Figure 10 数据集聚类统计：`{(root / 'data' / 'figure10_panel_statistics.csv').as_posix()}`

## 可复核性说明

所有运行使用固定根种子 `20260605`，分片任务使用稳定哈希派生独立种子。
运行器对缺失分片执行 fail-fast 检查；只有请求的全部分片存在时才生成汇总
CSV。完整协议回归测试位于 `tests/test_protocols.py`。运行配置 JSON、
提取后的论文参考、汇总 CSV、统计比较表、图和本报告均纳入 Git；体积较大
且可恢复的逐重复分片与日志被 `.gitignore` 排除。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the final Chinese Section 7 reproduction report."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/reproduction/published/full"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("artifacts/reproduction/paper_reference"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reproduction/reproduction_report_zh.md"),
    )
    args = parser.parse_args()
    write_report(args.root, args.reference_root, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
