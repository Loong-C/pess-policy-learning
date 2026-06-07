from __future__ import annotations

import argparse
import json
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


def _has_complete_coverage(primary: pd.DataFrame) -> bool:
    if "coverage_complete" in primary:
        return bool(primary["coverage_complete"].fillna(False).all())
    if {"matched_cells", "reference_cells"}.issubset(primary.columns):
        return bool(
            (primary["matched_cells"] == primary["reference_cells"]).all()
        )
    return True


def _figure_verdict(row: pd.Series) -> str:
    figure = int(row["figure"])
    clustered = bool(row.get("clustered_mean_equivalent", False))
    if figure == 10:
        return (
            "数据集聚类后的平均差异等价"
            if clustered
            else "未证明数据集聚类后的平均差异等价"
        )
    if bool(row["all_cells_equivalent"]):
        return "全部匹配单元等价，聚类均值等价"
    equivalent = int(row["equivalent_cells"])
    different = int(row["different_cells"])
    inconclusive = int(row["inconclusive_cells"])
    prefix = "聚类均值等价；" if clustered else "聚类均值未证明等价；"
    if different:
        return (
            prefix
            + f"{different} 个单元明确不同，{inconclusive} 个证据不足"
        )
    return (
        prefix
        + f"{equivalent} 个单元等价，{inconclusive} 个证据不足"
    )


def _overall_verdict(primary: pd.DataFrame) -> str:
    synthetic = primary[primary["figure"].between(4, 9)]
    all_synthetic = bool(
        synthetic["all_cells_equivalent"].fillna(False).all()
    )
    all_clustered = bool(
        primary["clustered_mean_equivalent"].fillna(False).all()
    )
    if not _has_complete_coverage(primary):
        if all_synthetic and all_clustered:
            return (
                "已覆盖实验单元在预先声明的标准下通过等价性判定，但实验范围"
                "没有覆盖论文全部单元，因此只能认定为截断范围内复现成功，"
                "不能认定论文第 7 节已完整复现。"
            )
        if all_clustered:
            return (
                "已覆盖实验单元支持总体平均层面复现，但既存在逐点不等价或"
                "证据不足，也缺少论文中的部分实验单元，不能认定为完整复现。"
            )
        return (
            "已覆盖实验单元未全部通过总体等价性判定，且实验范围不完整，"
            "不能认定论文第 7 节成功复现。"
        )
    if all_synthetic and all_clustered:
        return (
            "在预先声明的等价界和显著性水平下，所有主要图均通过等价性"
            "判定，可认定为本次复现成功。"
        )
    if all_clustered:
        return (
            "所有主要图的聚类平均差异均通过等价性检验，但部分逐单元结果"
            "未等价，因此只能认定为总体平均层面复现，不能认定为完整逐点复现。"
        )
    failed = [
        str(int(row["figure"]))
        for _, row in primary.iterrows()
        if not bool(row.get("clustered_mean_equivalent", False))
    ]
    return (
        "不能在预先声明的标准下把整组实验认定为成功复现。"
        f"未通过聚类平均等价性判定的图为 Figure {', '.join(failed)}。"
    )


def _results_table(primary: pd.DataFrame) -> str:
    lines = [
        "| 图 | 等价界 | 匹配/论文单元 | 覆盖完整 | 聚类均值等价 | 等价 | 明确不同 | 证据不足 | MAE | 判定 |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in primary.iterrows():
        figure = int(row["figure"])
        if figure == 10:
            eq = different = inconclusive = "-"
        else:
            eq = _as_int(row["equivalent_cells"])
            different = _as_int(row["different_cells"])
            inconclusive = _as_int(row["inconclusive_cells"])
        coverage = bool(
            row.get(
                "coverage_complete",
                row["matched_cells"] == row["reference_cells"],
            )
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(figure),
                    f"{float(row['margin']):g}",
                    f"{_as_int(row['matched_cells'])}/{_as_int(row['reference_cells'])}",
                    "是" if coverage else "否",
                    (
                        "是"
                        if bool(row.get("clustered_mean_equivalent", False))
                        else "否"
                    ),
                    eq,
                    different,
                    inconclusive,
                    f"{float(row['mae']):.5f}",
                    _figure_verdict(row),
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


def _excluded_t_values(root: Path) -> list[int]:
    excluded = set()
    for name in ("tree", "ts", "ts_cv"):
        path = root / f"run_config_{name}.json"
        if not path.exists():
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(config.get("excluded_contextual_t_values", []))
    return sorted(int(value) for value in excluded)


def _observed_t_values(root: Path) -> list[int]:
    observed = set()
    for name in (
        "contextual_nonadaptive_results.csv",
        "ts_synthetic_results.csv",
        "ts_cv_results.csv",
    ):
        path = root / "data" / name
        if path.exists():
            frame = pd.read_csv(path, usecols=["T"])
            observed.update(frame["T"].dropna().astype(int).tolist())
    return sorted(observed)


def _skipped_datasets(root: Path) -> str:
    path = root / "data" / "real_skipped.csv"
    if not path.exists():
        return "无。"
    skipped = pd.read_csv(path)
    if skipped.empty:
        return "无。"
    return "\n".join(
        f"- `{row.dataset}`：{row.reason}"
        for row in skipped.itertuples(index=False)
    )


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
    excluded = _excluded_t_values(root)
    observed = _observed_t_values(root)
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    if excluded:
        scope_note = (
            f"本轮按用户要求排除合成实验样本规模 `T={excluded}`。"
            f"实际纳入统计的样本规模为 `T={observed}`。因此，下文的逐点"
            "结论只适用于这些已运行样本规模，不能外推到被排除的条件。"
        )
    else:
        scope_note = (
            f"合成实验实际纳入统计的样本规模为 `T={observed}`，"
            "未主动排除论文样本规模。"
        )

    report = f"""# Pessimistic Policy Learning 第 7 节复现报告

生成时间：{generated}

## 最终结论

**{verdict}**

{scope_note}

这里的“结果一致”采用双单侧等价性检验（TOST），不是把“未拒绝零差异”
误当作相同。显著性水平固定为 `alpha=0.05`；主要等价界为 Figure 4 的
`0.0025`、Figure 5-9 的 `0.05`、Figure 10 的 `0.02`。其余界限仅用于
敏感性分析，不用于事后替换主要结论。

## 主要结果

{_results_table(primary)}

Figure 4-9 的逐点判定单位是论文图中的方法、参数、样本量和实验设置单元。
“等价”要求均值差异的 90% 置信区间完全位于等价界内；“明确不同”要求
95% 置信区间完全位于等价界外；其余归为“证据不足”。总体 TOST 先在共享
同一模拟数据的配置内对方法差异取平均，再以实验配置作为独立聚类单位。
Figure 10 先在数据集内平均，再以数据集为独立聚类单位。

## 敏感性分析

{_sensitivity_table(summary)}

## 实验范围

- Figure 4：3 个 MAB 设置，论文规模 `N=1000`。
- Figure 5：3 个情景、5 个 overlap 衰减设置及公开资产中的全部可见方法。
- Figure 6-8：3 个 TS 情景、2 个 batch size、4 个探索下界设置，每单元 `N=200`。
- Figure 9：Setting 2、batch size 10、固定 beta 与 5-fold CV，每单元 `N=200`。
- Figure 10：发布图中的 32 个数据集、2 个 batch size、4 个探索设置。
- 论文数值参考来自 arXiv 矢量图提取：`{reference_root.as_posix()}`。

## 论文与公开代码的不一致

1. 论文第 7 节写明合成实验噪声标准差为 `0.1`，发布 Figure 5-9 脚本依赖
   DGP 默认值 `sigma=1`。本报告的 `published` 协议采用发布路径的 `sigma=1`。
2. 论文展示的 Algorithm 1/2 与发布 MM/CV 实现不完全相同；代码将
   `paper-spec` 与 `published` 两条路径分开。
3. 发布 TS 收集器未把首批确定性动作写入返回向量，留下未初始化内存；
   维护版修复了该问题。
4. Figure 5 图注称有 6 个 PPL beta，矢量图和发布脚本实际包含 9 个。
5. Figure 10 文字称有 33 个数据集，发布脚本与图实际使用 32 个，遗漏
   `skin-segmentation`。
6. 图注将阴影称为标准差，但矢量带宽与发布聚合尺度更接近标准误。该来源
   不确定性保留在结论中。

## 数据集异常

{_skipped_datasets(root)}

## 结果资产

- 汇总数据：`{(root / 'data').as_posix()}`
- 复现图：`{(root / 'figures').as_posix()}`
- 等价性汇总：`{summary_path.as_posix()}`
- 逐单元比较：`{(root / 'data').as_posix()}/figure*_comparison_*.csv`
- Figure 10 聚类统计：`{(root / 'data' / 'figure10_panel_statistics.csv').as_posix()}`

## 可复核性

所有运行使用固定根种子 `20260605`，分片任务使用稳定哈希派生独立种子。
运行器对缺失分片执行 fail-fast 检查，并只汇总当前任务网格内的分片。
协议回归测试位于 `tests/test_protocols.py`。运行配置、论文参考、汇总 CSV、
统计比较表、图和本报告纳入 Git；体积较大且可恢复的逐重复分片和日志由
`.gitignore` 排除。
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
