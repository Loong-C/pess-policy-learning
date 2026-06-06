from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


T_VALUES = [500, 1000, 2000, 5000]
SVG_COORD_TOLERANCE = 0.05
BLACK = "19.999695%,19.999695%,19.999695%"
GRID = "92.156982%,92.156982%,92.156982%"
PATH_RE = re.compile(r"<path\s+style=\"(?P<style>.*?)\"\s+d=\"(?P<d>.*?)\"\s*/>", re.DOTALL)
POINT_RE = re.compile(r"[ML]\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class FigureConfig:
    stem: str
    figure: int
    rows: list
    columns: list
    methods: list[str]
    y_ticks: list[list[float]]
    line_width: str
    setting: int | None = None
    ribbon_opacity: str | None = None


CONFIGS = {
    "figure5": FigureConfig(
        stem="syn_dt_with_lin",
        figure=5,
        rows=[1, 2, 3],
        columns=["0.2", "0.4", "0.6", "0.8", "pure"],
        methods=[
            "greedy",
            "pess_0.0001",
            "pess_0.001",
            "pess_0.01",
            "pess_0.1",
            "pess_0.2",
            "pess_0.5",
            "pess_1",
            "pess_5",
            "pess_10",
            "lin_0.0001",
            "lin_0.001",
            "lin_0.01",
            "lin_0.1",
            "lin_0.2",
            "lin_0.5",
            "lin_1",
            "lin_5",
            "lin_10",
        ],
        y_ticks=[
            [1.25, 1.50, 1.75, 2.00],
            [1.2, 1.4, 1.6, 1.8],
            [0.9, 1.2, 1.5],
        ],
        line_width="0.85",
    ),
    "figure6": FigureConfig(
        stem="syn_lin",
        figure=6,
        rows=[10, 100],
        columns=["0.2", "0.5", "0.8", "pure"],
        methods=["greedy", "pess_0.1", "pess_0.2", "pess_0.5", "pess_1", "pess_5", "pess_10"],
        y_ticks=[[1.7, 1.8, 1.9, 2.0, 2.1]] * 2,
        line_width="0.85",
        setting=1,
        ribbon_opacity="0.051",
    ),
    "figure7": FigureConfig(
        stem="syn_opt",
        figure=7,
        rows=[10, 100],
        columns=["0.2", "0.5", "0.8", "pure"],
        methods=["greedy", "pess_0.1", "pess_0.2", "pess_0.5", "pess_1", "pess_5", "pess_10"],
        y_ticks=[[1.4, 1.5, 1.6, 1.7, 1.8, 1.9]] * 2,
        line_width="0.85",
        setting=2,
        ribbon_opacity="0.051",
    ),
    "figure8": FigureConfig(
        stem="syn_miss",
        figure=8,
        rows=[10, 100],
        columns=["0.2", "0.5", "0.8", "pure"],
        methods=[
            "greedy",
            "pess_0.001",
            "pess_0.01",
            "pess_0.1",
            "pess_0.2",
            "pess_0.5",
            "pess_1",
            "pess_5",
            "pess_10",
        ],
        y_ticks=[[1.4, 1.5, 1.6, 1.7, 1.8]] * 2,
        line_width="0.85",
        setting=3,
        ribbon_opacity="0.051",
    ),
    "figure9": FigureConfig(
        stem="syn_cv",
        figure=9,
        rows=[10],
        columns=["0.2", "0.5", "0.8", "pure"],
        methods=[
            "greedy",
            "pess_0.1",
            "pess_0.2",
            "pess_0.5",
            "pess_1",
            "pess_5",
            "pess_10",
            "CV_pess",
        ],
        y_ticks=[[1.4, 1.5, 1.6, 1.7, 1.8, 1.9]],
        line_width="1.28",
        setting=2,
    ),
}


def _paths(svg_text: str):
    for match in PATH_RE.finditer(svg_text):
        points = [(float(x), float(y)) for x, y in POINT_RE.findall(match.group("d"))]
        if points:
            yield match.group("style"), points


def _rectangles(paths):
    rectangles = []
    for style, points in paths:
        if len(points) != 5 or BLACK not in style or "stroke-width:1.07" not in style:
            continue
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        rectangle = (min(x_values), max(x_values), min(y_values), max(y_values))
        if rectangle[1] - rectangle[0] > 50 and rectangle[3] - rectangle[2] > 30:
            rectangles.append(rectangle)
    return sorted(set(rectangles), key=lambda rect: (-(rect[2] + rect[3]) / 2.0, (rect[0] + rect[1]) / 2.0))


def _inside_x(rectangle, x):
    return rectangle[0] - SVG_COORD_TOLERANCE <= x <= rectangle[1] + SVG_COORD_TOLERANCE


def _inside_panel(rectangle, point):
    return (
        _inside_x(rectangle, point[0])
        and rectangle[2] - SVG_COORD_TOLERANCE
        <= point[1]
        <= rectangle[3] + SVG_COORD_TOLERANCE
    )


def _method_parts(label: str):
    if label == "greedy":
        return "greedy", 0.0
    if label == "CV_pess":
        return "CV_pess", np.nan
    method, beta = label.split("_", 1)
    return method, float(beta)


def extract_figure(svg_path: Path, config: FigureConfig) -> pd.DataFrame:
    paths = list(_paths(svg_path.read_text(encoding="utf-8")))
    rectangles = _rectangles(paths)
    expected_panels = len(config.rows) * len(config.columns)
    if len(rectangles) != expected_panels:
        raise ValueError(f"{svg_path}: expected {expected_panels} panels, found {len(rectangles)}.")

    rows = []
    for panel_idx, rectangle in enumerate(rectangles):
        row_idx = panel_idx // len(config.columns)
        col_idx = panel_idx % len(config.columns)
        mean_paths = []
        grid_y = []
        ribbon_paths = []
        for style, points in paths:
            if (
                len(points) == 2
                and GRID in style
                and "stroke-width:1.07" in style
                and abs(points[0][1] - points[1][1]) < 1e-8
                and _inside_panel(rectangle, points[0])
                and _inside_panel(rectangle, points[1])
            ):
                grid_y.append(points[0][1])
            if (
                len(points) >= 2
                and "fill:none" in style
                and f"stroke-width:{config.line_width}" in style
                and "stroke:rgb" in style
                and BLACK not in style
                and GRID not in style
                and _inside_panel(rectangle, points[0])
            ):
                mean_paths.append(points)
            if (
                config.ribbon_opacity is not None
                and len(points) >= 2 * len(T_VALUES)
                and f"fill-opacity:{config.ribbon_opacity}" in style
                and _inside_x(rectangle, points[0][0])
                and max(point[1] for point in points) >= rectangle[2]
                and min(point[1] for point in points) <= rectangle[3]
            ):
                ribbon_paths.append(points)

        grid_y = sorted(set(grid_y))
        tick_values = config.y_ticks[row_idx]
        if len(grid_y) != len(tick_values):
            raise ValueError(
                f"{svg_path}: panel {panel_idx} has {len(grid_y)} y grids, expected {len(tick_values)}."
            )
        panel_methods = config.methods
        if config.figure == 5 and row_idx == 0 and len(mean_paths) == len(config.methods) - 3:
            # The three smallest PPL penalties are clipped entirely above the
            # plotting area for Setting 1, so the vector PDF contains no paths.
            panel_methods = [config.methods[0], *config.methods[4:]]
        if len(mean_paths) != len(panel_methods):
            raise ValueError(
                f"{svg_path}: panel {panel_idx} has {len(mean_paths)} visible curves, "
                f"expected {len(panel_methods)}."
            )
        y_slope, y_intercept = np.polyfit(np.asarray(grid_y), np.asarray(tick_values), 1)
        use_ribbons = len(ribbon_paths) == len(panel_methods)

        for method_idx, (method_label, points) in enumerate(zip(panel_methods, mean_paths)):
            method, beta = _method_parts(method_label)
            if len(points) != len(T_VALUES):
                raise ValueError(
                    f"{svg_path}: {method_label} in panel {panel_idx} has {len(points)} points, expected 4."
                )
            paper_se = [0.0] * len(T_VALUES)
            if use_ribbons:
                ribbon = ribbon_paths[method_idx]
                first = ribbon[: len(T_VALUES)]
                second = list(reversed(ribbon[len(T_VALUES) : 2 * len(T_VALUES)]))
                paper_se = [
                    abs(first_point[1] - second_point[1]) * abs(y_slope) / 4.0
                    for first_point, second_point in zip(first, second)
                ]
            for t_value, point, se_value in zip(T_VALUES, points, paper_se):
                record = {
                    "figure": config.figure,
                    "T": t_value,
                    "method": method,
                    "beta": beta,
                    "paper_mean": point[1] * y_slope + y_intercept,
                    "paper_se": se_value,
                    "source": f"arXiv:2212.09900 Figure {config.figure} vector PDF",
                }
                if config.figure == 5:
                    record.update({"scenario": config.rows[row_idx], "decay": config.columns[col_idx]})
                else:
                    record.update(
                        {
                            "setting": config.setting,
                            "batch_size": config.rows[row_idx],
                            "floor": config.columns[col_idx],
                        }
                    )
                rows.append(record)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract Figures 5-9 curve values from arXiv vector SVGs.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--figure", choices=[*CONFIGS, "all"], default="all")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/reproduction/paper_reference"),
    )
    args = parser.parse_args()

    selected = CONFIGS if args.figure == "all" else {args.figure: CONFIGS[args.figure]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, config in selected.items():
        result = extract_figure(args.source_dir / f"{config.stem}.svg", config)
        output = args.out_dir / f"{name}.csv"
        result.to_csv(output, index=False)
        print(f"wrote {output} ({len(result)} rows)")


if __name__ == "__main__":
    main()
