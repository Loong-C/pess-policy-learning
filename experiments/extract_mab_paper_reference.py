from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ["greedy", "pess_0.1", "pess_0.2", "pess_0.5", "pess_1", "pess_2", "pess_5", "pess_10"]
SETTINGS = ["Optimal", "Suboptimal", "Uniform"]
T_VALUES = [100, 500, 1000, 2000, 5000, 10000, 20000]
PANELS = [(40.0, 163.0), (168.0, 291.0), (295.0, 418.0)]

# Figure 4 uses common y ticks. These are the SVG coordinates for 0.00 and
# 0.04 after pdftocairo conversion of the arXiv source vector PDF.
Y_ZERO = 40.011719
Y_AT_004 = 128.128906
Y_SCALE = 0.04 / (Y_AT_004 - Y_ZERO)

PATH_RE = re.compile(r"<path\s+style=\"(?P<style>.*?)\"\s+d=\"(?P<d>.*?)\"\s*/>", re.DOTALL)
POINT_RE = re.compile(r"[ML]\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def _paths(svg_text: str):
    for match in PATH_RE.finditer(svg_text):
        points = [(float(x), float(y)) for x, y in POINT_RE.findall(match.group("d"))]
        if points:
            yield match.group("style"), points


def _panel_index(x: float) -> int | None:
    for idx, (left, right) in enumerate(PANELS):
        if left <= x <= right:
            return idx
    return None


def _interpolate(points: list[tuple[float, float]], xs: list[float]) -> np.ndarray:
    points = sorted(points)
    point_x = np.array([point[0] for point in points], dtype=float)
    point_y = np.array([point[1] for point in points], dtype=float)
    return np.interp(np.asarray(xs, dtype=float), point_x, point_y)


def extract_reference(svg_path: Path) -> pd.DataFrame:
    svg_text = svg_path.read_text(encoding="utf-8")
    line_paths = [[] for _ in PANELS]
    ribbons = [[] for _ in PANELS]

    for style, points in _paths(svg_text):
        panel = _panel_index(points[0][0])
        if panel is None:
            continue
        if (
            "fill:none" in style
            and "stroke-width:1.07" in style
            and "stroke:rgb" in style
            and "92.156982%" not in style
            and "100%,100%,100%" not in style
            and "19.999695%,19.999695%,19.999695%" not in style
            and len(points) >= 2
        ):
            line_paths[panel].append(points)
        elif "fill-opacity:0.102" in style and len(points) >= 14:
            ribbons[panel].append(points)

    rows = []
    for panel, setting in enumerate(SETTINGS):
        if len(line_paths[panel]) != len(METHODS):
            raise ValueError(f"Expected {len(METHODS)} mean curves in panel {panel}, found {len(line_paths[panel])}.")
        if len(ribbons[panel]) != len(METHODS):
            raise ValueError(f"Expected {len(METHODS)} ribbons in panel {panel}, found {len(ribbons[panel])}.")

        standard_x = [point[0] for point in ribbons[panel][0][: len(T_VALUES)]]
        for method_idx, method in enumerate(METHODS):
            mean_y = _interpolate(line_paths[panel][method_idx], standard_x)
            ribbon = ribbons[panel][method_idx]
            upper = {round(x, 6): y for x, y in ribbon[: len(T_VALUES)]}
            lower = {round(x, 6): y for x, y in ribbon[len(T_VALUES) : 2 * len(T_VALUES)]}
            for t_value, x, y in zip(T_VALUES, standard_x, mean_y):
                key = round(x, 6)
                if key not in upper or key not in lower:
                    raise ValueError(f"Ribbon coordinates do not align for {setting}, {method}, T={t_value}.")
                band_half_width = abs(lower[key] - upper[key]) * Y_SCALE / 2.0
                rows.append(
                    {
                        "setting_name": setting,
                        "T": t_value,
                        "method": method,
                        "paper_mean": (float(y) - Y_ZERO) * Y_SCALE,
                        # The vector ribbon is mean +/- 2 times the quantity
                        # plotted by the authors. Its scale matches a standard
                        # error, despite the caption calling it an SD.
                        "paper_se": band_half_width / 2.0,
                        "paper_count": 1000,
                        "digitization_resolution": Y_SCALE,
                        "source": "arXiv:2212.09900 Figure 4 vector PDF",
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract Figure 4 means and empirical SDs from its vector SVG.")
    parser.add_argument("--svg", required=True, type=Path, help="SVG made with: pdftocairo -svg MAB_subopt.pdf MAB_subopt.svg")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/reproduction/paper_reference/figure4_mab.csv"),
    )
    args = parser.parse_args()
    result = extract_reference(args.svg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(result)} rows)")


if __name__ == "__main__":
    main()
