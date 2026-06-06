from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_DATASETS = [
    "waveform-5000",
    "Long",
    "cmc",
    "artificial-characters",
    "Click_prediction_small",
    "allrep",
    "mfeat-morphological",
    "satellite_image",
    "jungle_chess_2pcs_endgame_elephant_elephant",
    "wilt",
    "Satellite",
    "ringnorm",
    "mammography",
    "delta_ailerons",
    "PhishingWebsites",
    "splice",
    "pendigits",
    "texture",
    "cardiotocography",
    "volcanoes-d4",
    "volcanoes-b3",
    "dis",
    "optdigits",
    "electricity",
    "kr-vs-kp",
    "bank-marketing",
    "satimage",
    "MagicTelescope",
    "houses",
    "eeg-eye-state",
    "car",
    "segment",
]

BLACK = "19.999695%,19.999695%,19.999695%"
POINT_COLORS = {
    "74.510193%,74.510193%,74.510193%": "zero",
    "0%,0%,100%": "positive",
    "100%,40.783691%,39.607239%": "negative",
}
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
ML_POINT_RE = re.compile(r"[ML]\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def _numbers(path_data: str) -> np.ndarray:
    values = np.asarray([float(value) for value in NUMBER_RE.findall(path_data)])
    if len(values) % 2:
        raise ValueError("SVG path contains an odd number of coordinates.")
    return values.reshape(-1, 2)


def _panel_rectangles(paths: list[tuple[str, str]]) -> list[tuple[float, float, float, float]]:
    rectangles = []
    for style, path_data in paths:
        points = [(float(x), float(y)) for x, y in ML_POINT_RE.findall(path_data)]
        if (
            len(points) != 5
            or BLACK not in style
            or "stroke-width:1.07" not in style
        ):
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        rectangle = (min(xs), max(xs), min(ys), max(ys))
        if rectangle[1] - rectangle[0] > 100 and rectangle[3] - rectangle[2] > 70:
            rectangles.append(rectangle)
    # The PDF paths use a vertical flip transform, so larger source y is the
    # visually upper row.
    return sorted(
        set(rectangles),
        key=lambda rectangle: (
            -(rectangle[2] + rectangle[3]) / 2.0,
            (rectangle[0] + rectangle[1]) / 2.0,
        ),
    )


def _point_color(style: str) -> str | None:
    for rgb, label in POINT_COLORS.items():
        if f"fill:rgb({rgb})" in style and "stroke-width:0.71" in style:
            return label
    return None


def extract_reference(svg_path: Path) -> pd.DataFrame:
    root = ET.parse(svg_path).getroot()
    paths = [
        (element.attrib.get("style", ""), element.attrib.get("d", ""))
        for element in root.iter()
        if element.tag.endswith("path")
    ]
    rectangles = _panel_rectangles(paths)
    if len(rectangles) != 8:
        raise ValueError(f"{svg_path}: expected 8 panels, found {len(rectangles)}.")

    source_ticks = [
        [113.679688, 126.351562, 139.011719, 151.671875, 164.328125, 177.0],
        [32.128906, 44.789062, 57.449219, 70.121094, 82.78125, 95.441406],
    ]
    tick_values = np.arange(0.0, 0.30, 0.05)
    columns = ["pure", "0.8", "0.5", "0.2"]
    batch_sizes = [10, 100]
    rows = []

    for style, path_data in paths:
        color = _point_color(style)
        if color is None:
            continue
        coordinates = _numbers(path_data)
        center_x = float((coordinates[:, 0].min() + coordinates[:, 0].max()) / 2.0)
        center_y = float((coordinates[:, 1].min() + coordinates[:, 1].max()) / 2.0)

        matched_panel = None
        for panel_idx, rectangle in enumerate(rectangles):
            if (
                rectangle[0] <= center_x <= rectangle[1]
                and rectangle[2] <= center_y <= rectangle[3]
            ):
                matched_panel = (panel_idx, rectangle)
                break
        if matched_panel is None:
            raise ValueError(f"Point ({center_x}, {center_y}) is outside all panels.")

        panel_idx, rectangle = matched_panel
        row_idx = panel_idx // 4
        col_idx = panel_idx % 4
        y_slope, y_intercept = np.polyfit(source_ticks[row_idx], tick_values, 1)

        x_margin = 1.75
        x_step = (rectangle[1] - rectangle[0] - 2.0 * x_margin) / (
            len(PAPER_DATASETS) - 1
        )
        dataset_index = int(round((center_x - rectangle[0] - x_margin) / x_step))
        expected_x = rectangle[0] + x_margin + dataset_index * x_step
        if not 0 <= dataset_index < len(PAPER_DATASETS) or abs(center_x - expected_x) > 0.12:
            raise ValueError(
                f"Could not map x={center_x} in panel {panel_idx} to a dataset index."
            )

        paper_improvement = center_y * y_slope + y_intercept
        if color == "positive" and paper_improvement < -1e-4:
            raise ValueError("Positive-colored point has a negative coordinate.")
        if color == "negative" and paper_improvement > 1e-4:
            raise ValueError("Negative-colored point has a positive coordinate.")
        rows.append(
            {
                "figure": 10,
                "dataset_index_paper": dataset_index,
                "dataset": PAPER_DATASETS[dataset_index],
                "batch_size": batch_sizes[row_idx],
                "floor": columns[col_idx],
                "paper_improvement": paper_improvement,
                "point_color": color,
                "source": "arXiv:2212.09900 Figure 10 vector PDF",
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["batch_size", "floor", "dataset_index_paper"]
    )
    duplicate_keys = ["dataset", "batch_size", "floor"]
    if result.duplicated(duplicate_keys).any():
        raise ValueError("Figure 10 extraction produced duplicate dataset cells.")
    return result.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Figure 10 real-data points from the arXiv vector SVG."
    )
    parser.add_argument("--svg", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/reproduction/paper_reference/figure10_real.csv"),
    )
    args = parser.parse_args()

    result = extract_reference(args.svg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(result)} visible points)")
    print(result.groupby(["batch_size", "floor"]).size().to_string())


if __name__ == "__main__":
    main()
