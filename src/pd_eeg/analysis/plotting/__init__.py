"""Render manuscript figures from the distributed aggregate estimates."""

from __future__ import annotations

from copy import deepcopy
import csv
from pathlib import Path


def render_all(output, *, formats=("pdf", "svg", "png"), dpi=600):
    from . import architecture, charts, design, representation

    formats = tuple(formats)
    if not formats or any(ext not in ("pdf", "svg", "png") for ext in formats):
        raise ValueError("Figure formats must be chosen from pdf, svg and png")
    if dpi < 72:
        raise ValueError("PNG resolution must be at least 72 dpi")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    charts.OUT, charts.FORMATS, charts.DPI = output, formats, dpi
    published = charts.load(charts.DATA / "benchmark.json")
    comparison = charts.load(charts.DATA / "comparison_figures.json")
    primary = charts.load(charts.DATA / "primary.json")
    plt = charts.setup()
    rendered = {}
    for stem in ("s_paired_differences", "s_absolute_intervals", "w_both"):
        foot = ("BE-MSNet minus comparator. Bars: 95% conditional bootstrap intervals."
                if stem == "s_paired_differences" else
                "Dashed line: 0.5 reference. Bars: 95% conditional bootstrap intervals.")
        rendered[stem] = charts.render_forest(
            plt, stem, charts.forest_panels(published, stem, primary),
            ("ds002778", "ds003490 (EO)"), foot)
    rendered["within_recording_vs_subject_independent"] = charts.render_ws(plt, published, primary)
    rendered["probability_bins"] = charts.probability_bins(plt, primary)
    for stem, specification in comparison["figures"].items():
        headers = (("ds002778 → ds003490 (EO)", "ds003490 (EO) → ds002778")
                   if stem == "direct_transfer" else ("ds002778", "ds003490 (EO)"))
        foot = ("Frozen transfer; capped lines show 95% conditional bootstrap intervals"
                if stem == "direct_transfer" else
                "Five-fold mean; capped lines show 95% conditional bootstrap intervals")
        rendered[stem] = charts.render_forest(plt, stem, deepcopy(specification["panels"]), headers, foot)
    rendered["representation_control"] = representation.draw()
    design.draw(output, formats, dpi)
    architecture.draw(output, formats, dpi)
    coordinates = []
    for name, figure in rendered.items():
        for panel in figure.get("panels", []):
            for record in panel.get("records", []):
                coordinates.append({
                    "figure": name, "panel": panel["panel"],
                    "cohort": panel.get("cohort", ""), "metric": panel.get("metric", ""),
                    "model": record["model"], "point": record["point"],
                    "ci_low": record["ci95"][0], "ci_high": record["ci95"][1],
                })
    with (output / "figure_coordinates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coordinates[0]))
        writer.writeheader()
        writer.writerows(coordinates)
    return [name + "." + ext for name in (*rendered, "design_overview", "model_architecture") for ext in formats]
