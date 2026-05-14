"""Lightweight GDS viewer: parse with gdstk, render with matplotlib.

Color palette is shared with docs/render_figures.py for visual consistency
between the interactive preview and exported SVGs/PNGs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import gdstk
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Curated soft palette per (layer, datatype) — synced with docs/render_figures.py
# ---------------------------------------------------------------------------

LAYER_STYLES: dict[tuple[int, int], dict] = {
    # ---- gf180mcuC ----------------------------------------------------------
    (12, 0):  {"fill": "#e6f7ff", "stroke": "#7fb8d4", "stroke-width": "1", "fill-opacity": "0.30"},
    (21, 0):  {"fill": "#fff5cc", "stroke": "#c9a227", "stroke-width": "1", "fill-opacity": "0.30"},
    (204, 0): {"fill": "#f0f0f0", "stroke": "#888888", "stroke-width": "1", "fill-opacity": "0.30"},
    (22, 0):  {"fill": "#e2d5f0", "stroke": "#6b4c93", "stroke-width": "1", "fill-opacity": "0.55"},
    (31, 0):  {"fill": "#cfe8e0", "stroke": "#2f7a6a", "stroke-width": "1", "fill-opacity": "0.45"},
    (32, 0):  {"fill": "#e8e0c8", "stroke": "#8a7a2e", "stroke-width": "1", "fill-opacity": "0.45"},
    (30, 0):  {"fill": "#ffd1d1", "stroke": "#c0392b", "stroke-width": "1", "fill-opacity": "0.65"},
    (33, 0):  {"fill": "#888888", "stroke": "#555555", "stroke-width": "0.5", "fill-opacity": "0.75"},
    (34, 0):  {"fill": "#cfe2ff", "stroke": "#3f7fbf", "stroke-width": "1", "fill-opacity": "0.55"},
    (36, 0):  {"fill": "#d4f0d0", "stroke": "#3f9f4f", "stroke-width": "1", "fill-opacity": "0.55"},
    (42, 0):  {"fill": "#fff2b3", "stroke": "#bfa022", "stroke-width": "1", "fill-opacity": "0.55"},
    (46, 0):  {"fill": "#ffd9b3", "stroke": "#cf7f2a", "stroke-width": "1", "fill-opacity": "0.55"},
    (81, 0):  {"fill": "#ffd1e6", "stroke": "#c0398a", "stroke-width": "1", "fill-opacity": "0.55"},
    (35, 0):  {"fill": "#3f7fbf", "stroke": "#1f3f7f", "stroke-width": "1", "fill-opacity": "0.85"},
    (38, 0):  {"fill": "#3f9f4f", "stroke": "#1f5f2f", "stroke-width": "1", "fill-opacity": "0.85"},
    (40, 0):  {"fill": "#bfa022", "stroke": "#7f6f10", "stroke-width": "1", "fill-opacity": "0.85"},
    (41, 0):  {"fill": "#cf7f2a", "stroke": "#7f4f10", "stroke-width": "1", "fill-opacity": "0.85"},
    (117, 5): {"fill": "#f0c8f0", "stroke": "#9f3f9f", "stroke-width": "1", "fill-opacity": "0.40"},
    (127, 5): {"fill": "#ffe0c8", "stroke": "#bf6f1f", "stroke-width": "1", "fill-opacity": "0.30"},
    (118, 5): {"fill": "#c8e0ff", "stroke": "#1f6fbf", "stroke-width": "1", "fill-opacity": "0.30"},
    # ---- sky130 -------------------------------------------------------------
    (64, 18): {"fill": "#e6f7ff", "stroke": "#7fb8d4", "stroke-width": "1", "fill-opacity": "0.30"},
    (64, 20): {"fill": "#fff5cc", "stroke": "#c9a227", "stroke-width": "1", "fill-opacity": "0.30"},
    (64, 44): {"fill": "#f0f0f0", "stroke": "#888888", "stroke-width": "1", "fill-opacity": "0.30"},
    (65, 20): {"fill": "#e2d5f0", "stroke": "#6b4c93", "stroke-width": "1", "fill-opacity": "0.55"},
    (65, 44): {"fill": "#d8c8ec", "stroke": "#5b3c83", "stroke-width": "1", "fill-opacity": "0.55"},
    (93, 44): {"fill": "#e8e0c8", "stroke": "#8a7a2e", "stroke-width": "1", "fill-opacity": "0.45"},
    (94, 20): {"fill": "#cfe8e0", "stroke": "#2f7a6a", "stroke-width": "1", "fill-opacity": "0.45"},
    (66, 20): {"fill": "#ffd1d1", "stroke": "#c0392b", "stroke-width": "1", "fill-opacity": "0.65"},
    (66, 44): {"fill": "#888888", "stroke": "#555555", "stroke-width": "0.5", "fill-opacity": "0.75"},
    (67, 44): {"fill": "#888888", "stroke": "#555555", "stroke-width": "0.5", "fill-opacity": "0.75"},
    (67, 20): {"fill": "#ffe9b8", "stroke": "#bf8b22", "stroke-width": "1", "fill-opacity": "0.55"},
    (68, 20): {"fill": "#cfe2ff", "stroke": "#3f7fbf", "stroke-width": "1", "fill-opacity": "0.55"},
    (68, 44): {"fill": "#3f7fbf", "stroke": "#1f3f7f", "stroke-width": "1", "fill-opacity": "0.85"},
    (69, 20): {"fill": "#d4f0d0", "stroke": "#3f9f4f", "stroke-width": "1", "fill-opacity": "0.55"},
    (69, 44): {"fill": "#3f9f4f", "stroke": "#1f5f2f", "stroke-width": "1", "fill-opacity": "0.85"},
    (70, 20): {"fill": "#fff2b3", "stroke": "#bfa022", "stroke-width": "1", "fill-opacity": "0.55"},
    (70, 44): {"fill": "#bfa022", "stroke": "#7f6f10", "stroke-width": "1", "fill-opacity": "0.85"},
    (71, 20): {"fill": "#ffd9b3", "stroke": "#cf7f2a", "stroke-width": "1", "fill-opacity": "0.55"},
    (89, 44): {"fill": "#f0c8f0", "stroke": "#9f3f9f", "stroke-width": "1", "fill-opacity": "0.40"},
    (95, 20): {"fill": "#e8d8c0", "stroke": "#8a6a3e", "stroke-width": "1", "fill-opacity": "0.30"},
    # ---- ihp130 --------------------------------------------------------------
    # wells
    (31, 0):  {"fill": "#fff5cc", "stroke": "#c9a227", "stroke-width": "1", "fill-opacity": "0.30"},   # ihp130 nwell
    (46, 0):  {"fill": "#f0f0f0", "stroke": "#888888", "stroke-width": "1", "fill-opacity": "0.30"},   # ihp130 pwell
    # active / implants
    (1, 0):   {"fill": "#e2d5f0", "stroke": "#6b4c93", "stroke-width": "1", "fill-opacity": "0.55"},   # ihp130 activ
    (7, 0):   {"fill": "#e8e0c8", "stroke": "#8a7a2e", "stroke-width": "1", "fill-opacity": "0.45"},   # ihp130 nsd (n+)
    (14, 0):  {"fill": "#cfe8e0", "stroke": "#2f7a6a", "stroke-width": "1", "fill-opacity": "0.45"},   # ihp130 psd (p+)
    # poly + contact
    (5, 0):   {"fill": "#ffd1d1", "stroke": "#c0392b", "stroke-width": "1", "fill-opacity": "0.65"},   # ihp130 gatpoly
    (6, 0):   {"fill": "#888888", "stroke": "#555555", "stroke-width": "0.5", "fill-opacity": "0.75"}, # ihp130 cont
    # metal stack — cool->warm by level
    (8, 0):   {"fill": "#cfe2ff", "stroke": "#3f7fbf", "stroke-width": "1", "fill-opacity": "0.55"},   # ihp130 metal1
    (10, 0):  {"fill": "#d4f0d0", "stroke": "#3f9f4f", "stroke-width": "1", "fill-opacity": "0.55"},   # ihp130 metal2
    (30, 0):  {"fill": "#fff2b3", "stroke": "#bfa022", "stroke-width": "1", "fill-opacity": "0.55"},   # ihp130 metal3
    (50, 0):  {"fill": "#ffd9b3", "stroke": "#cf7f2a", "stroke-width": "1", "fill-opacity": "0.55"},   # ihp130 metal4
    (67, 0):  {"fill": "#ffd1e6", "stroke": "#c0398a", "stroke-width": "1", "fill-opacity": "0.55"},   # ihp130 metal5
    # vias
    (19, 0):  {"fill": "#3f7fbf", "stroke": "#1f3f7f", "stroke-width": "1", "fill-opacity": "0.85"},   # ihp130 via1
    (29, 0):  {"fill": "#3f9f4f", "stroke": "#1f5f2f", "stroke-width": "1", "fill-opacity": "0.85"},   # ihp130 via2
    (49, 0):  {"fill": "#bfa022", "stroke": "#7f6f10", "stroke-width": "1", "fill-opacity": "0.85"},   # ihp130 via3
    (66, 0):  {"fill": "#cf7f2a", "stroke": "#7f4f10", "stroke-width": "1", "fill-opacity": "0.85"},   # ihp130 via4
    # mim cap
    (36, 0):  {"fill": "#f0c8f0", "stroke": "#9f3f9f", "stroke-width": "1", "fill-opacity": "0.40"},   # ihp130 mim
    (70, 20): {"fill": "#e8d8f4", "stroke": "#7f5fa7", "stroke-width": "1", "fill-opacity": "0.30"},   # ihp130 capmetbottom
    (71, 20): {"fill": "#f0d8e8", "stroke": "#a75f7f", "stroke-width": "1", "fill-opacity": "0.30"},   # ihp130 capmettop
}

DEFAULT_STYLE = {
    "fill": "#dddddd",
    "stroke": "#666666",
    "stroke-width": "1",
    "fill-opacity": "0.40",
}


def parse_klayout_map(map_path: Path) -> dict[tuple[int, int], str]:
    """Parse a klayout layer-map file (LEF/DEF style: 4 whitespace-separated
    columns ``name purpose layer_num datatype``).

    Returns a ``{(layer, datatype): "name"}`` lookup. Lines whose name is
    ``NAME`` are skipped (those rows describe text-label streams, not the
    drawing geometry the preview renders), as are blank/comment lines and
    rows with a non-integer layer/datatype. The first non-NAME entry for a
    given ``(layer, datatype)`` wins -- subsequent rows usually only
    re-purpose the same stream for PIN/LEFOBS variants.
    """
    out: dict[tuple[int, int], str] = {}
    try:
        text = Path(map_path).read_text(errors="replace")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//")):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        name, _purpose, layer_s, dtype_s = parts[0], parts[1], parts[-2], parts[-1]
        if name == "NAME":
            continue
        try:
            key = (int(layer_s), int(dtype_s))
        except ValueError:
            continue
        out.setdefault(key, name)
    return out


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert '#rrggbb' hex string to (r, g, b) tuple in 0..1 range."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _style_for(layer: int, datatype: int) -> dict:
    """Return fill/stroke colours and opacity for a (layer, datatype) pair."""
    s = LAYER_STYLES.get((layer, datatype), DEFAULT_STYLE)
    return {
        "fill": _hex_to_rgb(s["fill"]),
        "fill_alpha": float(s["fill-opacity"]),
        "stroke": _hex_to_rgb(s["stroke"]),
        "stroke_alpha": 0.95,
        "stroke_width": float(s["stroke-width"]),
    }


def render_gds(
    gds_path: Path,
    fig: Figure,
    *,
    max_polys_per_layer: int = 4000,
    layer_names: Mapping[tuple[int, int], str] | None = None,
) -> str:
    """Render ``gds_path`` into ``fig``. Returns a one-line summary.

    The figure is cleared and a single Axes is configured with equal aspect.
    If the GDS contains many polygons we cap the count per layer to keep the
    UI responsive (the cap is per-layer; the visual coverage is unchanged for
    typical analog cells).
    """
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")

    lib = gdstk.read_gds(str(gds_path))
    tops = lib.top_level()
    if not tops:
        ax.text(0.5, 0.5, "(empty GDS)", transform=ax.transAxes, ha="center")
        return "empty GDS"

    cell = tops[0]
    polys: list[gdstk.Polygon] = list(cell.get_polygons(depth=None))

    by_key: dict[tuple[int, int], list[Iterable]] = {}
    for p in polys:
        by_key.setdefault((p.layer, p.datatype), []).append(p.points)

    # cap per-layer count for responsiveness
    capped_total = 0
    for k in list(by_key):
        if len(by_key[k]) > max_polys_per_layer:
            by_key[k] = by_key[k][:max_polys_per_layer]
        capped_total += len(by_key[k])

    legend_handles: list[Patch] = []
    for (layer, datatype), polylist in sorted(by_key.items()):
        style = _style_for(layer, datatype)
        coll = PolyCollection(
            polylist,
            facecolors=[(*style["fill"], style["fill_alpha"])],
            edgecolors=[(*style["stroke"], style["stroke_alpha"])],
            linewidths=style["stroke_width"],
        )
        ax.add_collection(coll)
        gds_label = f"{layer}/{datatype}"
        if layer_names is not None:
            name = layer_names.get((layer, datatype))
            label = f"{name} ({gds_label})" if name else gds_label
        else:
            label = gds_label
        legend_handles.append(Patch(facecolor=style["fill"], edgecolor=style["stroke"], label=label))

    bb = cell.bounding_box()
    if bb is not None:
        (xmin, ymin), (xmax, ymax) = bb
        pad = 0.05 * max(xmax - xmin, ymax - ymin, 1e-6)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title(cell.name)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            title="layer" if layer_names else "layer/dt",
            frameon=False,
        )
    fig.tight_layout()
    return f"{cell.name}: {len(polys)} polygons across {len(by_key)} (layer,dt) pairs"
