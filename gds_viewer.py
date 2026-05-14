"""Lightweight GDS viewer: parse with gdstk, render with matplotlib.

Color palettes are loaded from JSON files (one per PDK) in the palettes/
directory for visual consistency between the interactive preview and exported
SVGs/PNGs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import gdstk
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Layer palettes — loaded from palettes/*.json at import time
# ---------------------------------------------------------------------------

_PALETTES_DIR = Path(__file__).resolve().parent / "palettes"


def _load_palettes() -> tuple[dict[tuple[int, int], dict], dict]:
    layers: dict[tuple[int, int], dict] = {}
    default: dict = {"fill": "#dddddd", "stroke": "#666666", "stroke-width": "1", "fill-opacity": "0.40"}

    default_path = _PALETTES_DIR / "default.json"
    if default_path.exists():
        with open(default_path) as f:
            data = json.load(f)
            default = data.get("default", default)

    for p in sorted(_PALETTES_DIR.glob("*.json")):
        if p.name == "default.json":
            continue
        with open(p) as f:
            data = json.load(f)
            for key_str, style in data.get("layers", {}).items():
                l, d = key_str.split("/")
                layers[(int(l), int(d))] = style

    return layers, default


LAYER_STYLES, DEFAULT_STYLE = _load_palettes()


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
