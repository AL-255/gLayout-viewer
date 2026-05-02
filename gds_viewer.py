"""Lightweight GDS viewer: parse with gdstk, render with matplotlib.

Used by the Tk menu so we can preview generated layouts without depending on
gdsfactory's quickplot (which needs a layer_views.yaml). The colour for each
(layer, datatype) pair is derived deterministically from the layer/datatype
integers, so re-renders look stable across runs.
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Iterable

import gdstk
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch


def _color_for(layer: int, datatype: int) -> tuple[float, float, float]:
    # Hash layer/datatype to a deterministic hue.
    h = ((layer * 0x9E3779B1) ^ (datatype * 0x85EBCA77)) & 0xFFFFFFFF
    hue = (h % 360) / 360.0
    sat = 0.55 + ((h >> 8) & 0xFF) / 1024.0  # 0.55..0.80
    val = 0.75 + ((h >> 16) & 0xFF) / 2048.0  # 0.75..0.87
    return colorsys.hsv_to_rgb(hue, sat, val)


def render_gds(gds_path: Path, fig: Figure, *, max_polys_per_layer: int = 4000) -> str:
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
        rgb = _color_for(layer, datatype)
        coll = PolyCollection(
            polylist,
            facecolors=[(*rgb, 0.45)],
            edgecolors=[(*rgb, 0.95)],
            linewidths=0.4,
        )
        ax.add_collection(coll)
        legend_handles.append(Patch(facecolor=rgb, edgecolor="black", label=f"{layer}/{datatype}"))

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
            title="layer/dt",
            frameon=False,
        )
    fig.tight_layout()
    return f"{cell.name}: {len(polys)} polygons across {len(by_key)} (layer,dt) pairs"
