"""Runtime introspection of glayout pcell generators.

The GUI reads its catalogue from this module so the menu never has to be
edited when cells/primitives are added or renamed. We walk the public
re-exports of ``glayout.primitives`` and ``glayout.cells.{elementary,composite}``
and keep every callable whose first parameter is ``pdk``.
"""

from __future__ import annotations

import importlib
import inspect
import typing
from dataclasses import dataclass, field
from typing import Any, Callable

# (category label, dotted module to import)
_PACKAGES: list[tuple[str, str]] = [
    ("Primitives", "glayout.primitives"),
    ("Elementary", "glayout.cells.elementary"),
    ("Composite", "glayout.cells.composite"),
]


@dataclass
class ParamInfo:
    name: str
    annotation: Any
    default: Any
    has_default: bool
    kind: inspect._ParameterKind
    raw: inspect.Parameter = field(repr=False)


@dataclass
class GeneratorInfo:
    name: str
    category: str
    module: str
    func: Callable[..., Any]
    params: list[ParamInfo]
    doc: str

    @property
    def qualified(self) -> str:
        return f"{self.category}: {self.name}"


def _is_pdk_first(sig: inspect.Signature) -> bool:
    if not sig.parameters:
        return False
    first = next(iter(sig.parameters.values()))
    if first.name == "pdk":
        return True
    ann = first.annotation
    if ann is inspect.Parameter.empty:
        return False
    name = getattr(ann, "__name__", "") or str(ann)
    return "MappedPDK" in name


def _looks_like_generator(name: str, func: Callable) -> bool:
    if name.startswith("_"):
        return False
    if name.endswith("_netlist"):
        return False
    # helpers that mutate / decorate an existing component, not pcell generators
    lname = name.lower()
    if lname.startswith("add_") or "_label" in lname or lname.endswith("_labels"):
        return False
    if lname == "get_component_netlist":
        return False
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return _is_pdk_first(sig)


def _collect_from_package(category: str, dotted: str) -> list[GeneratorInfo]:
    mod = importlib.import_module(dotted)
    out: list[GeneratorInfo] = []
    seen: set[int] = set()
    # Prefer __all__ if defined; otherwise fall back to public attributes.
    candidate_names: list[str]
    if hasattr(mod, "__all__"):
        candidate_names = list(mod.__all__)
    else:
        candidate_names = [n for n in dir(mod) if not n.startswith("_")]

    for name in candidate_names:
        obj = getattr(mod, name, None)
        if obj is None or not callable(obj):
            continue
        if isinstance(obj, type):
            continue  # classes
        if id(obj) in seen:
            continue
        if not _looks_like_generator(name, obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        params: list[ParamInfo] = []
        for p in list(sig.parameters.values())[1:]:  # skip pdk
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            params.append(
                ParamInfo(
                    name=p.name,
                    annotation=p.annotation,
                    default=p.default,
                    has_default=p.default is not inspect.Parameter.empty,
                    kind=p.kind,
                    raw=p,
                )
            )
        seen.add(id(obj))
        out.append(
            GeneratorInfo(
                name=name,
                category=category,
                module=getattr(obj, "__module__", dotted),
                func=obj,
                params=params,
                doc=(inspect.getdoc(obj) or "").strip(),
            )
        )
    out.sort(key=lambda g: g.name)
    return out


def discover_generators() -> list[GeneratorInfo]:
    found: list[GeneratorInfo] = []
    for category, dotted in _PACKAGES:
        try:
            found.extend(_collect_from_package(category, dotted))
        except Exception as exc:  # pragma: no cover
            print(f"[discovery] failed to import {dotted}: {exc}")
    return found


def discover_pdks() -> list[tuple[str, Any]]:
    """Return non-None mapped PDK instances exported by ``glayout.pdk``."""
    pdk_mod = importlib.import_module("glayout.pdk")
    pdks: list[tuple[str, Any]] = []
    for attr in ("sky130_mapped_pdk", "gf180_mapped_pdk", "ihp130_mapped_pdk"):
        obj = getattr(pdk_mod, attr, None)
        if obj is not None:
            short = attr.replace("_mapped_pdk", "")
            pdks.append((short, obj))
    return pdks


# ---------------------------------------------------------------------------
# typing helpers used by the GUI form builder

def annotation_summary(ann: Any) -> str:
    if ann is inspect.Parameter.empty:
        return ""
    if isinstance(ann, str):
        return ann
    origin = typing.get_origin(ann)
    if origin is None:
        return getattr(ann, "__name__", str(ann))
    args = typing.get_args(ann)
    arg_strs = [annotation_summary(a) for a in args]
    origin_name = getattr(origin, "__name__", str(origin))
    if origin is typing.Union:
        non_none = [a for a in arg_strs if a != "NoneType"]
        if len(non_none) == 1 and len(args) == 2:
            return f"Optional[{non_none[0]}]"
        return "Union[" + ", ".join(arg_strs) + "]"
    return f"{origin_name}[{', '.join(arg_strs)}]"


def underlying_types(ann: Any) -> tuple[Any, bool]:
    """Strip ``Optional[X]`` to ``(X, optional?)``."""
    if ann is inspect.Parameter.empty:
        return ann, False
    origin = typing.get_origin(ann)
    if origin is typing.Union:
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return ann, False
