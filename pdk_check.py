"""DRC/LVS dispatch that reads rule decks straight out of the PDK install.

The bundled per-PDK adapters in ``glayout/pdk/<name>_mapped/`` ship .lydrc
files that try to ``%include`` rule decks from a directory that doesn't
exist in this repo, so ``MappedPDK.drc()`` aborts before klayout ever
opens the GDS. This module skips those adapters and runs the rule decks
that ship with the actual PDK install under
``$PDK_ROOT/<variant>/libs.tech/...``.

Variant selection is keyed off the PDK name the GUI passes in (one of
"gf180", "sky130", "ihp130"); each maps to a fixed install directory
under ``$PDK_ROOT``. We deliberately do not look at ``$PDKPATH`` -- it
ties the entire process to one variant chosen at shell-init time, which
made selecting a different PDK in the GUI silently keep using the old
one.

Each ``run_drc`` / ``run_lvs`` returns ``(passed: bool, report_path:
Path | None, log: str)`` so the caller can update the UI without
re-implementing violation/verdict parsing.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


# ---------------------------------------------------------------------------
# PDK locator

@dataclass(frozen=True)
class PdkPaths:
    """Filesystem locations for one PDK variant -- everything we need to
    invoke the stock klayout / magic / netgen flows."""
    name: str            # gLayout PDK key, e.g. "gf180"
    variant: str         # PDK install dir name, e.g. "gf180mcuC"
    pdk_root: Path       # $PDK_ROOT (the parent of all variants)
    pdk_path: Path       # pdk_root / variant
    klayout_dir: Path    # libs.tech/klayout
    drc_script: Path     # script to feed klayout via -r
    drc_vars: dict[str, str]  # -rd switches the rule deck reads


# Per-PDK config. The canonical variant directory under $PDK_ROOT, the
# klayout DRC entry point relative to that variant's libs.tech/klayout/,
# and the -rd switches the rule deck expects. To support a new PDK,
# add an entry here -- nothing else in this file needs changing.
@dataclass(frozen=True)
class _PdkConfig:
    variant: str
    drc_script_rel: str           # relative to libs.tech/klayout/
    drc_vars: dict[str, str]


_PDK_CONFIGS: dict[str, _PdkConfig] = {
    "gf180": _PdkConfig(
        variant="gf180mcuC",
        drc_script_rel="drc/rule_decks/main.drc",
        # variant=C defaults; the rule deck reads these via -rd globals.
        drc_vars={
            "metal_top": "9K",
            "metal_level": "5LM",
            "mim_option": "B",
            "verbose": "false",
            "run_mode": "flat",
            "feol": "true",
            "beol": "true",
            "conn_drc": "true",
            "offgrid": "true",
            "ball": "false",
            "gold": "false",
            "wedge": "false",
            "split_deep": "false",
            "slow_via": "false",
            "table_name": "main",
        },
    ),
    "sky130": _PdkConfig(
        variant="sky130A",
        # sky130A ships a single self-contained .lydrc that just needs
        # $input/$report; no extra knobs to wire up.
        drc_script_rel="drc/sky130A.lydrc",
        drc_vars={},
    ),
    "ihp130": _PdkConfig(
        variant="ihp130",
        drc_script_rel="drc/rule_decks/main.drc",
        drc_vars={},
    ),
}


def resolve_pdk_paths(pdk_name: str) -> PdkPaths:
    """Resolve install paths for ``pdk_name`` under ``$PDK_ROOT``.

    The variant is fixed per PDK (see ``_PDK_CONFIGS``) and not pulled
    from any env var -- ``$PDKPATH`` would pin one variant at shell-init
    time and silently shadow the user's GUI selection.
    """
    cfg = _PDK_CONFIGS.get(pdk_name)
    if cfg is None:
        raise ValueError(f"no DRC integration for PDK {pdk_name!r}")

    pdk_root_env = os.environ.get("PDK_ROOT", "").strip()
    if not pdk_root_env:
        raise RuntimeError(
            "PDK_ROOT is not set; configure it under Tools -> Env settings "
            "or export it from the shell."
        )
    pdk_root = Path(pdk_root_env).expanduser().resolve()
    pdk_path = pdk_root / cfg.variant
    if not pdk_path.is_dir():
        raise RuntimeError(
            f"PDK install for {pdk_name!r} not found at {pdk_path}"
        )

    klayout_dir = pdk_path / "libs.tech" / "klayout"
    drc_script = klayout_dir / cfg.drc_script_rel
    if not drc_script.is_file():
        raise RuntimeError(
            f"DRC script for {pdk_name!r} not found at {drc_script}"
        )
    return PdkPaths(
        name=pdk_name,
        variant=cfg.variant,
        pdk_root=pdk_root,
        pdk_path=pdk_path,
        klayout_dir=klayout_dir,
        drc_script=drc_script,
        drc_vars=dict(cfg.drc_vars),
    )


# ---------------------------------------------------------------------------
# DRC

def _count_drc_violations(report_path: Path) -> int:
    """Count <item> entries in a klayout lyrdb XML report."""
    try:
        root = ET.parse(report_path).getroot()
    except ET.ParseError:
        return -1
    items = root.find("items")
    if items is None:
        return 0
    return sum(1 for _ in items.findall("item"))


@contextlib.contextmanager
def _track_subprocesses(callback: Callable[[subprocess.Popen | None], None] | None):
    """Module-level wrapper around ``subprocess.Popen`` for the duration
    of the with-block. Every Popen spawned in this thread (and any other
    thread that happens to call subprocess.Popen during the window) is
    routed through ``callback`` so the GUI's Cancel button can terminate
    them. Restores the original Popen on exit.

    The DRC/LVS dispatchers stay readable -- they call subprocess
    normally; the wrapper hides the registration plumbing. ``pdk.lvs_netgen``
    is bundled and we don't want to fork it just to add a hook, so this
    interception strategy is the cheapest way to track its child procs
    too.
    """
    if callback is None:
        yield
        return
    orig = subprocess.Popen

    class _TrackedPopen(orig):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            try:
                callback(self)
            except Exception:
                pass

    subprocess.Popen = _TrackedPopen  # type: ignore[assignment]
    try:
        yield
    finally:
        subprocess.Popen = orig


def run_drc(
    pdk: Any, gds_path: Path, out_dir: Path,
    *, klayout_bin: str = "klayout",
    on_proc: Callable[[subprocess.Popen | None], None] | None = None,
) -> tuple[bool, Path, str]:
    """Run klayout DRC on ``gds_path`` using the PDK's stock rule deck.

    Returns ``(clean, report_path, log)``. ``clean`` is True when the
    report XML has zero ``<item>`` entries. If ``on_proc`` is given,
    every subprocess spawned for this run is registered through it so
    the caller can terminate them on cancel.
    """
    paths = resolve_pdk_paths(pdk.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{pdk.name}_{Path(gds_path).stem}_drc.lyrdb"

    args = [
        klayout_bin, "-b",
        "-r", str(paths.drc_script),
        "-rd", f"input={Path(gds_path).resolve()}",
        "-rd", f"report={report_path.resolve()}",
    ]
    for k, v in paths.drc_vars.items():
        args.extend(["-rd", f"{k}={v}"])

    with _track_subprocesses(on_proc):
        proc = subprocess.run(args, capture_output=True, text=True)
    log = (
        f"$ {' '.join(args)}\n\n"
        f"-- stdout --\n{proc.stdout}\n"
        f"-- stderr --\n{proc.stderr}\n"
        f"-- exit code --\n{proc.returncode}\n"
    )
    if proc.returncode != 0 or not report_path.is_file():
        return False, report_path, log + "\nklayout exited with an error\n"
    n_violations = _count_drc_violations(report_path)
    log += f"\nDRC verdict: {'CLEAN' if n_violations == 0 else f'{n_violations} violation(s)'}\n"
    log += f"report: {report_path}\n"
    return n_violations == 0, report_path, log


# ---------------------------------------------------------------------------
# LVS
#
# Mirrors tests/lvs/run_cell_lvs.py: call ``pdk.lvs_netgen`` with
# (layout, design_name, netlist, output_file_path), then inspect the
# report it drops at "<output_file_path>/lvs/<design_name>/<design_name>_lvs.rpt"
# using the same parser the CI uses. The only thing we need to add over
# the CI is a bash shim for the netgen launcher: that script uses
# bash-only parameter expansion but ships with #!/bin/sh, so dash-based
# /bin/sh boxes (Ubuntu) abort with "Bad substitution" before reaching
# netgen.

# Re-use the CI parser instead of inventing our own; same source-of-truth
# for what counts as pass/fail.
_REPO_TESTS_LVS = (
    Path(__file__).resolve().parent.parent / "tests" / "lvs"
)
if str(_REPO_TESTS_LVS) not in sys.path:
    sys.path.insert(0, str(_REPO_TESTS_LVS))
try:
    from run_cell_lvs import _parse_lvs_report as _ci_parse_lvs_report  # noqa: E402
except ImportError:
    _ci_parse_lvs_report = None


def _parse_lvs_report_text(text: str) -> dict:
    """Verdict dict in the CI's shape: is_pass, conclusion, unmatched_*."""
    if _ci_parse_lvs_report is not None:
        return _ci_parse_lvs_report(text)
    # Fallback so the GUI still runs if the tests/ tree isn't available.
    out = {
        "is_pass": False,
        "conclusion": "LVS inconclusive",
        "unmatched_nets": 0,
        "unmatched_instances": 0,
        "raw_tail": text[-1200:] if text else "",
    }
    if "Netlists match" in text or "Circuits match uniquely" in text:
        out.update(is_pass=True, conclusion="Netlists match")
    elif "Netlists do not match" in text or "Netlist mismatch" in text:
        out["conclusion"] = "Netlists do not match"
    return out


@contextlib.contextmanager
def _netgen_shim(real_path: Path) -> Iterator[Path]:
    """Yield a temp dir holding a netgen wrapper that fixes two issues:

    1. The upstream netgen launcher uses bash-only parameter expansion
       but ships with #!/bin/sh; on dash-based /bin/sh boxes (Ubuntu) it
       aborts with "Bad substitution" before reaching netgen. We re-exec
       the launcher under bash.
    2. Older netgen builds (<= 1.5.98 here) don't auto-equate pins on
       cells that appear as empty placeholders in both circuits. The
       ``gf180mcuD_setup.tcl`` only equates them when LVS was invoked
       with ``-blackbox`` (its `if {[model blackbox]}` guard). Newer
       containerised netgens default to that behaviour and the CI
       passes; on local boxes we have to inject the flag. We append
       ``-blackbox`` to any ``lvs`` invocation.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="glayout_shim_"))
    try:
        shim = tmpdir / "netgen"
        shim.write_text(
            "#!/usr/bin/env bash\n"
            "# Re-exec netgen under bash and force -blackbox for lvs runs.\n"
            "args=()\n"
            "saw_lvs=0\n"
            'for a in "$@"; do\n'
            '    args+=("$a")\n'
            '    if [[ "$a" == "lvs" && $saw_lvs -eq 0 ]]; then\n'
            '        saw_lvs=1\n'
            "    fi\n"
            "done\n"
            'if [[ $saw_lvs -eq 1 ]]; then\n'
            '    args+=("-blackbox")\n'
            "fi\n"
            f'exec bash "{real_path}" "${{args[@]}}"\n'
        )
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_lvs(
    pdk: Any, gds_path: Path, spice_path: Path, design_name: str, out_dir: Path,
    *, on_proc: Callable[[subprocess.Popen | None], None] | None = None,
) -> tuple[bool, Path | None, str]:
    """Run LVS the same way ``tests/lvs/run_cell_lvs.py`` does.

    Calls ``pdk.lvs_netgen`` with the CI's argument shape, then reads
    ``<out_dir>/lvs/<design_name>/<design_name>_lvs.rpt`` and parses it
    with the CI's verdict logic. Wraps the call so the dash-incompatible
    netgen launcher is invoked through bash.
    """
    if not shutil.which("netgen"):
        return False, None, (
            "netgen is not on PATH. Configure it under Tools -> Env settings.\n"
        )
    if not shutil.which("magic"):
        return False, None, "magic is not on PATH.\n"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = [
        f"LVS for {design_name} on {pdk.name}",
        f"  magic   = {shutil.which('magic')}",
        f"  netgen  = {shutil.which('netgen')}",
        f"  PDK_ROOT= {os.environ.get('PDK_ROOT')}",
        f"  rpt_dir = {out_dir}",
        "",
    ]

    saved_path = os.environ.get("PATH", "")
    try:
        with _netgen_shim(Path(shutil.which("netgen"))) as shim_dir, \
             _track_subprocesses(on_proc):
            os.environ["PATH"] = f"{shim_dir}{os.pathsep}{saved_path}"
            try:
                ret = pdk.lvs_netgen(
                    layout=str(gds_path),
                    design_name=design_name,
                    netlist=str(spice_path),
                    output_file_path=str(out_dir),
                )
            except Exception as exc:  # noqa: BLE001
                return False, None, "\n".join(log_lines) + f"\n!! lvs_netgen raised: {exc}\n"
    finally:
        os.environ["PATH"] = saved_path

    rpt_file = out_dir / "lvs" / design_name / f"{design_name}_lvs.rpt"
    if not rpt_file.is_file():
        log_lines.append(f"!! report not produced at {rpt_file}")
        log_lines.append(f"   lvs_netgen returned: {ret!r}")
        return False, None, "\n".join(log_lines) + "\n"

    parsed = _parse_lvs_report_text(rpt_file.read_text())
    log_lines.append(f"verdict: {parsed['conclusion']}")
    log_lines.append(f"  unmatched nets:      {parsed['unmatched_nets']}")
    log_lines.append(f"  unmatched instances: {parsed['unmatched_instances']}")
    log_lines.append(f"  report: {rpt_file}")
    if parsed.get("raw_tail"):
        log_lines.append("")
        log_lines.append("-- report tail --")
        log_lines.append(parsed["raw_tail"])
    return bool(parsed["is_pass"]), rpt_file, "\n".join(log_lines) + "\n"


