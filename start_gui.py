"""PySide6 GUI for browsing and generating glayout pcells.

Run via ``menu/start_gui.sh`` (or directly with the ``gLayout`` conda env's
python). The catalogue and parameter form are derived at runtime from the
glayout source code via ``menu.discovery`` -- no edits to this file are
needed when cells/primitives are added or modified.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import traceback
import typing
from pathlib import Path
from typing import Any

# Make sibling modules importable when run from anywhere.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_REPO_SRC = _HERE.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

# Force matplotlib onto the Qt backend before any pyplot use.
import matplotlib  # noqa: E402

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure  # noqa: E402

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from discovery import (  # noqa: E402
    GeneratorInfo,
    ParamInfo,
    annotation_summary,
    discover_generators,
    discover_pdks,
    underlying_types,
)
from gds_viewer import render_gds, parse_klayout_map, LAYER_STYLES, DEFAULT_STYLE  # noqa: E402
import pdk_check  # noqa: E402
from themes import THEMES, Theme, get_theme  # noqa: E402

OUT_DIR = _HERE.parent / "out"
DEFAULT_LAYOUT_FILE = _HERE / "default_layout.json"
_LEGACY_LAYOUT_FILE = _HERE / "default_layout.bin"
# Version 2: log moved out of BottomDockWidgetArea into the LEFT area as a
# vertical sibling of cells/params/preview. v1 blobs would re-pin the log to
# the bottom area on restore, which silently brings back the one-way-resize
# bug -- so they are dropped on load and the v2 default is rewritten.
_LAYOUT_FORMAT_VERSION = 2
_SENTINEL = object()
_DEFAULT_THEME = "Fusion light"

# UI scale settings. QT_SCALE_FACTOR is read by QGuiApplication during
# construction, so the value has to be set in the environment *before* the
# QApplication exists. The menu therefore writes to QSettings and asks for a
# restart; main() reads QSettings and applies the env var before constructing
# the app.
_UI_SCALE_KEY = "ui_scale"
_UI_SCALE_SYSTEM = "system"
_UI_SCALE_PERCENTS: tuple[int, ...] = tuple(range(50, 401, 50))


# ---------------------------------------------------------------------------
# parameter parsing

def _parse_value(raw: str, param: ParamInfo) -> Any:
    """Convert text into the python value expected by the generator.

    Empty input means "use the default" (returns a sentinel so the caller
    can omit the kwarg). The literal string ``None`` forces a None argument.
    """
    txt = raw.strip()
    if txt == "" and param.has_default:
        return _SENTINEL
    if txt.lower() == "none":
        return None

    base, _ = underlying_types(param.annotation)
    origin = typing.get_origin(base)

    if origin in (tuple, list, dict, set) or base in (tuple, list, dict, set):
        try:
            return ast.literal_eval(txt)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"{param.name}: cannot parse {txt!r} as {annotation_summary(base)}: {exc}"
            )

    if base is bool:
        low = txt.lower()
        if low in ("true", "1", "yes", "y"):
            return True
        if low in ("false", "0", "no", "n"):
            return False
        raise ValueError(f"{param.name}: expected bool, got {txt!r}")

    if base is int:
        return int(txt)
    if base is float:
        return float(txt)
    if base is str:
        return txt

    try:
        return ast.literal_eval(txt)
    except (ValueError, SyntaxError):
        return txt


def _format_default(val: Any) -> str:
    if val is inspect.Parameter.empty:
        return ""
    if val is None:
        return ""
    return repr(val) if not isinstance(val, str) else val


def _coerce_to_component(obj: Any) -> Any:
    """Wrap a ComponentReference (or anything else without ``write_gds``) into
    a fresh Component so callers can ``write_gds`` it.

    Several generators in the repo annotate ``-> Component`` but actually
    return a ``ComponentReference`` (e.g. ``diff_pair_ibias``) or a tuple
    whose first element is the top-level Component plus a few helper refs
    (e.g. ``diff_pair_stackedcmirror`` returns
    ``(toplevel, drain_routeref, gate_routeref, c_ref)``). Pull the first
    element out and recurse so the rest of the GUI sees a single object
    with ``write_gds``/``info["netlist"]``.
    """
    if isinstance(obj, tuple) and obj:
        return _coerce_to_component(obj[0])
    if hasattr(obj, "write_gds"):
        return obj
    try:
        from gdsfactory.component import Component  # local import — gdsfactory is heavy
    except Exception:
        return obj
    wrapped = Component()
    try:
        wrapped.add(obj)
    except Exception:
        return obj
    try:
        wrapped.add_ports(obj.get_ports_list())
    except Exception:
        pass
    try:
        info = getattr(obj, "info", None)
        if info:
            wrapped.info.update(info)
    except Exception:
        pass
    return wrapped


def _initial_text(p: ParamInfo) -> str:
    if not p.has_default or p.default is None:
        return ""
    if isinstance(p.default, (int, float, str)):
        return str(p.default)
    return repr(p.default)


def _placeholder_for(p: ParamInfo) -> str:
    """Sample value to show as greyed text in an empty required field.

    The bare ``cannot parse '' as tuple[float, float, int]`` error a user
    gets after clicking Generate is uninformative; surfacing an example
    of the right shape inline makes it obvious what to type.
    """
    if p.has_default:
        return ""
    base, _ = underlying_types(p.annotation)
    origin = typing.get_origin(base)
    args = typing.get_args(base) if origin is not None else ()
    if origin is tuple and args:
        sample = []
        for a in args:
            if a is float:
                sample.append("1.0")
            elif a is int:
                sample.append("1")
            elif a is str:
                sample.append('"x"')
            elif a is bool:
                sample.append("False")
            else:
                sample.append("0")
        return "e.g. (" + ", ".join(sample) + ")"
    if base is int:
        return "e.g. 1"
    if base is float:
        return "e.g. 1.0"
    return ""


# ---------------------------------------------------------------------------
# background workers — Qt's signal/slot is the right place to marshal results
# back to the GUI thread.

class _Worker(QtCore.QObject):
    finished = QtCore.Signal(bool, str)         # success, message
    log = QtCore.Signal(str)
    text_out = QtCore.Signal(str, str)          # tab key, text payload
    artifacts = QtCore.Signal(object)           # dict with last_* values

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        # Subprocesses spawned by the worker, tracked so the GUI's Cancel
        # button can terminate them. The list is mutated from the worker
        # thread (register_proc) and the main thread (cancel), so it
        # needs its own lock; QtCore.QMutex would also work but plain
        # threading.Lock keeps the dependency footprint small.
        import threading
        self._cancel_lock = threading.Lock()
        self._cancelled: bool = False
        self._active_procs: list[subprocess.Popen] = []

    def register_proc(self, proc: subprocess.Popen | None) -> None:
        """Hand a freshly-spawned Popen to the worker so a subsequent
        ``cancel()`` can terminate it. Pass ``None`` to garbage-collect
        any procs that have since exited."""
        with self._cancel_lock:
            if proc is None:
                self._active_procs = [p for p in self._active_procs if p.poll() is None]
                return
            self._active_procs.append(proc)
            if self._cancelled:
                # User cancelled before this proc was registered -- terminate
                # it immediately so it doesn't run to completion in the gap.
                self._terminate_locked(proc)

    def cancel(self) -> None:
        """Best-effort cancel: mark the worker cancelled and terminate any
        subprocess it has registered. Pure-Python work (e.g. the
        gdsfactory-heavy GeneratorWorker.run body) cannot be safely
        interrupted from outside, so for those workers cancel only
        prevents *future* subprocesses from running and lets the dialog
        close; the worker thread itself keeps going until it returns."""
        with self._cancel_lock:
            self._cancelled = True
            for p in list(self._active_procs):
                self._terminate_locked(p)

    @staticmethod
    def _terminate_locked(proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class GeneratorWorker(_Worker):
    def __init__(self, gen: GeneratorInfo, pdk: Any, pdk_name: str,
                 kwargs: dict[str, Any], out_dir: Path) -> None:
        super().__init__()
        self.gen = gen
        self.pdk = pdk
        self.pdk_name = pdk_name
        self.kwargs = kwargs
        self.out_dir = out_dir

    @QtCore.Slot()
    def run(self) -> None:
        gen = self.gen
        try:
            comp = gen.func(self.pdk, **self.kwargs)
        except Exception as exc:
            self.log.emit(f"!! generator raised: {exc}")
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, str(exc))
            return

        # Some generators (e.g. diff_pair_ibias) return a ComponentReference
        # despite annotating Component. Coerce so write_gds works.
        comp = _coerce_to_component(comp)

        design_name = gen.name.upper()
        try:
            comp.name = design_name
        except Exception:
            pass

        gds_path = self.out_dir / f"{gen.name}__{self.pdk_name}.gds"
        spice_path = self.out_dir / f"{gen.name}__{self.pdk_name}.spice"
        try:
            comp.write_gds(str(gds_path))
        except Exception as exc:
            self.log.emit(f"!! write_gds failed: {exc}")
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, f"write_gds failed: {exc}")
            return

        netlist = None
        try:
            netlist = comp.info.get("netlist") if hasattr(comp, "info") else None
        except Exception:
            netlist = None

        wrote_spice: Path | None = None
        netlist_status = "(no netlist attached)"
        if netlist is not None and hasattr(netlist, "generate_netlist"):
            try:
                spice_path.write_text(netlist.generate_netlist())
                wrote_spice = spice_path
                netlist_status = f"netlist -> {spice_path}"
            except Exception as exc:
                netlist_status = f"netlist generation failed: {exc}"
                self.log.emit(traceback.format_exc())

        self.log.emit(f"   gds -> {gds_path}")
        self.log.emit(f"   {netlist_status}")
        self.artifacts.emit({
            "comp": comp,
            "gds": gds_path,
            "spice": wrote_spice,
            "pdk": self.pdk,
            "pdk_name": self.pdk_name,
            "design_name": design_name,
        })
        self.finished.emit(True, f"wrote {gds_path.name}")


class DrcWorker(_Worker):
    """Run DRC by reading the rule deck out of $PDKPATH directly.

    The bundled MappedPDK.drc() entry-points point at .lydrc shims that
    %include rule decks from a path that doesn't ship in this repo, so
    they abort before opening the GDS. pdk_check.run_drc bypasses them
    and invokes klayout against the PDK install's main.drc.
    """

    def __init__(self, pdk: Any, gds: Path, out_dir: Path) -> None:
        super().__init__()
        self.pdk = pdk
        self.gds = gds
        self.out_dir = out_dir

    @QtCore.Slot()
    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            clean, report_path, log = pdk_check.run_drc(
                self.pdk, self.gds, self.out_dir,
                on_proc=self.register_proc,
            )
        except Exception as exc:
            self.text_out.emit("drc", f"!! DRC raised: {exc}\n" + traceback.format_exc())
            self.finished.emit(False, f"DRC error: {exc}")
            return
        if self.cancelled:
            self.text_out.emit("drc", log + "\n!! cancelled by user\n")
            self.finished.emit(False, "DRC cancelled")
            return
        verdict = "CLEAN" if clean else "VIOLATIONS"
        head = f"DRC verdict: {verdict}\nreport: {report_path}\n\n"
        self.text_out.emit("drc", head + log)
        self.finished.emit(bool(clean), f"DRC {verdict.lower()}")


class LvsWorker(_Worker):
    """Run LVS through pdk_check, which forces variant-correct setup files
    and shims the netgen launcher around its dash-incompatibility."""

    def __init__(self, pdk: Any, gds: Path, spice: Path,
                 design_name: str, pdk_root: str) -> None:
        super().__init__()
        self.pdk = pdk
        self.gds = gds
        self.spice = spice
        self.design_name = design_name
        self.pdk_root = pdk_root  # accepted for compat; pdk_check resolves it

    @QtCore.Slot()
    def run(self) -> None:
        out_dir = self.gds.parent / "lvs"
        try:
            success, _report, log = pdk_check.run_lvs(
                self.pdk, self.gds, self.spice, self.design_name, out_dir,
                on_proc=self.register_proc,
            )
        except Exception as exc:
            self.text_out.emit("lvs", f"!! LVS raised: {exc}\n" + traceback.format_exc())
            self.finished.emit(False, f"LVS error: {exc}")
            return
        if self.cancelled:
            self.text_out.emit("lvs", log + "\n!! cancelled by user\n")
            self.finished.emit(False, "LVS cancelled")
            return
        self.text_out.emit("lvs", log)
        self.finished.emit(success, "LVS match" if success else "LVS mismatch")


# ---------------------------------------------------------------------------
# env settings dialog

# Tool option lists. Currently single-entry but the combo boxes are wired up
# so dropping additional backends in (e.g. klayout-LVS, magic-DRC) only needs
# new strings here plus matching wiring in the run_drc / run_lvs paths.
_DRC_TOOLS = ("klayout",)
_LVS_TOOLS = ("magic+netgen",)


def _detect_env_defaults() -> dict[str, str]:
    """Resolve env-settings fields from the current shell environment.

    Used as the source of placeholder text in the dialog (so the user can see
    what the app *would* use for empty fields) and as the fallback at apply
    time. Values come straight from os.environ / shutil.which on the as-
    launched PATH; we do NOT bake these into QSettings -- that would pin
    machine-specific paths into the user's persisted config.
    """
    return {
        "pdk_root": os.environ.get("PDK_ROOT", ""),
        "klayout_path": shutil.which("klayout") or "",
        "magic_path": shutil.which("magic") or "",
        "netgen_path": shutil.which("netgen") or "",
    }


class EnvSettingsDialog(QtWidgets.QDialog):
    """Edit DRC/LVS-related environment paths and tool selection."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        current: dict[str, str],
        detected: dict[str, str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Env settings")
        self.setModal(True)
        self.setMinimumWidth(620)
        detected = detected or {}

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.pdk_root_edit = self._add_path_row(
            form, "PDK_ROOT:", current.get("pdk_root", ""),
            placeholder=detected.get("pdk_root", ""), pick_dir=True,
        )

        form.addRow(self._section_label("DRC"))
        self.drc_tool_combo = QtWidgets.QComboBox()
        self.drc_tool_combo.addItems(_DRC_TOOLS)
        i = self.drc_tool_combo.findText(current.get("drc_tool", _DRC_TOOLS[0]))
        self.drc_tool_combo.setCurrentIndex(i if i >= 0 else 0)
        form.addRow("DRC tool:", self.drc_tool_combo)
        self.klayout_edit = self._add_path_row(
            form, "klayout binary:", current.get("klayout_path", ""),
            placeholder=detected.get("klayout_path", ""), pick_dir=False,
        )

        form.addRow(self._section_label("LVS"))
        self.lvs_tool_combo = QtWidgets.QComboBox()
        self.lvs_tool_combo.addItems(_LVS_TOOLS)
        i = self.lvs_tool_combo.findText(current.get("lvs_tool", _LVS_TOOLS[0]))
        self.lvs_tool_combo.setCurrentIndex(i if i >= 0 else 0)
        form.addRow("LVS tool:", self.lvs_tool_combo)
        self.magic_edit = self._add_path_row(
            form, "magic binary:", current.get("magic_path", ""),
            placeholder=detected.get("magic_path", ""), pick_dir=False,
        )
        self.netgen_edit = self._add_path_row(
            form, "netgen binary:", current.get("netgen_path", ""),
            placeholder=detected.get("netgen_path", ""), pick_dir=False,
        )

        hint = QtWidgets.QLabel(
            "Greyed-out values are detected from the shell environment "
            "(PDK_ROOT, PATH lookups). Leaving a field empty keeps that "
            "auto-detected value; type to override."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text); padding-top: 4px;")

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(form)
        v.addWidget(hint)
        v.addWidget(buttons)

    def _add_path_row(
        self,
        form: QtWidgets.QFormLayout,
        label: str,
        value: str,
        *,
        pick_dir: bool,
        placeholder: str = "",
    ) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(value)
        if placeholder:
            edit.setPlaceholderText(placeholder)
        browse = QtWidgets.QPushButton("Browse...")
        browse.setAutoDefault(False)
        browse.clicked.connect(lambda: self._on_browse(edit, pick_dir))
        host = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(host)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        h.addWidget(browse, 0)
        form.addRow(label, host)
        return edit

    def _on_browse(self, edit: QtWidgets.QLineEdit, pick_dir: bool) -> None:
        start = edit.text().strip() or os.path.expanduser("~")
        if pick_dir:
            picked = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select directory", start
            )
        else:
            picked, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select binary", start
            )
        if picked:
            edit.setText(picked)

    @staticmethod
    def _section_label(text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        f = lbl.font()
        f.setBold(True)
        lbl.setFont(f)
        lbl.setStyleSheet("padding-top: 6px;")
        return lbl

    def values(self) -> dict[str, str]:
        return {
            "pdk_root": self.pdk_root_edit.text().strip(),
            "drc_tool": self.drc_tool_combo.currentText(),
            "klayout_path": self.klayout_edit.text().strip(),
            "lvs_tool": self.lvs_tool_combo.currentText(),
            "magic_path": self.magic_edit.text().strip(),
            "netgen_path": self.netgen_edit.text().strip(),
        }


# ---------------------------------------------------------------------------
# about dialog

# Drop the actual logo files into ./assets/ as either .png or .svg under
# these stems and they show up in the About dialog automatically. Leave
# them missing and the dialog falls back to text labels, so the build
# stays self-contained when nobody's checked the binary assets in.
_ASSET_DIR = _HERE / "assets"
_LOGO_STEMS = (("Brown", "brown"), ("U-M", "umich"))
_LOGO_EXTS = (".png", ".svg", ".jpg", ".jpeg")
_LICENSE_FILE = _HERE / "LICENSE"


def _find_logo(stem: str) -> Path | None:
    for ext in _LOGO_EXTS:
        cand = _ASSET_DIR / f"{stem}{ext}"
        if cand.is_file():
            return cand
    return None


class AboutDialog(QtWidgets.QDialog):
    """About dialog with school logos, lab link, and author info."""

    LAB_URL = "https://www.saliganelab.com/"
    AUTHOR_NAME = "Anhang Li"
    AUTHOR_EMAIL = "anhangli@umich.edu"
    LOGO_HEIGHT = 96  # px, for both logos so they line up nicely

    def __init__(self, parent: QtWidgets.QWidget | None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About")
        self.setModal(True)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 16)
        outer.setSpacing(14)

        # Logos row.
        logos = QtWidgets.QHBoxLayout()
        logos.setSpacing(28)
        logos.addStretch(1)
        for fallback_text, stem in _LOGO_STEMS:
            logos.addWidget(self._build_logo_widget(fallback_text, _find_logo(stem)))
        logos.addStretch(1)
        outer.addLayout(logos)

        # App / lab info.
        title = QtWidgets.QLabel("gLayout Viewer")
        f = title.font()
        f.setPointSize(f.pointSize() + 3)
        f.setBold(True)
        title.setFont(f)
        title.setAlignment(QtCore.Qt.AlignCenter)
        outer.addWidget(title)

        lab = QtWidgets.QLabel(
            f'<p align="center">'
            f'This utility is part of the gLayout project at <a href="{self.LAB_URL}">Saligane Lab</a>'
            f"</p>"
        )
        lab.setOpenExternalLinks(True)
        lab.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        outer.addWidget(lab)

        author = QtWidgets.QLabel(
            f'<p align="center">'
            f"Coded by {self.AUTHOR_NAME} "
            f'(<a href="mailto:{self.AUTHOR_EMAIL}">{self.AUTHOR_EMAIL}</a>)'
            f"</p>"
        )
        author.setOpenExternalLinks(True)
        author.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        outer.addWidget(author)

        # License section -- scrollable, read-only, monospaced. Loaded
        # straight from ./LICENSE so the dialog stays in sync if the
        # licence text is ever updated (no re-paste here required).
        license_label = QtWidgets.QLabel("License")
        f = license_label.font()
        f.setBold(True)
        license_label.setFont(f)
        outer.addWidget(license_label)

        license_view = QtWidgets.QPlainTextEdit()
        license_view.setReadOnly(True)
        license_view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        license_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        license_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        mono = QtGui.QFont("Monospace")
        mono.setStyleHint(QtGui.QFont.TypeWriter)
        license_view.setFont(mono)
        license_view.setMinimumHeight(180)
        try:
            license_view.setPlainText(_LICENSE_FILE.read_text())
        except OSError as exc:
            license_view.setPlainText(f"(could not read {_LICENSE_FILE}: {exc})")
        outer.addWidget(license_view, stretch=1)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        outer.addWidget(bb)

        # Sized so the licence text gets a comfortable read; user can resize.
        self.resize(560, 600)

    def _build_logo_widget(self, fallback_text: str, path: Path | None) -> QtWidgets.QWidget:
        """Return a QLabel showing the logo if found, else a styled text
        placeholder so the layout stays coherent until someone drops the
        real image into ./assets/."""
        lbl = QtWidgets.QLabel()
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFixedHeight(self.LOGO_HEIGHT)
        if path is not None:
            pix = QtGui.QPixmap(str(path))
            if not pix.isNull():
                lbl.setPixmap(
                    pix.scaledToHeight(
                        self.LOGO_HEIGHT, QtCore.Qt.SmoothTransformation
                    )
                )
                lbl.setToolTip(str(path))
                return lbl
        # fallback
        lbl.setText(fallback_text)
        f = lbl.font()
        f.setPointSize(f.pointSize() + 6)
        f.setBold(True)
        lbl.setFont(f)
        lbl.setStyleSheet(
            "QLabel {"
            "  border: 1px dashed palette(mid);"
            "  padding: 12px 24px;"
            "  color: palette(placeholder-text);"
            "}"
        )
        lbl.setToolTip(
            f"drop {fallback_text} logo at "
            f"{(_ASSET_DIR / _LOGO_STEMS[0 if fallback_text == _LOGO_STEMS[0][0] else 1][1]).with_suffix('.png')}"
        )
        return lbl


# ---------------------------------------------------------------------------
# main window

class PcellMenu(QtWidgets.QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("glayout pcell generator")
        self.resize(1500, 920)

        self._generators: list[GeneratorInfo] = []
        self._pdks: list[tuple[str, Any]] = []
        self._current: GeneratorInfo | None = None

        # parameter form rows
        self._param_rows: list[tuple[ParamInfo, QtWidgets.QWidget]] = []
        # type-annotation labels in the param form, kept around so the theme
        # switch can re-tint them (bright orange on dark, navy on light).
        self._type_labels: list[QtWidgets.QLabel] = []
        # Secondary/dim labels (docstring, "default value" column). Tracked so
        # _apply_theme can give them theme-aware colours -- a hardcoded
        # ``color:#444`` is unreadable on a Dracula/Nord background.
        self._secondary_labels: list[QtWidgets.QLabel] = []

        # last successful generation
        self._last_comp: Any = None
        self._last_gds: Path | None = None
        self._last_spice: Path | None = None
        self._last_pdk: Any = None
        self._last_pdk_name: str = ""
        self._last_design_name: str = ""

        self._busy = False
        self._workers: list[tuple[QtCore.QThread, _Worker]] = []  # keep refs

        self._settings = QtCore.QSettings("glayout", "pcell-menu")
        self._current_theme: Theme = get_theme(
            self._settings.value("theme", _DEFAULT_THEME, type=str)
        )
        self._theme_actions: dict[str, QtGui.QAction] = {}
        self._theme_default_style: str = ""

        # Env settings (PDK_ROOT, DRC/LVS tool paths) live in QSettings and
        # are pushed into os.environ early so PDK discovery sees the user's
        # configured PDK_ROOT instead of whatever the launch shell exported.
        self._env_settings: dict[str, str] = self._load_env_settings()
        self._apply_env_settings()

        # Dock widgets, in left-to-right / top-to-bottom default order so the
        # "Reset layout" entry can rebuild the initial arrangement.
        self._docks: dict[str, QtWidgets.QDockWidget] = {}

        self._build_ui()
        self._build_menubar()
        self._apply_theme(self._current_theme)
        QtCore.QTimer.singleShot(0, self._restore_dock_state)
        QtCore.QTimer.singleShot(50, self._initial_load)

    # ----- UI scaffolding -------------------------------------------------
    def _build_ui(self) -> None:
        # Dock-only main window: no central widget, every panel is a
        # QDockWidget that the user can drag, float, snap, tabify, or stack.
        # Removing the central widget (PySide6 supports None) is what makes
        # the inter-area separators (e.g. between Left and Bottom dock areas)
        # behave as first-class draggable handles.
        self.setCentralWidget(None)

        self.setDockOptions(
            QtWidgets.QMainWindow.AnimatedDocks
            | QtWidgets.QMainWindow.AllowNestedDocks
            | QtWidgets.QMainWindow.AllowTabbedDocks
            | QtWidgets.QMainWindow.GroupedDragging
        )
        self.setDockNestingEnabled(True)
        self.setTabPosition(QtCore.Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.North)

        # Make the inter-dock separators wide and obvious -- the default 1px
        # line is hard to grab, and users mistake it for non-draggable.
        self.setStyleSheet(
            "QMainWindow::separator { background: palette(mid); }"
            "QMainWindow::separator:horizontal { height: 5px; }"
            "QMainWindow::separator:vertical { width: 5px; }"
            "QMainWindow::separator:hover { background: palette(highlight); }"
        )

        self._build_toolbar()

        # Build dock contents.
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemSelectionChanged.connect(self._on_select)

        self._docks["cells"] = self._make_dock("cells", "Cells & primitives", self.tree)

        self._docks["params"] = self._make_dock(
            "params", "Parameters", self._build_param_body()
        )
        self._docks["preview"] = self._make_dock(
            "preview", "Preview", self._build_preview_body()
        )
        self.drc_text = self._build_text_widget()
        self._docks["drc"] = self._make_dock(
            "drc", "DRC report (klayout)", self.drc_text
        )
        self.lvs_text = self._build_text_widget()
        self._docks["lvs"] = self._make_dock(
            "lvs", "LVS report (magic+netgen)", self.lvs_text
        )
        self.log_widget = QtWidgets.QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumBlockCount(5000)
        self._docks["log"] = self._make_dock("log", "Log", self.log_widget)

        self._apply_default_layout()

    def _build_toolbar(self) -> None:
        tb = QtWidgets.QToolBar("Main controls", self)
        tb.setObjectName("toolbar_main")
        tb.setMovable(True)
        tb.setFloatable(True)
        self.addToolBar(QtCore.Qt.TopToolBarArea, tb)
        self._toolbar = tb

        tb.addWidget(QtWidgets.QLabel(" PDK: "))
        self.pdk_combo = QtWidgets.QComboBox()
        self.pdk_combo.setMinimumWidth(140)
        tb.addWidget(self.pdk_combo)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel(" Output: "))
        self.outdir_edit = QtWidgets.QLineEdit(str(OUT_DIR))
        self.outdir_edit.setMinimumWidth(280)
        tb.addWidget(self.outdir_edit)
        tb.addSeparator()

        self.reload_btn = QtWidgets.QPushButton("Reload")
        self.reload_btn.setToolTip("Re-discover cells & primitives")
        self.reload_btn.clicked.connect(self._initial_load)
        tb.addWidget(self.reload_btn)
        tb.addSeparator()

        self.gen_btn = QtWidgets.QPushButton("Generate GDS + netlist")
        self.gen_btn.setEnabled(False)
        self.gen_btn.clicked.connect(self._on_generate)
        tb.addWidget(self.gen_btn)

        self.drc_btn = QtWidgets.QPushButton("Run DRC")
        self.drc_btn.setToolTip("Run klayout DRC on the last generation")
        self.drc_btn.setEnabled(False)
        self.drc_btn.clicked.connect(self._on_run_drc)
        tb.addWidget(self.drc_btn)

        self.lvs_btn = QtWidgets.QPushButton("Run LVS")
        self.lvs_btn.setToolTip("Run magic+netgen LVS on the last generation")
        self.lvs_btn.setEnabled(False)
        self.lvs_btn.clicked.connect(self._on_run_lvs)
        tb.addWidget(self.lvs_btn)

        self.klayout_btn = QtWidgets.QPushButton("Open in KLayout")
        self.klayout_btn.setToolTip(
            "Launch klayout -e on the last generated GDS, with KLAYOUT_PATH "
            "pointing at the PDK's libs.tech/klayout"
        )
        self.klayout_btn.setEnabled(False)
        self.klayout_btn.clicked.connect(self._on_open_klayout)
        tb.addWidget(self.klayout_btn)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        tb.addWidget(spacer)

        self.status_lbl = QtWidgets.QLabel("")
        tb.addWidget(self.status_lbl)

    def _make_dock(self, key: str, title: str, body: QtWidgets.QWidget) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(f"dock_{key}")
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        dock.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        dock.setWidget(body)
        return dock

    def _build_param_body(self) -> QtWidgets.QWidget:
        host = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(host)
        v.setContentsMargins(6, 6, 6, 6)

        self.title_lbl = QtWidgets.QLabel("(select a cell on the left)")
        f = self.title_lbl.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 1)
        self.title_lbl.setFont(f)
        v.addWidget(self.title_lbl)

        self.module_lbl = QtWidgets.QLabel("")
        self.module_lbl.setStyleSheet("color:#555")
        v.addWidget(self.module_lbl)

        self.param_scroll = QtWidgets.QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_host = QtWidgets.QWidget()
        self.param_layout = QtWidgets.QVBoxLayout(self.param_host)
        self.param_layout.setAlignment(QtCore.Qt.AlignTop)
        self.param_scroll.setWidget(self.param_host)
        v.addWidget(self.param_scroll, stretch=1)
        return host

    def _build_preview_body(self) -> QtWidgets.QWidget:
        body = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)
        self.preview_status = QtWidgets.QLabel(
            "(no GDS rendered yet — click Generate). "
            "Drag to pan, scroll to zoom."
        )
        self.preview_status.setStyleSheet("color:#555")
        v.addWidget(self.preview_status)

        # Toolbar row: native matplotlib nav + a "show layer names" toggle.
        # Names come from the PDK's klayout .map file -- read on demand and
        # cached per PDK in self._layer_name_maps.
        toolrow = QtWidgets.QWidget()
        th = QtWidgets.QHBoxLayout(toolrow)
        th.setContentsMargins(0, 0, 0, 0)
        self.preview_fig = Figure(figsize=(6.0, 4.5), dpi=110)
        self.preview_canvas = FigureCanvasQTAgg(self.preview_fig)
        self.preview_toolbar = NavigationToolbar2QT(self.preview_canvas, body)
        th.addWidget(self.preview_toolbar, stretch=1)
        self.preview_show_layer_names = QtWidgets.QCheckBox("Show layer names")
        self.preview_show_layer_names.setToolTip(
            "Annotate the legend with PDK layer names from "
            "$PDK_ROOT/<variant>/libs.tech/klayout/tech/*.map"
        )
        self.preview_show_layer_names.toggled.connect(self._on_layer_name_toggle)
        th.addWidget(self.preview_show_layer_names, stretch=0)
        self.save_svg_btn = QtWidgets.QPushButton("Save SVG...")
        self.save_svg_btn.setToolTip(
            "Export the current GDS as an SVG via gdstk, plus a companion "
            "PNG of the preview rendering"
        )
        self.save_svg_btn.setEnabled(False)
        self.save_svg_btn.clicked.connect(self._on_save_svg)
        th.addWidget(self.save_svg_btn, stretch=0)
        v.addWidget(toolrow)

        v.addWidget(self.preview_canvas, stretch=1)
        self._install_preview_interactions()

        # Per-PDK lookup, lazily populated. Empty dict means "tried, found
        # nothing"; missing key means "haven't tried yet".
        self._layer_name_maps: dict[str, dict[tuple[int, int], str]] = {}
        return body

    # --- direct pan/zoom interactions on the preview canvas -------------
    def _install_preview_interactions(self) -> None:
        # Left-button drag to pan, scroll wheel to zoom centred on the
        # cursor. Coexists with the matplotlib toolbar: if the user has
        # toggled the toolbar's Pan or Zoom mode on, we step aside so the
        # toolbar's own handlers run instead.
        self._pan_state: dict[str, Any] | None = None
        self.preview_canvas.setCursor(QtCore.Qt.OpenHandCursor)
        # focus is required for some platforms to actually receive scroll
        # events from matplotlib's event loop.
        self.preview_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.preview_canvas.mpl_connect("scroll_event", self._on_preview_scroll)
        self.preview_canvas.mpl_connect("button_press_event", self._on_preview_press)
        self.preview_canvas.mpl_connect("motion_notify_event", self._on_preview_move)
        self.preview_canvas.mpl_connect("button_release_event", self._on_preview_release)

    def _toolbar_active(self) -> bool:
        # matplotlib >= 3.3 exposes Toolbar.mode as a property/string;
        # an empty string means no nav-mode is engaged.
        mode = getattr(self.preview_toolbar, "mode", "")
        return bool(str(mode).strip())

    def _on_preview_scroll(self, event) -> None:
        if self._toolbar_active():
            return
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        # Step factor: scroll up zooms in, scroll down zooms out.
        step = 1.0 / 1.2 if event.button == "up" else 1.2
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        ax.set_xlim(cx - (cx - xmin) * step, cx + (xmax - cx) * step)
        ax.set_ylim(cy - (cy - ymin) * step, cy + (ymax - cy) * step)
        self.preview_canvas.draw_idle()

    def _on_preview_press(self, event) -> None:
        if self._toolbar_active():
            return
        if event.button != 1:  # left-click only
            return
        ax = event.inaxes
        if ax is None or event.x is None or event.y is None:
            return
        self._pan_state = {
            "ax": ax,
            "x_press": event.x,
            "y_press": event.y,
            "xlim": ax.get_xlim(),
            "ylim": ax.get_ylim(),
        }
        self.preview_canvas.setCursor(QtCore.Qt.ClosedHandCursor)

    def _on_preview_move(self, event) -> None:
        if self._pan_state is None:
            return
        if event.x is None or event.y is None:
            return
        ax = self._pan_state["ax"]
        # Convert pixel delta to data-coordinate delta via the inverse
        # display transform of the axes captured at press time. Doing this
        # in pixel space avoids feedback loops (xlim moves while xdata is
        # measured against the moving xlim).
        inv = ax.transData.inverted()
        p_press = inv.transform((self._pan_state["x_press"], self._pan_state["y_press"]))
        p_now = inv.transform((event.x, event.y))
        ddx = p_now[0] - p_press[0]
        ddy = p_now[1] - p_press[1]
        xmin, xmax = self._pan_state["xlim"]
        ymin, ymax = self._pan_state["ylim"]
        ax.set_xlim(xmin - ddx, xmax - ddx)
        ax.set_ylim(ymin - ddy, ymax - ddy)
        self.preview_canvas.draw_idle()

    def _on_preview_release(self, event) -> None:
        if self._pan_state is None:
            return
        self._pan_state = None
        self.preview_canvas.setCursor(QtCore.Qt.OpenHandCursor)

    def _build_text_widget(self) -> QtWidgets.QPlainTextEdit:
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setMaximumBlockCount(20000)
        font = QtGui.QFont("Monospace")
        font.setStyleHint(QtGui.QFont.TypeWriter)
        text.setFont(font)
        text.setPlaceholderText("(no run yet)")
        return text

    def _apply_default_layout(self) -> None:
        # Detach any existing dock placement first.
        for d in self._docks.values():
            self.removeDockWidget(d)

        cells = self._docks["cells"]
        params = self._docks["params"]
        preview = self._docks["preview"]
        drc = self._docks["drc"]
        lvs = self._docks["lvs"]
        log = self._docks["log"]

        # Layout strategy: a single nested-split tree rooted in the LEFT dock
        # area. Top row is cells | params | (preview tabbed with drc/lvs);
        # log sits below as a horizontally-spanning sibling. We deliberately
        # avoid BottomDockWidgetArea here because, with no central widget,
        # Qt's QMainWindow lets you drag the inter-area separator one way
        # only -- the bottom dock can be grown but not shrunk past its
        # initial size. Routing log through splitDockWidget instead puts the
        # divider on a regular QSplitter handle, which resizes both ways.
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, cells)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, log)
        self.splitDockWidget(cells, log, QtCore.Qt.Vertical)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, params)
        self.splitDockWidget(cells, params, QtCore.Qt.Horizontal)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, preview)
        self.splitDockWidget(params, preview, QtCore.Qt.Horizontal)
        # Stack drc and lvs onto preview as tabs in the same slot.
        self.tabifyDockWidget(preview, drc)
        self.tabifyDockWidget(preview, lvs)
        preview.raise_()  # preview is the default visible tab

        # Reasonable initial sizes.
        self.resizeDocks([cells, params, preview], [240, 460, 700], QtCore.Qt.Horizontal)
        self.resizeDocks([preview, log], [520, 160], QtCore.Qt.Vertical)

        for d in self._docks.values():
            d.show()

    # ----- menu bar / themes ---------------------------------------------
    def _build_menubar(self) -> None:
        bar = self.menuBar()
        view_menu = bar.addMenu("&View")

        panels_menu = view_menu.addMenu("Panels")
        for key in ("cells", "params", "preview", "drc", "lvs", "log"):
            dock = self._docks[key]
            action = dock.toggleViewAction()
            action.setText(dock.windowTitle())
            panels_menu.addAction(action)
        panels_menu.addSeparator()
        save_action = QtGui.QAction("Save current as default layout", self)
        save_action.setStatusTip(f"Write current arrangement to {DEFAULT_LAYOUT_FILE.name}")
        save_action.triggered.connect(self._write_default_layout_file)
        panels_menu.addAction(save_action)
        reset_action = QtGui.QAction("Reload default layout", self)
        reset_action.setStatusTip(f"Re-apply layout from {DEFAULT_LAYOUT_FILE.name}")
        reset_action.triggered.connect(self._reset_layout)
        panels_menu.addAction(reset_action)

        view_menu.addSeparator()

        tools_menu = bar.addMenu("&Tools")
        env_action = QtGui.QAction("Env settings...", self)
        env_action.setStatusTip(
            "Configure PDK_ROOT and DRC/LVS tool paths"
        )
        env_action.triggered.connect(self._open_env_settings)
        tools_menu.addAction(env_action)

        help_menu = bar.addMenu("&Help")
        about_action = QtGui.QAction("About", self)
        about_action.setStatusTip("About this application")
        about_action.triggered.connect(self._open_about)
        help_menu.addAction(about_action)

        theme_menu = view_menu.addMenu("Theme")
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        for theme in THEMES:
            action = QtGui.QAction(theme.name, self, checkable=True)
            action.setChecked(theme.name == self._current_theme.name)
            action.triggered.connect(lambda _checked=False, t=theme: self._on_theme_chosen(t))
            group.addAction(action)
            theme_menu.addAction(action)
            self._theme_actions[theme.name] = action

        scale_menu = view_menu.addMenu("UI Scale")
        scale_group = QtGui.QActionGroup(self)
        scale_group.setExclusive(True)
        current_scale = (
            self._settings.value(_UI_SCALE_KEY, _UI_SCALE_SYSTEM, type=str)
            or _UI_SCALE_SYSTEM
        )
        sys_action = QtGui.QAction("System default", self, checkable=True)
        sys_action.setChecked(current_scale == _UI_SCALE_SYSTEM)
        sys_action.triggered.connect(
            lambda _checked=False: self._on_ui_scale_chosen(_UI_SCALE_SYSTEM)
        )
        scale_group.addAction(sys_action)
        scale_menu.addAction(sys_action)
        scale_menu.addSeparator()
        for pct in _UI_SCALE_PERCENTS:
            action = QtGui.QAction(f"{pct}%", self, checkable=True)
            action.setChecked(current_scale == str(pct))
            action.triggered.connect(
                lambda _checked=False, p=pct: self._on_ui_scale_chosen(str(p))
            )
            scale_group.addAction(action)
            scale_menu.addAction(action)

    def _open_about(self) -> None:
        AboutDialog(self).exec()

    def _on_theme_chosen(self, theme: Theme) -> None:
        self._current_theme = theme
        self._settings.setValue("theme", theme.name)
        self._apply_theme(theme)
        action = self._theme_actions.get(theme.name)
        if action is not None:
            action.setChecked(True)

    def _on_ui_scale_chosen(self, scale: str) -> None:
        current = (
            self._settings.value(_UI_SCALE_KEY, _UI_SCALE_SYSTEM, type=str)
            or _UI_SCALE_SYSTEM
        )
        if scale == current:
            return
        self._settings.setValue(_UI_SCALE_KEY, scale)
        self._settings.sync()
        ret = QtWidgets.QMessageBox.question(
            self,
            "Restart required",
            "UI scale changes take effect after restarting the application.\n\n"
            "Restart now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if ret == QtWidgets.QMessageBox.Yes:
            self._restart_application()

    def _restart_application(self) -> None:
        # Re-exec the current interpreter with the same argv so the new
        # QT_SCALE_FACTOR is picked up during QGuiApplication construction.
        # Defer until after the menu/event returns so Qt can finish dispatching
        # the triggering action cleanly.
        def _do_exec() -> None:
            os.execv(sys.executable, [sys.executable, *sys.argv])

        QtCore.QTimer.singleShot(0, _do_exec)

    def _apply_theme(self, theme: Theme) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        if not self._theme_default_style:
            cur = app.style()
            self._theme_default_style = cur.objectName() if cur is not None else ""
        # Always install a *fresh* QStyle instance. ``setStyle("Fusion")`` is
        # a no-op when Fusion is already active and would otherwise skip the
        # polish path, leaving widgets styled against the old palette until a
        # second click. ``QStyleFactory.create`` returns a new object every
        # time, so QApplication runs unpolish/polish across the widget tree.
        target_style = self._theme_default_style if theme.name == "System default" else "Fusion"
        new_style = QtWidgets.QStyleFactory.create(target_style)
        if new_style is not None:
            app.setStyle(new_style)
        # IMPORTANT: only call ``app.setPalette`` here. We deliberately do not
        # call ``setPalette`` on individual widgets -- our custom palettes only
        # define ~15 of the ~20 colour roles (the 3D-shading roles Mid/Light/
        # Midlight/Dark/Shadow are unset on purpose), and QApplication knows
        # to merge missing roles from the style's standardPalette. QWidget
        # does not do that merge, so per-widget setPalette would zero out the
        # shading roles AND freeze each widget against future theme changes.
        new_palette = (
            app.style().standardPalette()
            if theme.name == "System default"
            else theme.build_palette()
        )
        app.setPalette(new_palette)
        # Stylesheet ``palette(...)`` references are resolved at parse time,
        # so re-set the stylesheet to recompute them against the new palette.
        css = self.styleSheet()
        if css:
            self.setStyleSheet("")
            self.setStyleSheet(css)
        # Re-tint matplotlib so the embedded canvas matches.
        mpl = theme.mpl
        self.preview_fig.set_facecolor(mpl["face"])
        # Edge color of the figure patch (the rectangle around the figure)
        # otherwise stays the matplotlib default white and shows as a thin
        # light border around the dark figure.
        try:
            self.preview_fig.patch.set_edgecolor(mpl["face"])
        except Exception:
            pass
        for ax in self.preview_fig.axes:
            ax.set_facecolor(mpl["axes_face"])
            ax.tick_params(colors=mpl["fg"])
            for spine in ax.spines.values():
                spine.set_color(mpl["fg"])
            if ax.title is not None:
                ax.title.set_color(mpl["fg"])
            if ax.xaxis.label is not None:
                ax.xaxis.label.set_color(mpl["fg"])
            if ax.yaxis.label is not None:
                ax.yaxis.label.set_color(mpl["fg"])
            leg = ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(mpl["axes_face"])
                leg.get_frame().set_edgecolor(mpl["fg"])
                if leg.get_title() is not None:
                    leg.get_title().set_color(mpl["fg"])
                for txt in leg.get_texts():
                    txt.set_color(mpl["fg"])
                # Patch handles in render_gds() are created with a hardcoded
                # black edge -- invisible on dark themes. Re-tint them here
                # so the legend swatches stay readable.
                for patch in leg.get_patches():
                    patch.set_edgecolor(mpl["fg"])
        self.preview_canvas.draw_idle()
        # The Qt canvas widget paints its own background under the figure;
        # when tight_layout shrinks the drawing area, that widget background
        # shows as a light border in dark mode. Tint it to match.
        self.preview_canvas.setStyleSheet(f"background-color: {mpl['face']};")
        # Navigation toolbar: follow the chrome too. Auto-pick light/dark
        # tones for hover/pressed states based on the theme brightness.
        if theme.is_dark:
            self.preview_toolbar.setStyleSheet(
                f"QToolBar {{ background-color: {mpl['face']}; "
                f"border: 0px; color: {mpl['fg']}; }}"
                f"QToolButton {{ background: transparent; color: {mpl['fg']}; }}"
                f"QToolButton:hover {{ background: {mpl['axes_face']}; }}"
                f"QToolButton:pressed {{ background: {mpl['grid']}; }}"
            )
        else:
            self.preview_toolbar.setStyleSheet("")
        # Dim/secondary text everywhere uses one theme-aware colour: the
        # docstring, the "default value" column, the module label under the
        # title, and the preview status line.
        soft_fg = self._secondary_text_color()
        self.preview_status.setStyleSheet(f"color:{soft_fg}")
        self.module_lbl.setStyleSheet(f"color:{soft_fg}")
        for lbl in self._secondary_labels:
            lbl.setStyleSheet(f"color:{soft_fg}")
        # Re-tint the param-form "type" labels: bright orange on dark, navy on light.
        type_color = self._type_label_color()
        for lbl in self._type_labels:
            lbl.setStyleSheet(f"color:{type_color}")

    def _type_label_color(self) -> str:
        # Bright orange against dark backgrounds, dark navy against light ones.
        return "#ff9f3a" if self._current_theme.is_dark else "#334466"

    def _secondary_text_color(self) -> str:
        # Used for docstrings, default-value cell, status lines -- any text
        # that should read as "dim/secondary" but still be legible. The
        # original ``#444`` / ``#888`` values were unreadable on dark themes.
        return "#c0c0c0" if self._current_theme.is_dark else "#555555"

    # ----- dock state persistence ----------------------------------------
    # Layout is stored on disk in ``menu/default_layout.json``: a small
    # ASCII document wrapping the binary ``QMainWindow.saveState()`` output
    # in base64 (the inner blob is opaque Qt-versioned binary, but the file
    # itself is text and version-controllable). Users can persist their
    # current arrangement via ``View -> Panels -> Save current as default``.

    @staticmethod
    def _decode_layout_file(path: Path) -> QtCore.QByteArray | None:
        try:
            text = path.read_text()
        except Exception:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != _LAYOUT_FORMAT_VERSION:
            return None
        b64 = payload.get("qstate_b64")
        if not isinstance(b64, str):
            return None
        try:
            raw = base64.b64decode(b64.encode("ascii"), validate=True)
        except Exception:
            return None
        return QtCore.QByteArray(raw)

    def _restore_dock_state(self) -> None:
        # Migration: a previous version saved a raw .bin blob next to .json.
        # Drop it once we have a json file in hand so it can't confuse users.
        if DEFAULT_LAYOUT_FILE.is_file():
            ba = self._decode_layout_file(DEFAULT_LAYOUT_FILE)
            if ba is not None and self.restoreState(ba):
                if _LEGACY_LAYOUT_FILE.is_file():
                    try:
                        _LEGACY_LAYOUT_FILE.unlink()
                    except Exception:
                        pass
                return
            self._log(
                f"warning: {DEFAULT_LAYOUT_FILE.name} could not be applied — "
                "regenerating from programmatic default"
            )
        # No usable file — write the current programmatic default to disk so
        # the next launch can read it back without re-deriving it.
        self._write_default_layout_file()
        if _LEGACY_LAYOUT_FILE.is_file():
            try:
                _LEGACY_LAYOUT_FILE.unlink()
            except Exception:
                pass

    def _write_default_layout_file(self) -> None:
        try:
            DEFAULT_LAYOUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            raw = bytes(self.saveState())
            payload = {
                "version": _LAYOUT_FORMAT_VERSION,
                "comment": (
                    "Layout state for the glayout pcell menu. "
                    "qstate_b64 is base64-encoded QMainWindow.saveState() output."
                ),
                "qstate_b64": base64.b64encode(raw).decode("ascii"),
            }
            DEFAULT_LAYOUT_FILE.write_text(json.dumps(payload, indent=2) + "\n")
            self._log(f"saved current layout to {DEFAULT_LAYOUT_FILE}")
        except Exception as exc:
            self._log(f"warning: failed to write {DEFAULT_LAYOUT_FILE.name}: {exc}")

    def _reset_layout(self) -> None:
        # Reset = re-apply programmatic default and reload from file (which
        # may have been customized by the user via Save current).
        self._apply_default_layout()
        if DEFAULT_LAYOUT_FILE.is_file():
            ba = self._decode_layout_file(DEFAULT_LAYOUT_FILE)
            if ba is None or not self.restoreState(ba):
                self._log(
                    f"warning: reset could not load {DEFAULT_LAYOUT_FILE.name}"
                )

    # ----- env settings ---------------------------------------------------
    _ENV_KEYS: tuple[tuple[str, str], ...] = (
        ("pdk_root", ""),
        ("drc_tool", "klayout"),
        ("klayout_path", ""),
        ("lvs_tool", "magic+netgen"),
        ("magic_path", ""),
        ("netgen_path", ""),
    )

    def _load_env_settings(self) -> dict[str, str]:
        return {
            key: self._settings.value(f"env/{key}", default, type=str)
            for key, default in self._ENV_KEYS
        }

    def _save_env_settings(self) -> None:
        for key, _ in self._ENV_KEYS:
            self._settings.setValue(f"env/{key}", self._env_settings.get(key, ""))

    def _apply_env_settings(self) -> None:
        # Snapshot the as-launched environment once so successive applies
        # rebuild from a clean baseline instead of accreting prefixes. The
        # detected defaults are captured against this baseline so the dialog
        # always shows the *shell* values, not whatever the user previously
        # forced via overrides.
        if not hasattr(self, "_baseline_path"):
            self._baseline_path = os.environ.get("PATH", "")
            self._baseline_pdk_root = os.environ.get("PDK_ROOT", "")
            self._baseline_cad_root = os.environ.get("CAD_ROOT", "")
            self._detected_env: dict[str, str] = _detect_env_defaults()

        pdk_root = self._env_settings.get("pdk_root", "").strip()
        if pdk_root:
            os.environ["PDK_ROOT"] = pdk_root
        elif self._baseline_pdk_root:
            os.environ["PDK_ROOT"] = self._baseline_pdk_root
        else:
            os.environ.pop("PDK_ROOT", None)

        extra_dirs: list[str] = []
        for key in ("klayout_path", "magic_path", "netgen_path"):
            p = self._env_settings.get(key, "").strip()
            if not p:
                continue
            d = str(Path(p).expanduser().parent)
            if d and d not in extra_dirs:
                extra_dirs.append(d)
        if extra_dirs:
            prefix = os.pathsep.join(extra_dirs)
            os.environ["PATH"] = (
                prefix + os.pathsep + self._baseline_path
                if self._baseline_path else prefix
            )
        else:
            os.environ["PATH"] = self._baseline_path

        # Magic loads its tcl libs from $CAD_ROOT/magic/tcl, so a 8.3.411
        # binary on PATH but $CAD_ROOT pointing at a 8.3.99 lib runs as
        # 8.3.99 (the launcher script literally substitutes CAD_ROOT into
        # the tcl path). Whenever the user pins a magic binary, derive
        # CAD_ROOT from its sibling lib dir if that layout looks right;
        # otherwise leave the as-launched value alone.
        magic_path = self._env_settings.get("magic_path", "").strip()
        new_cad_root = ""
        if magic_path:
            mp = Path(magic_path).expanduser().resolve()
            candidate = mp.parent.parent / "lib"
            if (candidate / "magic" / "tcl" / "magic.tcl").is_file():
                new_cad_root = str(candidate)
        if new_cad_root:
            os.environ["CAD_ROOT"] = new_cad_root
        elif self._baseline_cad_root:
            os.environ["CAD_ROOT"] = self._baseline_cad_root
        else:
            os.environ.pop("CAD_ROOT", None)

    def _open_env_settings(self) -> None:
        dlg = EnvSettingsDialog(
            self, dict(self._env_settings), dict(self._detected_env)
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        new = dlg.values()
        if new == self._env_settings:
            return
        self._env_settings = new
        self._save_env_settings()
        self._apply_env_settings()
        self._log("env settings updated")

    # ----- catalogue load -------------------------------------------------
    def _initial_load(self) -> None:
        # Drop every cached glayout.* module so importlib re-reads the
        # source from disk -- otherwise edits the user just made never
        # take effect (sys.modules cache wins). The previously-discovered
        # generator funcs / PDK objects belong to the modules we're about
        # to evict, so clear the "_last_*" state too; leaving stale refs
        # would let DRC/LVS run against half-unloaded modules.
        purged = [n for n in list(sys.modules) if n == "glayout" or n.startswith("glayout.")]
        for name in purged:
            sys.modules.pop(name, None)
        if purged:
            self._log(f"reload: evicted {len(purged)} cached glayout module(s)")
        self._last_comp = None
        self._last_gds = None
        self._last_spice = None
        self._last_pdk = None
        self._last_pdk_name = ""
        self._last_design_name = ""
        self._current = None

        self._log("Discovering pcells...")
        try:
            self._generators = discover_generators()
            self._pdks = discover_pdks()
        except Exception as exc:
            self._log(f"discovery failed: {exc}")
            self._log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Discovery failed", str(exc))
            return
        self._log(f"  {len(self._generators)} generator(s) found, {len(self._pdks)} PDK(s) loaded")

        self.pdk_combo.blockSignals(True)
        self.pdk_combo.clear()
        for name, _ in self._pdks:
            self.pdk_combo.addItem(name)
        self.pdk_combo.blockSignals(False)
        if not self._pdks:
            self._log("WARNING: no mapped PDK could be imported")

        self.tree.clear()
        by_cat: dict[str, list[GeneratorInfo]] = {}
        for g in self._generators:
            by_cat.setdefault(g.category, []).append(g)
        for cat in ("Primitives", "Elementary", "Composite"):
            if cat not in by_cat:
                continue
            top_item = QtWidgets.QTreeWidgetItem(self.tree, [cat])
            top_item.setExpanded(True)
            top_item.setFlags(top_item.flags() & ~QtCore.Qt.ItemIsSelectable)
            for g in by_cat[cat]:
                child = QtWidgets.QTreeWidgetItem(top_item, [g.name])
                child.setData(0, QtCore.Qt.UserRole, (g.category, g.name))
        self.tree.expandAll()

        # The param panel and check buttons reference the now-evicted
        # generator/PDK objects; wipe them so the user starts from a
        # clean slate after reload.
        self.title_lbl.setText("(select a cell on the left)")
        self.module_lbl.setText("")
        while self.param_layout.count():
            child = self.param_layout.takeAt(0)
            w = child.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._param_rows.clear()
        self.gen_btn.setEnabled(False)
        self._refresh_check_buttons()

    # ----- selection ------------------------------------------------------
    def _on_select(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        data = items[0].data(0, QtCore.Qt.UserRole)
        if not data:
            return
        category, name = data
        gen = next((g for g in self._generators if g.category == category and g.name == name), None)
        if gen is None:
            return
        self._show_generator(gen)

    def _show_generator(self, gen: GeneratorInfo) -> None:
        self._current = gen
        self.title_lbl.setText(f"{gen.category}: {gen.name}")
        self.module_lbl.setText(gen.module)

        # clear existing rows
        while self.param_layout.count():
            child = self.param_layout.takeAt(0)
            w = child.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._param_rows.clear()
        self._type_labels.clear()
        self._secondary_labels.clear()

        header = QtWidgets.QWidget()
        hh = QtWidgets.QHBoxLayout(header)
        hh.setContentsMargins(0, 0, 0, 0)
        for txt, w in (("param", 180), ("type", 240), ("value", 320), ("default", 200)):
            lbl = QtWidgets.QLabel(txt)
            f = lbl.font()
            f.setBold(True)
            lbl.setFont(f)
            lbl.setMinimumWidth(w)
            hh.addWidget(lbl)
        hh.addStretch(1)
        self.param_layout.addWidget(header)

        for p in gen.params:
            self.param_layout.addWidget(self._build_param_row(p))

        if gen.doc:
            sep = QtWidgets.QFrame()
            sep.setFrameShape(QtWidgets.QFrame.HLine)
            sep.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.param_layout.addWidget(sep)
            doc = QtWidgets.QLabel(gen.doc)
            doc.setWordWrap(True)
            doc.setStyleSheet(f"color:{self._secondary_text_color()}")
            self._secondary_labels.append(doc)
            self.param_layout.addWidget(doc)

        self.gen_btn.setEnabled(bool(self._pdks))
        self.drc_btn.setEnabled(False)
        self.lvs_btn.setEnabled(False)
        self.klayout_btn.setEnabled(False)

    def _build_param_row(self, p: ParamInfo) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)

        name_lbl = QtWidgets.QLabel(p.name)
        name_lbl.setMinimumWidth(180)
        h.addWidget(name_lbl)

        type_str = annotation_summary(p.annotation) or "(no annotation)"
        if not p.has_default:
            type_str += "  (required)"
        type_lbl = QtWidgets.QLabel(type_str)
        type_lbl.setMinimumWidth(240)
        type_lbl.setStyleSheet(f"color:{self._type_label_color()}")
        h.addWidget(type_lbl)
        self._type_labels.append(type_lbl)

        base, _ = underlying_types(p.annotation)
        widget: QtWidgets.QWidget
        if base is bool:
            cb = QtWidgets.QCheckBox()
            if p.has_default and isinstance(p.default, bool):
                cb.setChecked(p.default)
            cb.setMinimumWidth(320)
            widget = cb
        else:
            entry = QtWidgets.QLineEdit(_initial_text(p))
            entry.setMinimumWidth(320)
            hint = _placeholder_for(p)
            if hint:
                entry.setPlaceholderText(hint)
            widget = entry
        h.addWidget(widget)

        default_lbl = QtWidgets.QLabel(_format_default(p.default))
        default_lbl.setMinimumWidth(200)
        default_lbl.setStyleSheet(f"color:{self._secondary_text_color()}")
        self._secondary_labels.append(default_lbl)
        h.addWidget(default_lbl)
        h.addStretch(1)

        self._param_rows.append((p, widget))
        return row

    def _collect_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        missing: list[str] = []
        for p, widget in self._param_rows:
            if isinstance(widget, QtWidgets.QCheckBox):
                kwargs[p.name] = widget.isChecked()
                continue
            assert isinstance(widget, QtWidgets.QLineEdit)
            value = _parse_value(widget.text(), p)
            if value is _SENTINEL:
                if not p.has_default:
                    missing.append(p.name)
                continue
            kwargs[p.name] = value
        if missing:
            raise ValueError("missing required parameter(s): " + ", ".join(missing))
        return kwargs

    # ----- generation -----------------------------------------------------
    def _on_generate(self) -> None:
        if self._busy or self._current is None:
            return
        if not self._pdks:
            QtWidgets.QMessageBox.critical(self, "No PDK", "No mapped PDK is available.")
            return
        try:
            kwargs = self._collect_kwargs()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Bad parameter", str(exc))
            return
        pdk_name = self.pdk_combo.currentText()
        pdk = next((obj for n, obj in self._pdks if n == pdk_name), None)
        if pdk is None:
            QtWidgets.QMessageBox.critical(self, "No PDK", f"PDK {pdk_name!r} not loaded.")
            return
        out_dir = Path(self.outdir_edit.text()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        gen = self._current
        self._set_busy(True, "generating...")
        self._log(f"\n>>> {gen.name} on {pdk_name}")
        self._log(f"    kwargs = {kwargs!r}")

        worker = GeneratorWorker(gen, pdk, pdk_name, kwargs, out_dir)
        worker.artifacts.connect(self._on_generation_artifacts)
        self._launch_worker(worker, on_finished=self._after_generation)

    def _on_generation_artifacts(self, payload: dict) -> None:
        self._last_comp = payload["comp"]
        self._last_gds = payload["gds"]
        self._last_spice = payload["spice"]
        self._last_pdk = payload["pdk"]
        self._last_pdk_name = payload["pdk_name"]
        self._last_design_name = payload["design_name"]
        # artifacts and finished are two separate cross-thread queued
        # signals; refresh here so the buttons end up correct regardless
        # of which slot lands first in the main loop.
        self._refresh_check_buttons()
        if self._last_gds is not None:
            self._render_preview(self._last_gds)

    def _after_generation(self, success: bool, msg: str) -> None:
        self._set_busy(False, ("ok: " if success else "error: ") + msg)

    def _refresh_check_buttons(self) -> None:
        """Single source of truth for the DRC/LVS/KLayout button enable state."""
        if self._busy:
            self.drc_btn.setEnabled(False)
            self.lvs_btn.setEnabled(False)
            self.klayout_btn.setEnabled(False)
            self.save_svg_btn.setEnabled(False)
            return
        self.drc_btn.setEnabled(self._last_gds is not None)
        self.lvs_btn.setEnabled(
            self._last_gds is not None and self._last_spice is not None
        )
        self.klayout_btn.setEnabled(self._last_gds is not None)
        self.save_svg_btn.setEnabled(self._last_gds is not None)

    def _on_save_svg(self) -> None:
        """Export the last-generated GDS as an SVG via gdstk's built-in writer."""
        if self._last_gds is None:
            return
        stem = self._last_design_name or self._last_gds.stem
        suggested = str(self._last_gds.parent / f"{stem}.svg")
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save SVG", suggested, "SVG files (*.svg);;All files (*)"
        )
        if not fname:
            return
        out = Path(fname)
        if out.suffix.lower() != ".svg":
            out = out.with_suffix(".svg")
        png_out = out.with_suffix(".png")
        try:
            import gdstk
            lib = gdstk.read_gds(str(self._last_gds))
            tops = lib.top_level()
            if not tops:
                raise RuntimeError("GDS has no top-level cell")
            # Build shape_style for layers actually present in the GDS,
            # using the curated palette shared with docs/render_figures.py.
            shape_style: dict[tuple[int, int], dict] = {}
            for cell in lib.cells:
                for poly in cell.polygons:
                    key = (poly.layer, poly.datatype)
                    shape_style.setdefault(key, LAYER_STYLES.get(key, DEFAULT_STYLE))
                for path in cell.paths:
                    for layer, datatype in zip(path.layers, path.datatypes):
                        key = (layer, datatype)
                        shape_style.setdefault(key, LAYER_STYLES.get(key, DEFAULT_STYLE))
            tops[0].write_svg(str(out), background="#ffffff", shape_style=shape_style)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Save SVG failed", f"could not write SVG: {exc}",
            )
            self._log(f"!! save SVG failed: {exc}")
            return
        self._log(f"saved SVG: {out}")
        # Companion PNG: rasterize the SVG we just wrote using QtSvg so the
        # PNG and SVG are byte-for-byte the same artwork (no matplotlib
        # re-render, no separate styling).
        try:
            from PySide6 import QtSvg
            renderer = QtSvg.QSvgRenderer(str(out))
            if not renderer.isValid():
                raise RuntimeError("QSvgRenderer rejected the SVG")
            size = renderer.defaultSize()
            if size.width() <= 0 or size.height() <= 0:
                raise RuntimeError("SVG has no intrinsic size")
            # 2x upscale keeps text/lines crisp without ballooning file size.
            scale = 2
            img = QtGui.QImage(
                size.width() * scale, size.height() * scale,
                QtGui.QImage.Format_ARGB32,
            )
            img.fill(QtCore.Qt.white)
            painter = QtGui.QPainter(img)
            try:
                renderer.render(painter)
            finally:
                painter.end()
            if not img.save(str(png_out), "PNG"):
                raise RuntimeError(f"QImage.save returned False for {png_out}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Save PNG failed",
                f"SVG saved, but PNG companion failed: {exc}",
            )
            self._log(f"!! save PNG failed: {exc}")
            return
        self._log(f"saved PNG: {png_out}")

    def _on_open_klayout(self) -> None:
        """Spawn ``klayout -e <gds>`` with KLAYOUT_PATH pinned at the PDK
        variant so klayout loads the right tech, layer view, and macros."""
        if self._last_gds is None:
            return
        if not shutil.which("klayout"):
            QtWidgets.QMessageBox.critical(
                self, "klayout missing",
                "klayout is not on PATH. Configure it under Tools -> Env settings.",
            )
            return
        env = os.environ.copy()
        variant = self._PDK_LAYER_MAP_VARIANT.get(self._last_pdk_name)
        pdk_root = env.get("PDK_ROOT", "").strip()
        if variant and pdk_root:
            klayout_path = Path(pdk_root).expanduser() / variant / "libs.tech" / "klayout"
            if klayout_path.is_dir():
                env["KLAYOUT_PATH"] = str(klayout_path)
                self._log(f"KLAYOUT_PATH={klayout_path}")
            else:
                self._log(
                    f"warning: {klayout_path} does not exist; "
                    "launching klayout without KLAYOUT_PATH override"
                )
        try:
            # Detach so the GUI doesn't block / inherit klayout's lifetime.
            subprocess.Popen(
                ["klayout", "-e", str(self._last_gds)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._log(f"launched: klayout -e {self._last_gds}")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "klayout launch failed", f"could not start klayout: {exc}",
            )

    def _render_preview(self, gds_path: Path) -> None:
        layer_names = None
        if self.preview_show_layer_names.isChecked():
            layer_names = self._layer_names_for(self._last_pdk_name)
        try:
            summary = render_gds(gds_path, self.preview_fig, layer_names=layer_names)
        except Exception as exc:
            self.preview_status.setText(f"render failed: {exc}")
            self._log(f"!! preview render failed: {exc}")
            self._log(traceback.format_exc())
            return
        self._apply_theme(self._current_theme)
        self.preview_canvas.draw_idle()
        self.preview_status.setText(f"{gds_path.name} — {summary}")

    # Per-PDK variant directory under $PDK_ROOT that contains the klayout
    # .map file. Mirrors pdk_check._PDK_CONFIGS so a user-configured
    # PDK_ROOT line up with what the DRC/LVS dispatcher uses.
    _PDK_LAYER_MAP_VARIANT: dict[str, str] = {
        "sky130": "sky130B",
        "gf180":  "gf180mcuC",
        "ihp130": "ihp130",
    }

    def _layer_names_for(self, pdk_name: str) -> dict[tuple[int, int], str]:
        """Lazy-load the klayout layer map for ``pdk_name`` from PDK_ROOT.

        Returns ``{}`` and logs a one-line warning if the file is missing
        so the legend silently falls back to bare layer/dt numbers rather
        than blowing up the preview.
        """
        if pdk_name in self._layer_name_maps:
            return self._layer_name_maps[pdk_name]
        variant = self._PDK_LAYER_MAP_VARIANT.get(pdk_name)
        pdk_root = os.environ.get("PDK_ROOT", "").strip()
        result: dict[tuple[int, int], str] = {}
        if variant and pdk_root:
            tech_dir = Path(pdk_root).expanduser() / variant / "libs.tech" / "klayout" / "tech"
            map_files = sorted(tech_dir.glob("*.map"))
            if not map_files:
                self._log(f"layer-name map not found under {tech_dir}")
            else:
                result = parse_klayout_map(map_files[0])
                if not result:
                    self._log(f"layer-name map at {map_files[0]} parsed empty")
        elif pdk_name:
            self._log(
                f"no layer-name mapping configured for PDK {pdk_name!r} "
                "(extend _PDK_LAYER_MAP_VARIANT in start_gui.py)"
            )
        self._layer_name_maps[pdk_name] = result
        return result

    def _on_layer_name_toggle(self, _checked: bool) -> None:
        if self._last_gds is not None:
            self._render_preview(self._last_gds)

    # ----- DRC ------------------------------------------------------------
    def _on_run_drc(self) -> None:
        if self._busy or self._last_gds is None or self._last_pdk is None:
            return
        if not shutil.which("klayout"):
            QtWidgets.QMessageBox.critical(
                self, "klayout missing",
                "klayout is not on PATH. Install klayout to run DRC.",
            )
            return
        self._set_busy(True, "DRC running...")
        self._set_text(self.drc_text, "running klayout DRC...\n")
        out_dir = Path(self.outdir_edit.text()).expanduser() / "drc"
        worker = DrcWorker(self._last_pdk, self._last_gds, out_dir)
        self._launch_worker(worker, on_finished=self._after_check)

    # ----- LVS ------------------------------------------------------------
    def _on_run_lvs(self) -> None:
        if (self._busy or self._last_gds is None or self._last_spice is None
                or self._last_pdk is None):
            return
        missing = [t for t in ("magic", "netgen") if not shutil.which(t)]
        if missing:
            QtWidgets.QMessageBox.critical(
                self, "tools missing",
                "missing on PATH: " + ", ".join(missing)
                + "\nInstall magic and netgen to run LVS.",
            )
            return
        pdk_root = os.environ.get("PDK_ROOT")
        if not pdk_root or not Path(pdk_root).is_dir():
            QtWidgets.QMessageBox.critical(
                self, "PDK_ROOT missing",
                "PDK_ROOT must point to a real PDK install for LVS.\n"
                f"current value: {pdk_root or '(unset)'}",
            )
            return
        self._set_busy(True, "LVS running...")
        self._set_text(self.lvs_text, "running magic+netgen LVS...\n")
        worker = LvsWorker(
            self._last_pdk, self._last_gds, self._last_spice,
            self._last_design_name, pdk_root,
        )
        self._launch_worker(worker, on_finished=self._after_check)

    def _after_check(self, success: bool, msg: str) -> None:
        self._set_busy(False, ("ok: " if success else "error: ") + msg)

    # ----- worker plumbing ------------------------------------------------
    def _launch_worker(self, worker: _Worker, *, on_finished) -> None:
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        worker.log.connect(self._log)
        worker.text_out.connect(self._on_text_out)

        # Single in-flight worker at a time -- _on_generate / _on_run_*
        # already gate on self._busy. Track it so the busy dialog's
        # Cancel button can call worker.cancel().
        self._active_worker = worker

        def _on_done(success: bool, msg: str):
            if self._active_worker is worker:
                self._active_worker = None
            on_finished(success, msg)
            thread.quit()

        worker.finished.connect(_on_done)
        thread.started.connect(worker.run)
        # cleanup
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._workers.remove((thread, worker)) if (thread, worker) in self._workers else None)
        self._workers.append((thread, worker))
        thread.start()

    @QtCore.Slot(str, str)
    def _on_text_out(self, key: str, text: str) -> None:
        if key == "drc":
            self._set_text(self.drc_text, text)
        elif key == "lvs":
            self._set_text(self.lvs_text, text)

    # ----- helpers --------------------------------------------------------
    def _set_busy(self, busy: bool, status_text: str = "") -> None:
        self._busy = busy
        if busy:
            self.gen_btn.setEnabled(False)
            self._show_busy_dialog(status_text or "working...")
        else:
            self.gen_btn.setEnabled(self._current is not None and bool(self._pdks))
            self._hide_busy_dialog()
        self._refresh_check_buttons()
        self.status_lbl.setText(status_text)

    # ----- busy spinner ---------------------------------------------------
    def _show_busy_dialog(self, label: str) -> None:
        """Pop a non-modal indeterminate-progress dialog so a long
        DRC/LVS/Generate run is visible and cancellable. Reusing the
        same dialog instance avoids repeated show/hide flicker."""
        if not hasattr(self, "_busy_dialog") or self._busy_dialog is None:
            dlg = QtWidgets.QProgressDialog(label, "Cancel", 0, 0, self)
            dlg.setWindowTitle("Working...")
            dlg.setWindowModality(QtCore.Qt.NonModal)
            dlg.setAutoClose(False)
            dlg.setAutoReset(False)
            dlg.setMinimumDuration(0)
            dlg.canceled.connect(self._on_busy_cancel)
            self._busy_dialog = dlg
        else:
            self._busy_dialog.setLabelText(label)
            self._busy_dialog.reset()  # clear any prior cancel state
        # reset() above calls cancel(); re-show explicitly.
        self._busy_dialog.show()
        self._busy_dialog.raise_()

    def _hide_busy_dialog(self) -> None:
        if getattr(self, "_busy_dialog", None) is not None:
            self._busy_dialog.hide()

    def _on_busy_cancel(self) -> None:
        worker = getattr(self, "_active_worker", None)
        if worker is None:
            return
        self._log("cancel requested")
        worker.cancel()
        # Don't clear _busy here -- the worker's finished signal will
        # arrive with an error verdict, which routes through the normal
        # _after_check / _after_generation path and clears busy/dialog.

    @QtCore.Slot(str)
    def _log(self, msg: str) -> None:
        self.log_widget.appendPlainText(msg)

    def _set_text(self, widget: QtWidgets.QPlainTextEdit, text: str) -> None:
        widget.setPlainText(text)
        widget.moveCursor(QtGui.QTextCursor.End)


def _apply_persisted_ui_scale() -> None:
    # QSettings can be constructed with explicit org/app before QApplication
    # exists; QT_SCALE_FACTOR must be in the environment before Qt reads it.
    s = QtCore.QSettings("glayout", "pcell-menu")
    raw = s.value(_UI_SCALE_KEY, _UI_SCALE_SYSTEM, type=str) or _UI_SCALE_SYSTEM
    if raw == _UI_SCALE_SYSTEM:
        return
    try:
        pct = int(raw)
    except (TypeError, ValueError):
        return
    if pct not in _UI_SCALE_PERCENTS:
        return
    os.environ["QT_SCALE_FACTOR"] = f"{pct / 100:.2f}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Be robust on systems where Qt's xcb plugin is fussy.
    if "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ:
        try:
            import PySide6  # noqa: F401
            plugins = Path(__import__("PySide6").__file__).parent / "Qt" / "plugins"
            if plugins.is_dir():
                os.environ.setdefault("QT_PLUGIN_PATH", str(plugins))
        except Exception:
            pass
    _apply_persisted_ui_scale()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = PcellMenu()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
