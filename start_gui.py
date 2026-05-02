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
from gds_viewer import render_gds  # noqa: E402
from themes import THEMES, Theme, get_theme  # noqa: E402

OUT_DIR = _HERE.parent / "out"
DEFAULT_LAYOUT_FILE = _HERE / "default_layout.json"
_LEGACY_LAYOUT_FILE = _HERE / "default_layout.bin"
_LAYOUT_FORMAT_VERSION = 1
_SENTINEL = object()
_DEFAULT_THEME = "Fusion light"


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
    return a ``ComponentReference`` (e.g. ``diff_pair_ibias``). The wrapper
    preserves ports and the ``info["netlist"]`` payload.
    """
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
    def __init__(self, pdk: Any, gds: Path, out_dir: Path) -> None:
        super().__init__()
        self.pdk = pdk
        self.gds = gds
        self.out_dir = out_dir

    @QtCore.Slot()
    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        captured = io.StringIO()
        clean = None
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                clean = self.pdk.drc(self.gds, output_dir_or_file=self.out_dir)
        except Exception as exc:
            text = captured.getvalue() + f"\n\n!! DRC raised: {exc}\n" + traceback.format_exc()
            self.text_out.emit("drc", text)
            self.finished.emit(False, f"DRC error: {exc}")
            return
        verdict = "CLEAN" if clean else "VIOLATIONS"
        head = f"DRC verdict: {verdict}\nreport dir: {self.out_dir}\n\n"
        self.text_out.emit("drc", head + captured.getvalue())
        self.finished.emit(bool(clean), f"DRC {verdict.lower()}")


class LvsWorker(_Worker):
    def __init__(self, pdk: Any, gds: Path, spice: Path,
                 design_name: str, pdk_root: str) -> None:
        super().__init__()
        self.pdk = pdk
        self.gds = gds
        self.spice = spice
        self.design_name = design_name
        self.pdk_root = pdk_root

    @QtCore.Slot()
    def run(self) -> None:
        captured = io.StringIO()
        result: Any = None
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                result = self.pdk.lvs_netgen(
                    layout=self.gds,
                    design_name=self.design_name,
                    pdk_root=self.pdk_root,
                    netlist=self.spice,
                )
        except Exception as exc:
            text = captured.getvalue() + f"\n\n!! LVS raised: {exc}\n" + traceback.format_exc()
            self.text_out.emit("lvs", text)
            self.finished.emit(False, f"LVS error: {exc}")
            return
        head = f"LVS result:\n{result!r}\n\n"
        self.text_out.emit("lvs", head + captured.getvalue())
        success = True
        if isinstance(result, dict):
            code = result.get("code") or result.get("return_code")
            if isinstance(code, int) and code != 0:
                success = False
        self.finished.emit(success, "LVS done" if success else "LVS mismatch")


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
        self.preview_fig = Figure(figsize=(6.0, 4.5), dpi=110)
        self.preview_canvas = FigureCanvasQTAgg(self.preview_fig)
        self.preview_toolbar = NavigationToolbar2QT(self.preview_canvas, body)
        v.addWidget(self.preview_toolbar)
        v.addWidget(self.preview_canvas, stretch=1)
        self._install_preview_interactions()
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

        # Bottom strip belongs entirely to the log dock; claim the bottom
        # corners so the log stretches edge-to-edge.
        self.setCorner(QtCore.Qt.BottomLeftCorner, QtCore.Qt.BottomDockWidgetArea)
        self.setCorner(QtCore.Qt.BottomRightCorner, QtCore.Qt.BottomDockWidgetArea)

        # Top row across the rest of the window: cells | params | (tab group
        # with preview/drc/lvs). Place them in the left dock area and split
        # left-to-right so they form three columns.
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, cells)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, params)
        self.splitDockWidget(cells, params, QtCore.Qt.Horizontal)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, preview)
        self.splitDockWidget(params, preview, QtCore.Qt.Horizontal)
        # Stack drc and lvs onto preview as tabs in the same slot.
        self.tabifyDockWidget(preview, drc)
        self.tabifyDockWidget(preview, lvs)
        preview.raise_()  # preview is the default visible tab

        # Log is anchored at the bottom and not allowed to leave that area.
        log.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, log)

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

    def _on_theme_chosen(self, theme: Theme) -> None:
        self._current_theme = theme
        self._settings.setValue("theme", theme.name)
        self._apply_theme(theme)
        action = self._theme_actions.get(theme.name)
        if action is not None:
            action.setChecked(True)

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
        b64 = payload.get("qstate_b64") if isinstance(payload, dict) else None
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

    # ----- catalogue load -------------------------------------------------
    def _initial_load(self) -> None:
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
        if self._last_gds is not None:
            self._render_preview(self._last_gds)

    def _after_generation(self, success: bool, msg: str) -> None:
        self._set_busy(False, ("ok: " if success else "error: ") + msg)
        if success:
            self.drc_btn.setEnabled(self._last_gds is not None)
            self.lvs_btn.setEnabled(self._last_gds is not None and self._last_spice is not None)

    def _render_preview(self, gds_path: Path) -> None:
        try:
            summary = render_gds(gds_path, self.preview_fig)
        except Exception as exc:
            self.preview_status.setText(f"render failed: {exc}")
            self._log(f"!! preview render failed: {exc}")
            self._log(traceback.format_exc())
            return
        self._apply_theme(self._current_theme)
        self.preview_canvas.draw_idle()
        self.preview_status.setText(f"{gds_path.name} — {summary}")

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
        self.drc_btn.setEnabled(self._last_gds is not None)
        self.lvs_btn.setEnabled(self._last_gds is not None and self._last_spice is not None)

    # ----- worker plumbing ------------------------------------------------
    def _launch_worker(self, worker: _Worker, *, on_finished) -> None:
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        worker.log.connect(self._log)
        worker.text_out.connect(self._on_text_out)

        def _on_done(success: bool, msg: str):
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
            self.drc_btn.setEnabled(False)
            self.lvs_btn.setEnabled(False)
        else:
            self.gen_btn.setEnabled(self._current is not None and bool(self._pdks))
        self.status_lbl.setText(status_text)

    @QtCore.Slot(str)
    def _log(self, msg: str) -> None:
        self.log_widget.appendPlainText(msg)

    def _set_text(self, widget: QtWidgets.QPlainTextEdit, text: str) -> None:
        widget.setPlainText(text)
        widget.moveCursor(QtGui.QTextCursor.End)


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
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = PcellMenu()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
