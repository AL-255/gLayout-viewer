"""Color palettes for the pcell menu.

Each theme returns:
  - a ``QPalette`` to install on the QApplication (Fusion style honours every
    palette role we care about)
  - a small ``mpl`` dict used to retint the matplotlib preview so the embedded
    canvas doesn't look stranded against a dark window chrome
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6 import QtGui
from PySide6.QtGui import QColor, QPalette


@dataclass(frozen=True)
class Theme:
    name: str
    is_dark: bool
    build_palette: Callable[[], QPalette]
    mpl: dict


def _palette_from(roles: dict[QPalette.ColorRole, QColor],
                  disabled: dict[QPalette.ColorRole, QColor] | None = None) -> QPalette:
    pal = QPalette()
    for role, color in roles.items():
        pal.setColor(QPalette.Active, role, color)
        pal.setColor(QPalette.Inactive, role, color)
    if disabled:
        for role, color in disabled.items():
            pal.setColor(QPalette.Disabled, role, color)
    return pal


# ---- system / Fusion light --------------------------------------------------

def _system_palette() -> QPalette:
    # Empty palette → Qt falls back to the platform default after we restore
    # the system style.
    return QPalette()


def _fusion_light() -> QPalette:
    base = QColor("#fafafa")
    alt = QColor("#f0f0f0")
    fg = QColor("#202020")
    accent = QColor("#3367d6")
    return _palette_from(
        {
            QPalette.Window: QColor("#f4f4f5"),
            QPalette.WindowText: fg,
            QPalette.Base: base,
            QPalette.AlternateBase: alt,
            QPalette.ToolTipBase: QColor("#ffffe1"),
            QPalette.ToolTipText: fg,
            QPalette.Text: fg,
            QPalette.Button: QColor("#e7e7ea"),
            QPalette.ButtonText: fg,
            QPalette.BrightText: QColor("#ff0000"),
            QPalette.Link: accent,
            QPalette.Highlight: accent,
            QPalette.HighlightedText: QColor("#ffffff"),
        },
        disabled={
            QPalette.Text: QColor("#888"),
            QPalette.ButtonText: QColor("#888"),
            QPalette.WindowText: QColor("#888"),
        },
    )


# ---- Fusion dark ------------------------------------------------------------

def _fusion_dark() -> QPalette:
    bg = QColor("#2b2b2b")
    base = QColor("#232323")
    alt = QColor("#2f2f2f")
    fg = QColor("#dcdcdc")
    accent = QColor("#3d8bff")
    return _palette_from(
        {
            QPalette.Window: bg,
            QPalette.WindowText: fg,
            QPalette.Base: base,
            QPalette.AlternateBase: alt,
            QPalette.ToolTipBase: QColor("#3a3a3a"),
            QPalette.ToolTipText: fg,
            QPalette.Text: fg,
            QPalette.Button: QColor("#3a3a3a"),
            QPalette.ButtonText: fg,
            QPalette.BrightText: QColor("#ff7777"),
            QPalette.Link: accent,
            QPalette.Highlight: accent,
            QPalette.HighlightedText: QColor("#ffffff"),
            QPalette.PlaceholderText: QColor("#888"),
        },
        disabled={
            QPalette.Text: QColor("#777"),
            QPalette.ButtonText: QColor("#777"),
            QPalette.WindowText: QColor("#777"),
        },
    )


# ---- Solarized --------------------------------------------------------------

def _solarized_light() -> QPalette:
    base = QColor("#fdf6e3")
    base2 = QColor("#eee8d5")
    fg = QColor("#586e75")
    accent = QColor("#268bd2")
    return _palette_from(
        {
            QPalette.Window: base2,
            QPalette.WindowText: fg,
            QPalette.Base: base,
            QPalette.AlternateBase: QColor("#f5efdc"),
            QPalette.ToolTipBase: base,
            QPalette.ToolTipText: fg,
            QPalette.Text: fg,
            QPalette.Button: QColor("#e8e2cf"),
            QPalette.ButtonText: fg,
            QPalette.BrightText: QColor("#dc322f"),
            QPalette.Link: accent,
            QPalette.Highlight: accent,
            QPalette.HighlightedText: base,
        },
    )


def _solarized_dark() -> QPalette:
    bg = QColor("#002b36")
    base = QColor("#073642")
    fg = QColor("#93a1a1")
    accent = QColor("#268bd2")
    return _palette_from(
        {
            QPalette.Window: bg,
            QPalette.WindowText: fg,
            QPalette.Base: base,
            QPalette.AlternateBase: QColor("#0c3a47"),
            QPalette.ToolTipBase: QColor("#0c3a47"),
            QPalette.ToolTipText: fg,
            QPalette.Text: QColor("#cfd8d8"),
            QPalette.Button: QColor("#0c3a47"),
            QPalette.ButtonText: fg,
            QPalette.BrightText: QColor("#dc322f"),
            QPalette.Link: accent,
            QPalette.Highlight: accent,
            QPalette.HighlightedText: bg,
            QPalette.PlaceholderText: QColor("#586e75"),
        },
    )


# ---- Nord (dark) ------------------------------------------------------------

def _nord() -> QPalette:
    bg = QColor("#2e3440")
    base = QColor("#3b4252")
    alt = QColor("#434c5e")
    fg = QColor("#e5e9f0")
    accent = QColor("#88c0d0")
    return _palette_from(
        {
            QPalette.Window: bg,
            QPalette.WindowText: fg,
            QPalette.Base: base,
            QPalette.AlternateBase: alt,
            QPalette.ToolTipBase: alt,
            QPalette.ToolTipText: fg,
            QPalette.Text: fg,
            QPalette.Button: alt,
            QPalette.ButtonText: fg,
            QPalette.BrightText: QColor("#bf616a"),
            QPalette.Link: accent,
            QPalette.Highlight: QColor("#5e81ac"),
            QPalette.HighlightedText: QColor("#eceff4"),
            QPalette.PlaceholderText: QColor("#7b8494"),
        },
    )


# ---- Dracula ----------------------------------------------------------------

def _dracula() -> QPalette:
    bg = QColor("#282a36")
    base = QColor("#1e1f29")
    alt = QColor("#343746")
    fg = QColor("#f8f8f2")
    accent = QColor("#bd93f9")
    return _palette_from(
        {
            QPalette.Window: bg,
            QPalette.WindowText: fg,
            QPalette.Base: base,
            QPalette.AlternateBase: alt,
            QPalette.ToolTipBase: alt,
            QPalette.ToolTipText: fg,
            QPalette.Text: fg,
            QPalette.Button: alt,
            QPalette.ButtonText: fg,
            QPalette.BrightText: QColor("#ff5555"),
            QPalette.Link: accent,
            QPalette.Highlight: QColor("#6272a4"),
            QPalette.HighlightedText: fg,
            QPalette.PlaceholderText: QColor("#6272a4"),
        },
    )


# ---- registry ---------------------------------------------------------------

THEMES: list[Theme] = [
    Theme("System default", False, _system_palette,
          {"face": "white", "axes_face": "white", "fg": "#222", "grid": "#e0e0e0"}),
    Theme("Fusion light", False, _fusion_light,
          {"face": "#fafafa", "axes_face": "#ffffff", "fg": "#222", "grid": "#dadada"}),
    Theme("Fusion dark", True, _fusion_dark,
          {"face": "#2b2b2b", "axes_face": "#1f1f1f", "fg": "#dcdcdc", "grid": "#404040"}),
    Theme("Solarized light", False, _solarized_light,
          {"face": "#eee8d5", "axes_face": "#fdf6e3", "fg": "#586e75", "grid": "#d8d2c0"}),
    Theme("Solarized dark", True, _solarized_dark,
          {"face": "#002b36", "axes_face": "#073642", "fg": "#93a1a1", "grid": "#1c4854"}),
    Theme("Nord", True, _nord,
          {"face": "#2e3440", "axes_face": "#3b4252", "fg": "#e5e9f0", "grid": "#4c566a"}),
    Theme("Dracula", True, _dracula,
          {"face": "#282a36", "axes_face": "#1e1f29", "fg": "#f8f8f2", "grid": "#44475a"}),
]


def get_theme(name: str) -> Theme:
    for t in THEMES:
        if t.name == name:
            return t
    return THEMES[0]
