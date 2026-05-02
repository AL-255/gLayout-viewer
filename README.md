# gLayout-viewer

PySide6 GUI for browsing and generating
[gLayout](https://github.com/idea-fasoc/gLayout) pcells.

The catalogue (cells, primitives, parameters, defaults) is introspected at
runtime from the `glayout` package via `discovery.py` -- adding or renaming
cells in the upstream repo does not require any changes to this viewer.

## Features

- Dockable sub-window interface (`QDockWidget`s): cells tree, parameter
  form, GDS preview (matplotlib), DRC report, LVS report, log strip.
- Drag-to-pan and scroll-to-zoom on the preview canvas.
- Theme picker (Fusion light/dark, Solarized, Nord, Dracula, system) with
  matplotlib retinting.
- Layout persistence in `default_layout.json` (ASCII; base64-wrapped
  `QMainWindow.saveState()` payload).
- DRC via `pdk.drc(...)` (klayout) and LVS via `pdk.lvs_netgen(...)`
  (magic + netgen). Pre-flight checks for missing tools / `PDK_ROOT`.

## Running

Used as a submodule of a gLayout checkout:

```sh
./start_gui.sh
```

The launcher prefers `$GLAYOUT_PYTHON`, then a `gLayout` conda env, then
the system `python3`. It sets `PYTHONPATH=<repo>/src` and stubs `PDK_ROOT`
if unset (override `PDK_ROOT` to point at a real PDK install for LVS).

## Required Python deps

`PySide6`, `matplotlib`, `gdstk` (in addition to glayout's own deps).
