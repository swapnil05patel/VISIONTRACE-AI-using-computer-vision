"""
Alias package so that `artifacts.api_server` resolves to the
original `artifacts/api-server` directory (which contains a hyphen).
"""

import importlib.util
import pathlib
import sys

# Resolve the real dash‑folder location
_real_path = pathlib.Path(__file__).resolve().parent.parent / "api-server"

# Add the dash‑folder to sys.path so sub‑packages can be imported
if str(_real_path) not in sys.path:
    sys.path.insert(0, str(_real_path))
# Also add the ml subdirectory so sibling modules like config can be imported directly
_ml_path = _real_path / "ml"
if str(_ml_path) not in sys.path:
    sys.path.insert(0, str(_ml_path))
# Ensure the dash-folder is part of this package's __path__ for submodule resolution
if str(_real_path) not in __path__:
    __path__.append(str(_real_path))
# Also add ml folder to __path__ for sibling imports
if str(_ml_path) not in __path__:
    __path__.append(str(_ml_path))

# Automatically expose sub‑modules (e.g. `ml`) by importing them lazily
def __getattr__(name):
    # Attempt to import the module from the real folder
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(name) from exc
