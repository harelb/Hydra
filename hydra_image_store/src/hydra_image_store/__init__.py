"""ADT4 compatibility shim: ``hydra_image_store`` = ``hydra_python.image_store``.

The image store moved into Hydra's python package (``hydra_python.image_store``, CLI
``hydra image-store``). ADT4 code imports ``hydra_image_store`` and finds it at
``<hydra>/hydra_image_store/src``; this shim keeps that working unchanged:

- ``hydra_python`` is taken from the environment, else from this checkout's
  ``hydra/python/src`` (ADT4 does not build the bindings; the pure-python image store
  does not need them, so their import warning is silenced here);
- every submodule is the same module object under both names
  (``hydra_image_store.store is hydra_python.image_store.store``), so monkeypatching and
  isinstance checks behave as before;
- ``hydra-image-store`` / ``python -m hydra_image_store`` run the same click CLI.

ADT4-only (branch adt4/image-store-extras); not part of upstream Hydra. Drop it once
the consumers import ``hydra_python.image_store``.
"""

import importlib
import pathlib
import pkgutil
import sys
import warnings


def _import_impl():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="hydra_python: compiled bindings not available"
        )
        try:
            return importlib.import_module("hydra_python.image_store")
        except ImportError:
            if "hydra_python" in sys.modules:
                raise  # another hydra_python without image_store comes first
            src = (
                pathlib.Path(__file__).resolve().parents[3] / "hydra" / "python" / "src"
            )
            if not (src / "hydra_python" / "image_store").is_dir():
                raise
            sys.path.insert(0, str(src))
            return importlib.import_module("hydra_python.image_store")


_impl = _import_impl()

for _info in pkgutil.iter_modules(_impl.__path__):
    if _info.name == "__main__":
        continue
    _mod = importlib.import_module(f"{_impl.__name__}.{_info.name}")
    sys.modules[f"{__name__}.{_info.name}"] = _mod
    globals()[_info.name] = _mod

from hydra_python.image_store import *  # noqa: E402, F403
from hydra_python.image_store import __all__  # noqa: E402, F401

#: appended to the CLI error for a store that still holds sub-keyframes (frame kind 1)
CONVERT_HINT = (
    "\nThis is an ADT4 store from before hydra_python.image_store (sub-keyframes, frame "
    "kind 1). Convert it once (close its writers first):\n"
    "  hydra-image-store-adt4 convert STORE"
)


def main(argv=None) -> int:
    """``hydra-image-store`` entry point (``hydra image-store``). An unconverted ADT4
    store gets a hint to ``hydra-image-store-adt4 convert`` it."""
    import click

    try:
        rv = _impl.cli.cli.main(
            args=argv, prog_name="hydra-image-store", standalone_mode=False
        )
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except click.ClickException as exc:
        if isinstance(exc.__cause__, _impl.FrameKindError) and (
            "frame kind 1 " in str(exc.__cause__)
        ):
            exc.message += CONVERT_HINT
        exc.show()
        return exc.exit_code
    except click.Abort:
        click.echo("Aborted!", err=True)
        return 1
    return rv if isinstance(rv, int) else 0
