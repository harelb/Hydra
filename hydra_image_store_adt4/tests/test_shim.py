"""The ADT4 shim exposes hydra_python.image_store under the old name."""

import subprocess
import sys


def test_same_modules():
    import hydra_image_store.store
    import hydra_image_store.tail
    import hydra_python.image_store as impl
    from hydra_image_store.exr import read_exr

    import hydra_image_store

    assert hydra_image_store.store is impl.store
    assert hydra_image_store.tail is impl.tail
    assert hydra_image_store.ImageStore is impl.ImageStore
    assert read_exr is impl.exr.read_exr
    assert set(hydra_image_store.__all__) == set(impl.__all__)


def test_cli_and_no_bindings_warning():
    out = subprocess.run(
        [sys.executable, "-W", "error", "-m", "hydra_image_store", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "hydra-image-store" in out.stdout and "tail" in out.stdout
