# hydra_image_store (ADT4 compatibility shim)

The image store is `hydra_python.image_store` (`hydra/python/src/hydra_python/image_store`,
CLI `hydra image-store`, docs in its README). This directory only keeps the old import
name, path and `hydra-image-store` command working for ADT4 code; it is not part of
upstream Hydra.

- `PYTHONPATH=<hydra>/hydra_image_store/src` works as before: the shim adds
  `<hydra>/hydra/python/src` when `hydra_python` is not importable.
- A pip install of this directory alone needs `hydra_python` installed too
  (`pip install <hydra>/hydra`; without a Hydra build only the pure-python parts).
- `hydra_image_store.X is hydra_python.image_store.X` for every submodule.
- `tests` links to `hydra/python/tests/image_store`: ADT4 tests import the indexer's test
  writers (`raw_logs`, `test_tail`) from `<hydra_image_store.__file__>/../../../tests`.
