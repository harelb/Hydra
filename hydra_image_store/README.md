# hydra_image_store

Pure-python indexer and reader for the Hydra image store: one SQLite index
(`store.sqlite`, schema in `src/hydra_image_store/schema.sql`, `schema_version` 1) per
robot process, with the image files next to it.
