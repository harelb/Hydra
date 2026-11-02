import pathlib

import pytest

from hydra_image_store import schema_sql

# Hydra's canonical schema (hydra/resources/image_store/schema.sql); absent in a
# checkout without the C++ image store
CPP_SCHEMA = (
    pathlib.Path(__file__).resolve().parents[2]
    / "hydra/resources/image_store/schema.sql"
)


@pytest.mark.skipif(
    not CPP_SCHEMA.exists(), reason="no Hydra C++ image store schema in this checkout"
)
def test_schema_matches_hydra():
    """The packaged schema is a byte-identical copy of Hydra's canonical schema."""
    assert schema_sql().encode() == CPP_SCHEMA.read_bytes()
