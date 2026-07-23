import pytest

from schema_based_ie.schemata.ie_schemata.aas_ie import AAS_SCHEMA_REGISTRY
from schema_based_ie.schemata.rag_schemata.rag_retrieve_schema import (
    RETRIEVE_SCHEMA_REGISTRY,
)
from schema_based_ie.utils.utils import SCHEMA_REGISTRY, get_model


def test_registry_uses_bundled_schemata() -> None:
    assert get_model("aas", "aas_factory") is AAS_SCHEMA_REGISTRY["aas_factory"]
    assert (
        get_model("retrieval", "retrieve_factory")
        is RETRIEVE_SCHEMA_REGISTRY["retrieve_factory"]
    )
    assert get_model("unknown", "aas_elastic") is AAS_SCHEMA_REGISTRY["aas_elastic"]
    assert set(SCHEMA_REGISTRY) == {"aas", "retrieval", "common"}


def test_registry_rejects_unknown_schema() -> None:
    with pytest.raises(KeyError, match="missing"):
        get_model("aas", "missing")
