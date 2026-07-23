import json
import sys
import types

from schema_based_ie.langgraphs import generic_pipeline as gp
from schema_based_ie.langgraphs.icl import icl_database_creation, rdf_queries
from schema_based_ie.model_clients.client_configs import DatasheetEmbeddingCfg


def test_ranked_icl_examples_prefers_layered_metadata_and_formats_instructions():
    definitions = [{"id": "P1", "name": "Voltage", "definition": {"en": "Rated voltage"}}]
    cfg = gp.ICLCfg(
        candidates_per_property=10,
        examples_per_property=2,
        use_datasheet_embedding_similarity=False,
    )
    rows = [
        {
            "requestedPropertyId": "P1",
            "requestedPropertyName": "Voltage",
            "sourceName": "less-good.aasx",
            "propertyIdShort": "Voltage",
            "values": ["110 V"],
            "submodelIdShort": "TechnicalData",
            "elementPath": ["TechnicalData", "TechnicalProperties", "Voltage"],
            "manufacturerName": "Other",
            "eclassId": "12345678",
            "extractionInstructions": [],
        },
        {
            "requestedPropertyId": "P1",
            "requestedPropertyName": "Voltage",
            "sourceName": "best.aasx",
            "propertyIdShort": "Voltage",
            "values": ["230 V", "230 V"],
            "submodelIdShort": "TechnicalData",
            "elementPath": ["TechnicalData", "TechnicalProperties", "Electrical", "Voltage"],
            "manufacturerName": "ACME GmbH",
            "eclassId": "12345678",
            "extractionInstructions": [
                {
                    "unit": "V",
                    "extractionRule": "Prefer the rated voltage row.",
                    "formattingRule": "Return the value with unit.",
                }
            ],
        },
        {
            "requestedPropertyId": "P1",
            "requestedPropertyName": "Voltage",
            "sourceName": "same-maker-no-path.aasx",
            "propertyIdShort": "Voltage",
            "values": ["24 V"],
            "submodelIdShort": "MarketingData",
            "elementPath": ["MarketingData", "Voltage"],
            "manufacturerName": "Acme",
            "eclassId": "12345678",
            "extractionInstructions": [],
        },
    ]

    examples_by_property = gp.build_ranked_icl_examples(
        definitions,
        rows,
        cfg,
        target_manufacturer_name="Acme",
        target_eclass_id="12345678",
    )
    examples = examples_by_property[gp.definition_icl_lookup_key(definitions[0])]

    assert [example["source"] for example in examples] == ["best.aasx", "same-maker-no-path.aasx"]
    assert examples[0]["values"] == ["230 V"]
    assert examples[0]["unit"] == "V"
    assert examples[0]["instructions"][0]["extractionRule"] == "Prefer the rated voltage row."


def test_retrieve_icl_examples_queries_neo4j_and_uses_compatible_datasheet_similarity(monkeypatch):
    definitions = [{"id": "P1", "name": "Voltage", "definition": {"en": "Rated voltage"}}]
    cfg = gp.ICLCfg(
        candidates_per_property=5,
        examples_per_property=1,
    )
    captured = {}

    candidate_rows = [
        {
            "requestedPropertyId": "P1",
            "requestedPropertyName": "Voltage",
            "sourceName": "close.aasx",
            "propertyIdShort": "Voltage",
            "values": ["230 V"],
            "submodelIdShort": "TechnicalData",
            "elementPath": ["TechnicalData", "TechnicalProperties", "Voltage"],
            "manufacturerName": "Acme",
            "eclassId": "12345678",
            "extractionInstructions": [],
        },
        {
            "requestedPropertyId": "P1",
            "requestedPropertyName": "Voltage",
            "sourceName": "far.aasx",
            "propertyIdShort": "Voltage",
            "values": ["24 V"],
            "submodelIdShort": "TechnicalData",
            "elementPath": ["TechnicalData", "TechnicalProperties", "Voltage"],
            "manufacturerName": "Acme",
            "eclassId": "12345678",
            "extractionInstructions": [],
        },
    ]

    def fake_query_values(definitions_arg, **kwargs):
        captured["definitions"] = definitions_arg
        captured["value_query"] = kwargs
        return candidate_rows

    def fake_query_embeddings(**kwargs):
        captured["embedding_query"] = kwargs
        expected_hash = gp.datasheet_embedding_config_hash(cfg.embedding_cfg)
        return [
            {
                "sourceName": "close.aasx",
                "embeddingJson": json.dumps([1.0, 0.0]),
                "embeddingConfigHash": expected_hash,
            },
            {
                "sourceName": "far.aasx",
                "embeddingJson": json.dumps([0.0, 1.0]),
                "embeddingConfigHash": expected_hash,
            },
            {
                "sourceName": "incompatible.aasx",
                "embeddingJson": json.dumps([1.0, 0.0]),
                "embeddingConfigHash": "different-config",
            },
        ]

    class FakeEmbeddingClient:
        def embed(self, texts, parameters):
            captured["embed"] = {"texts": texts, "parameters": parameters}
            return {"embeddings": [[1.0, 0.0] for _ in texts], "usage": {"input_tokens": len(texts)}}

    monkeypatch.setattr(gp, "query_property_values_with_metadata", fake_query_values)
    monkeypatch.setattr(gp, "query_datasheet_embeddings", fake_query_embeddings)

    state = gp.State(
        input_path="/tmp/example",
        cfg=gp.Cfg(icl_cfg=cfg),
        preprocessed_text="current datasheet text",
        definitions=definitions,
        product_metadata={"manufacturer_name": "Acme", "eclass_id": "12345678"},
    )
    services = gp.Services(
        icl_embedding_client=FakeEmbeddingClient(),
        embedding_client=None,
        ephemeral_vector_store=None,
        query_generator=None,
        schema_based_extractor=None,
    )

    result = gp.retrieve_icl_examples(state, services)
    examples = result.icl[gp.definition_icl_lookup_key(definitions[0])]

    assert captured["value_query"]["limit"] == 5
    assert captured["value_query"]["uri"] == gp.DEFAULT_NEO4J_URI
    assert captured["value_query"]["user"] == gp.DEFAULT_NEO4J_USER
    assert captured["value_query"]["password"] == gp.DEFAULT_NEO4J_PASSWORD
    assert captured["value_query"]["target_eclass_id"] == "12345678"
    assert captured["value_query"]["target_manufacturer_name"] == "Acme"
    assert captured["embedding_query"]["embedding_config_hash"] == gp.datasheet_embedding_config_hash(cfg.embedding_cfg)
    assert set(captured["embedding_query"]["source_names"]) == {"close.aasx", "far.aasx"}
    assert captured["embed"]["parameters"] == {"model": cfg.embedding_cfg.client_cfg.model}
    assert examples[0]["source"] == "close.aasx"
    assert examples[0]["datasheet_similarity"] == 1.0
    assert result.usage["icl_embedding"] == [{"input_tokens": 1}]


def test_incompatible_datasheet_embedding_rows_are_ignored():
    compatible_hash = "same-config"
    rows = [
        {"sourceName": "a", "embeddingConfigHash": compatible_hash},
        {"sourceName": "b", "embeddingConfigHash": "other-config"},
        {"sourceName": "c"},
    ]

    assert gp.compatible_datasheet_embedding_rows(rows, compatible_hash) == [rows[0]]


def test_eclass_similarity_uses_segment_group_class_subclass_prefixes():
    assert gp.eclass_similarity("27-44-01-01", "27440101") == 1.0
    assert gp.eclass_similarity("27440199", "27440101") == 0.75
    assert gp.eclass_similarity("27449999", "27440101") == 0.5
    assert gp.eclass_similarity("27999999", "27440101") == 0.25
    assert gp.eclass_similarity("28999999", "27440101") == 0.0


def test_datasheet_embedding_defaults_and_chunking_are_shared():
    assert icl_database_creation.DatasheetEmbeddingCfg is DatasheetEmbeddingCfg
    assert gp.ICLCfg().embedding_cfg == DatasheetEmbeddingCfg()
    assert "neo4j_uri" not in gp.ICLCfg.model_fields
    assert "neo4j_user" not in gp.ICLCfg.model_fields
    assert "neo4j_password" not in gp.ICLCfg.model_fields
    assert "target_eclass_id" not in gp.ICLCfg.model_fields
    assert "target_manufacturer_name" not in gp.ICLCfg.model_fields

    text = "Voltage rating.\n\n" * 30
    cfg = DatasheetEmbeddingCfg(chunk_size=100, chunk_overlap=10)
    assert gp.chunk_datasheet_text(text, cfg.chunk_size, cfg.chunk_overlap) == icl_database_creation._chunk_text(
        text,
        cfg.chunk_size,
        cfg.chunk_overlap,
    )


def test_assemble_ie_context_adds_icl_examples_without_mutating_definitions():
    definitions = [{"id": "P1", "name": "Voltage", "definition": {"en": "Rated voltage"}}]
    original_definitions = [dict(definition) for definition in definitions]
    key = gp.definition_icl_lookup_key(definitions[0])
    state = gp.State(
        input_path="/tmp/example",
        cfg=gp.Cfg(icl_cfg=gp.ICLCfg(), schema_based_extractor_cfg=gp.SchemaBasedExtractorCfg(batch_size=1)),
        preprocessed_text="The rated voltage is 230 V.",
        definitions=definitions,
        icl={key: [{"source": "example.aasx", "values": ["230 V"], "unit": "V"}]},
        prompts={"schema_based_extraction_prompt": "Extract strictly."},
        schemata={"schema_based_extraction_schema": lambda property_definitions: object()},
        extraction_batch_index=0,
        extraction_batch_start=0,
        extraction_batch_end=1,
    )

    result = gp.assemble_ie_context(state)
    prompt = result.prompts["schema_based_extraction_prompt_instance"]

    assert "ICL Guidance Rules" in prompt
    assert "Never copy an example value" in prompt
    assert '"icl_examples"' in prompt
    assert definitions == original_definitions


def test_load_definitions_extracts_product_metadata_from_aasx(tmp_path, monkeypatch):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "sample.aasx").write_bytes(b"aasx")
    object_store = object()
    dictionary = {"properties": [{"id": "P1", "name": "Voltage"}]}

    monkeypatch.setattr(gp, "load_aasx_object_store", lambda file_bytes: object_store)
    monkeypatch.setattr(gp, "extract_technical_data_properties", lambda store: (dictionary, {}))
    monkeypatch.setattr(
        gp,
        "extract_product_metadata",
        lambda store: {"eclass_id": "27440101", "manufacturer_name": "Acme"},
    )

    state = gp.State(input_path=sample_dir, cfg=gp.Cfg())

    result = gp.load_definitions(state)

    assert result.dictionary == dictionary
    assert result.product_metadata == {"eclass_id": "27440101", "manufacturer_name": "Acme"}


def test_query_datasheet_embeddings_passes_config_hash_and_sources(monkeypatch):
    captured = {}

    class FakeRecord:
        def data(self):
            return {"sourceName": "Example.aasx"}

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def run(self, cypher, **params):
            captured["cypher"] = cypher
            captured["params"] = params
            return [FakeRecord()]

    class FakeDriver:
        def session(self):
            return FakeSession()

        def close(self):
            captured["closed"] = True

    def fake_connect_neo4j(graph_database, uri, user, password):
        captured["connection"] = (graph_database, uri, user, password)
        return FakeDriver(), None, None

    monkeypatch.setitem(sys.modules, "neo4j", types.SimpleNamespace(GraphDatabase=object()))
    monkeypatch.setattr(
        "schema_based_ie.langgraphs.icl.neo4j_connection.connect_neo4j",
        fake_connect_neo4j,
    )

    rows = rdf_queries.query_datasheet_embeddings(
        uri="bolt://example",
        user="neo4j",
        password="password",
        embedding_config_hash="abc123",
        source_names=["Example.aasx"],
    )

    assert rows == [{"sourceName": "Example.aasx"}]
    assert "ProductDatasheetEmbedding" in captured["cypher"]
    assert captured["params"] == {
        "embedding_config_hash": "abc123",
        "source_names": ["Example.aasx"],
    }
    assert captured["closed"] is True
