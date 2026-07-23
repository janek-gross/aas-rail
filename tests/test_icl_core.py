import importlib.util
import json
import sys
import types
from pathlib import Path

from pydantic import BaseModel
from basyx.aas import model
from basyx.aas.model.provider import DictObjectStore
from rdflib import Graph, URIRef

from aas_rail.langgraphs.icl.aasx_io import (
    extract_product_metadata,
    extract_technical_data_properties,
    filter_property_records_to_technical_data,
    filter_property_records_to_technical_properties,
    is_technical_data_property_record,
    is_technical_properties_property_record,
)
from aas_rail.langgraphs.icl.extraction_helper_models import (
    ExtractionHelper,
    PropertyRecord,
    PropertyRecordMember,
    RdfPropertyRecord,
    unique_non_empty,
)
from aas_rail.langgraphs.icl.extraction_helper_pipeline import (
    ExtractionHelperCfg,
    State,
    _fallback_helper_for_group,
    _helpers_from_response,
    _make_json_safe,
    has_more_batches,
    select_batch,
)
from aas_rail.langgraphs.icl.icl_database_creation import (
    _extraction_helper_payloads_from_turtle,
    _prepare_fetch_file_uri,
    _tag_imported_resources,
    _upsert_extraction_helpers,
    TurtleExport,
)
from aas_rail.langgraphs.icl.icl_database_creation import (
    DatasheetEmbeddingCfg,
    DatasheetEmbeddingClientCfg,
    build_cached_icl_turtle_export_from_sample_dir,
    convert_paired_aasx_pdf_files_to_turtle,
    create_datasheet_embedding,
    discover_icl_example_sample_dirs,
    extraction_helper_generation_hash,
)
from aas_rail.langgraphs.icl.rdf_export import safe_uri_ref
from aas_rail.langgraphs.icl.rdf_extraction_helper import (
    _add_instruction_node,
    _add_unique_literals,
    add_datasheet_embedding_to_turtle,
    add_helper_artifact_ids_to_turtle,
    add_property_record_ids_to_turtle,
)
from aas_rail.langgraphs.icl.rdf_property_grouping import group_property_records
from aas_rail.langgraphs.icl.rdf_serialization import NS_AAS
from aas_rail.langgraphs.icl import icl_database_creation, rdf_queries
from aas_rail.langgraphs.icl.rdf_queries import (
    PROPERTY_VALUES_WITH_METADATA_CYPHER,
    normalize_property_definition_params,
)
from aas_rail.model_clients.client_configs import OpenAIClientCfg


def test_unique_non_empty_removes_empty_and_duplicate_values():
    values = ["a", "", None, " a ", "a", "b", "B"]
    assert unique_non_empty(values) == ["a", "b", "B"]


def test_normalize_property_definition_params_supports_multiple_input_shapes():
    input_definitions = {
        "properties": [
            {"id": "P1", "name": "Voltage"},
        ],
    }
    normalized = normalize_property_definition_params(input_definitions)
    assert normalized == [
        {
            "property_id": "P1",
            "property_name": "Voltage",
            "property_definition": "",
        }
    ]

    normalized = normalize_property_definition_params(
        [{"property_id": "P2", "name": {"en": "Current"}}]
    )
    assert normalized == [
        {
            "property_id": "P2",
            "property_name": "Current",
            "property_definition": "",
        }
    ]


def test_property_value_query_passes_technical_properties_filter(monkeypatch):
    captured = {}

    class FakeRecord:
        def data(self):
            return {"ok": True}

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
        "aas_rail.langgraphs.icl.neo4j_connection.connect_neo4j",
        fake_connect_neo4j,
    )

    rows = rdf_queries.query_property_values_with_metadata(
        [{"id": "P1", "name": "Voltage"}],
        uri="bolt://example",
        user="neo4j",
        password="password",
        technical_data_only=True,
        technical_properties_only=True,
        helper_generation_hash="helper-hash",
        helper_artifact_id="experiment_10",
        helper_provider="llama",
        helper_model="gpt-oss:20b",
    )

    assert rows == [{"ok": True}]
    assert captured["params"]["technical_data_only"] is True
    assert captured["params"]["technical_properties_only"] is True
    assert captured["params"]["helper_generation_hash"] == "helper-hash"
    assert captured["params"]["helper_artifact_id"] == "experiment_10"
    assert captured["params"]["helper_provider"] == "llama"
    assert captured["params"]["helper_model"] == "gpt-oss:20b"
    assert "submodelInfo.path AS elementPath" in captured["cypher"]
    assert captured["closed"] is True


def test_property_value_query_cypher_filters_by_technical_properties_path():
    assert "$technical_properties_only" in PROPERTY_VALUES_WITH_METADATA_CYPHER
    assert 'identifier CONTAINS "technicalproperties"' in PROPERTY_VALUES_WITH_METADATA_CYPHER
    assert "submodelInfo.path AS elementPath" in PROPERTY_VALUES_WITH_METADATA_CYPHER
    assert "$helper_generation_hash" in PROPERTY_VALUES_WITH_METADATA_CYPHER
    assert "__helperArtifactId" in PROPERTY_VALUES_WITH_METADATA_CYPHER
    assert "toString(instruction[key]) CONTAINS $helper_artifact_id" not in PROPERTY_VALUES_WITH_METADATA_CYPHER
    assert "toStringList(instruction[key])" in PROPERTY_VALUES_WITH_METADATA_CYPHER
    assert 'key ENDS WITH "__helperArtifactId"' in PROPERTY_VALUES_WITH_METADATA_CYPHER


def test_icl_query_display_rows_can_filter_by_match_scope():
    icl_query = load_icl_query_module()
    definitions = [
        {"id": "P1", "name": "Voltage"},
        {"id": "P2", "name": "Ingress"},
        {"id": "P3", "name": "Label"},
    ]
    query_rows = [
        {
            "requestedPropertyId": "P1",
            "requestedPropertyName": "Voltage",
            "sourceName": "product",
            "values": ["230 V"],
            "propertyIdShort": "Voltage",
            "submodelIdShort": "TechnicalData",
            "elementPath": ["TechnicalData", "Electrical", "Voltage"],
        },
        {
            "requestedPropertyId": "P2",
            "requestedPropertyName": "Ingress",
            "sourceName": "product",
            "values": ["IP64"],
            "propertyIdShort": "Ingress",
            "submodelIdShort": "ProductData",
            "elementPath": ["ProductData", "TechnicalProperties", "Ingress"],
        },
        {
            "requestedPropertyId": "P3",
            "requestedPropertyName": "Label",
            "sourceName": "product",
            "values": ["Nameplate"],
            "propertyIdShort": "Label",
            "submodelIdShort": "ProductInformation",
            "elementPath": ["ProductInformation", "GeneralInformation", "Label"],
        },
    ]

    all_rows = icl_query.build_definition_result_rows(definitions, query_rows)
    technical_data_rows = icl_query.build_definition_result_rows(
        definitions,
        query_rows,
        display_scope="technical_data",
    )
    technical_properties_rows = icl_query.build_definition_result_rows(
        definitions,
        query_rows,
        display_scope="technical_properties",
    )

    assert [row["Property ID"] for row in all_rows] == ["P1", "P2", "P3"]
    assert [row["Property ID"] for row in technical_data_rows] == ["P1"]
    assert [row["Property ID"] for row in technical_properties_rows] == ["P2"]
    assert "path: ProductData / TechnicalProperties / Ingress" in technical_properties_rows[0][
        "Retrieved property information"
    ][0]


def test_extract_technical_data_properties_returns_vamos_compatible_dictionary():
    voltage = model.Property(
        "Voltage",
        value_type=str,
        value="230 V",
        description={"en": "Rated voltage"},
    )
    current_range = model.Range(
        "CurrentRange",
        value_type=float,
        min=1.5,
        max=3.5,
        description={"en": "Allowed current range"},
    )
    nested = model.SubmodelElementCollection("Electrical", value=[voltage, current_range])
    technical_properties = model.SubmodelElementCollection("TechnicalProperties", value=[nested])
    general_information = model.SubmodelElementCollection(
        "GeneralInformation",
        value=[
            model.MultiLanguageProperty(
                "ManufacturerProductDesignation",
                value={"en": "Example terminal"},
            )
        ],
    )
    technical_data = model.Submodel(
        "technical-data-submodel",
        id_short="TechnicalData",
        submodel_element=[general_information, technical_properties],
    )

    dictionary, labels = extract_technical_data_properties(DictObjectStore([technical_data]))

    assert dictionary["Type"] == "CustomDictionary"
    assert dictionary["release"] == "0.0"
    assert dictionary["classes"]["TechnicalData"]["name"] == "Example terminal"
    assert dictionary["classes"]["TechnicalData"]["properties"] == ["Voltage", "CurrentRange"]
    assert dictionary["properties"]["Voltage"] == {
        "id": "Voltage",
        "name": {"en": "Voltage"},
        "type": "string",
        "definition": {"en": "Rated voltage"},
        "unit": "",
        "values": [],
    }
    assert dictionary["properties"]["CurrentRange"]["type"] == "numeric"
    assert labels["Voltage"] == {
        "name": "Voltage",
        "unit": "",
        "type": "string",
        "definition": "Rated voltage",
        "value": "230 V",
    }
    assert labels["CurrentRange"]["value"] == {"min": 1.5, "max": 3.5}


def test_extract_product_metadata_returns_eclass_and_manufacturer():
    product_class = model.Property(
        "ProductClassId",
        value_type=str,
        value="27-44-01-01",
    )
    manufacturer = model.Property(
        "ManufacturerName",
        value_type=str,
        value="ACME GmbH",
    )
    fallback_brand = model.Property(
        "Brand",
        value_type=str,
        value="Other brand",
    )
    technical_data = model.Submodel(
        "technical-data-submodel",
        id_short="TechnicalData",
        submodel_element=[
            model.SubmodelElementCollection(
                "GeneralInformation",
                value=[fallback_brand, manufacturer, product_class],
            )
        ],
    )

    metadata = extract_product_metadata(DictObjectStore([technical_data]))

    assert metadata == {
        "eclass_id": "27-44-01-01",
        "manufacturer_name": "ACME GmbH",
    }


def test_create_datasheet_embedding_averages_chunk_embeddings(monkeypatch):
    from aas_rail.model_clients import llm_clients

    class FakeEmbeddingClient:
        def embed(self, texts, parameters):
            assert parameters == {"model": "fake-embedding-model"}
            return {
                "embeddings": [
                    [float(index), float(index + 2)]
                    for index, _ in enumerate(texts)
                ],
                "usage": {"input_tokens": 7},
            }

    monkeypatch.setitem(
        llm_clients.CACHED_EMBEDDING_CLIENT_REGISTRY,
        "llama",
        lambda: FakeEmbeddingClient(),
    )

    result = create_datasheet_embedding(
        "a" * 250,
        cfg=DatasheetEmbeddingCfg(
            client_cfg=DatasheetEmbeddingClientCfg(
                client_name="llama",
                model="fake-embedding-model",
            ),
            chunk_size=100,
            chunk_overlap=0,
            max_pdf_chars=1000,
        ),
    )

    assert result.embedding == [1.0, 3.0]
    assert result.model == "fake-embedding-model"
    assert result.provider == "llama"
    assert result.chunk_size == 100
    assert result.chunk_overlap == 0
    assert result.max_pdf_chars == 1000
    assert result.text_char_count == 250
    assert result.chunk_count == 3
    assert result.usage == {"input_tokens": 7}


def test_default_datasheet_embedding_model_uses_local_llama_model():
    cfg = DatasheetEmbeddingCfg()
    assert cfg.client_cfg.model == "ggml-org/gpt-oss-20b-GGUF:MXFP4"
    assert cfg.chunk_size == 512


def test_convert_paired_files_does_not_write_turtle_when_requested_embedding_fails(monkeypatch):
    from aas_rail.langgraphs.icl import icl_database_creation

    monkeypatch.setattr(icl_database_creation, "load_aasx_object_store", lambda file_bytes: object())
    monkeypatch.setattr(
        icl_database_creation,
        "object_store_to_turtle",
        lambda object_store: "@prefix schemaie: <https://schema-based-ie.org/icl#> .\n",
    )
    monkeypatch.setattr(icl_database_creation, "collect_property_records_from_object_store", lambda *args, **kwargs: [])
    monkeypatch.setattr(icl_database_creation, "load_pdf_text_from_bytes", lambda file_bytes: "datasheet text")

    def fail_embedding(pdf_text, cfg=None):
        raise RuntimeError("embedding model missing")

    monkeypatch.setattr(icl_database_creation, "create_datasheet_embedding", fail_embedding)

    results = convert_paired_aasx_pdf_files_to_turtle(
        [("Example.aasx", b"aasx")],
        [("Example.pdf", b"pdf")],
        generate_instructions=False,
        generate_datasheet_embeddings=True,
    )

    assert len(results) == 1
    assert results[0].status == "Error"
    assert results[0].turtle is None
    assert "Datasheet embedding generation failed: embedding model missing" in (results[0].error or "")


def test_add_datasheet_embedding_to_turtle_writes_source_embedding_node():
    turtle = add_datasheet_embedding_to_turtle(
        "@prefix schemaie: <https://schema-based-ie.org/icl#> .\n",
        source_name="Example.aasx",
        pdf_name="Example.pdf",
        embedding=[0.1, 0.2],
        embedding_model="fake-model",
        embedding_provider="llama",
        embedding_chunk_size=512,
        embedding_chunk_overlap=64,
        embedding_max_pdf_chars=1000,
        text_char_count=123,
        chunk_count=2,
    )

    graph = Graph()
    graph.parse(data=turtle, format="turtle")
    subject = URIRef("https://schema-based-ie.org/icl/datasheet-embedding/Example")
    predicate = URIRef("https://schema-based-ie.org/icl#embeddingJson")
    assert str(graph.value(subject, predicate)) == "[0.1,0.2]"
    assert str(graph.value(subject, URIRef("https://schema-based-ie.org/icl#sourceName"))) == "Example.aasx"
    assert str(graph.value(subject, URIRef("https://schema-based-ie.org/icl#embeddingDimensions"))) == "2"
    assert str(graph.value(subject, URIRef("https://schema-based-ie.org/icl#embeddingChunkSize"))) == "512"
    assert str(graph.value(subject, URIRef("https://schema-based-ie.org/icl#embeddingChunkOverlap"))) == "64"
    assert str(graph.value(subject, URIRef("https://schema-based-ie.org/icl#embeddingMaxPdfChars"))) == "1000"
    assert str(graph.value(subject, URIRef("https://schema-based-ie.org/icl#embeddingConfigJson"))) == (
        '{"chunk_overlap":64,"chunk_size":512,"max_pdf_chars":1000,'
        '"model":"fake-model","provider":"llama"}'
    )
    assert graph.value(subject, URIRef("https://schema-based-ie.org/icl#embeddingConfigHash")) is not None


def test_extraction_helper_metadata_and_artifact_ids_are_written_to_rdf():
    graph = Graph()
    instruction_uri = URIRef("https://schema-based-ie.org/icl/instruction/example/hash/Voltage")
    group = PropertyRecord(
        group_key="group",
        response_field_name="Voltage",
        records=[PropertyRecordMember(record_id="r1", name="Voltage", value="230 V")],
        id_short="Voltage",
        unit="V",
    )
    helper = ExtractionHelper(
        group_key="group",
        response_field_name="Voltage",
        property_record_id="r1",
        property_name="Voltage",
        grounding="exact",
        evidence="Technical table row for rated voltage.",
        extraction_rule="Select the value in the matching voltage row.",
        formatting_rule="Return the value with unit as written.",
    )
    metadata = {
        "helper_generation_hash": "helper-hash",
        "helper_generation_config_json": '{"model":"gpt-oss"}',
        "helper_provider": "ollama",
        "helper_model": "gpt-oss:20b",
        "helper_instruction_hash": "instruction-hash",
        "helper_temperature": 0.0,
        "helper_batch_size": 8,
        "helper_max_pdf_chars": 60000,
        "helper_technical_data_only": True,
        "helper_technical_properties_only": False,
    }

    _add_instruction_node(
        graph,
        instruction_uri,
        helper,
        group,
        source_name="Example.aasx",
        pdf_name="Example.pdf",
        helper_generation_metadata=metadata,
        helper_artifact_ids=["experiment_10", "experiment_10", "experiment_20"],
    )

    assert str(graph.value(instruction_uri, URIRef("https://schema-based-ie.org/icl#helperGenerationHash"))) == "helper-hash"
    assert str(graph.value(instruction_uri, URIRef("https://schema-based-ie.org/icl#helperProvider"))) == "ollama"
    assert str(graph.value(instruction_uri, URIRef("https://schema-based-ie.org/icl#evidence"))) == (
        "Technical table row for rated voltage."
    )
    assert str(graph.value(instruction_uri, URIRef("https://schema-based-ie.org/icl#extractionRule"))) == (
        "Select the value in the matching voltage row."
    )
    assert str(graph.value(instruction_uri, URIRef("https://schema-based-ie.org/icl#formattingRule"))) == (
        "Return the value with unit as written."
    )
    artifacts = sorted(str(value) for value in graph.objects(instruction_uri, URIRef("https://schema-based-ie.org/icl#helperArtifactId")))
    assert artifacts == ["experiment_10", "experiment_20"]


def test_add_helper_artifact_ids_to_turtle_tags_all_instruction_nodes():
    turtle = """
@prefix schemaie: <https://schema-based-ie.org/icl#> .
<https://example.org/instruction/a> a schemaie:ExtractionInstruction .
"""
    updated = add_helper_artifact_ids_to_turtle(turtle, ["experiment_5"])
    graph = Graph()
    graph.parse(data=updated, format="turtle")

    subject = URIRef("https://example.org/instruction/a")
    predicate = URIRef("https://schema-based-ie.org/icl#helperArtifactId")
    assert str(graph.value(subject, predicate)) == "experiment_5"


def test_add_property_record_ids_to_turtle_tags_base_property_nodes():
    turtle = f"""
<http://example.com/submodel/technicaldata> a <{NS_AAS}Submodel> ;
    <{NS_AAS}Submodel/submodelElements> <http://example.com/property/voltage> .
<http://example.com/property/voltage> a <{NS_AAS}Property> ;
    <{NS_AAS}Referable/idShort> "Voltage" ;
    <{NS_AAS}Property/value> "230" .
"""

    updated = add_property_record_ids_to_turtle(turtle, [make_property_record("rid-voltage")])

    graph = Graph()
    graph.parse(data=updated, format="turtle")
    subject = URIRef("http://example.com/property/voltage")
    predicate = URIRef("https://schema-based-ie.org/icl#propertyRecordId")
    assert str(graph.value(subject, predicate)) == "rid-voltage"


def test_cached_icl_source_turtle_reuses_per_sample_cache(tmp_path: Path, monkeypatch):
    sample_dir = tmp_path / "Sample"
    sample_dir.mkdir()
    (sample_dir / "Sample.aasx").write_bytes(b"aasx")
    (sample_dir / "Sample.pdf").write_bytes(b"pdf")
    fake_turtle = """
@prefix schemaie: <https://schema-based-ie.org/icl#> .
<https://example.org/instruction/a> a schemaie:ExtractionInstruction .
"""
    fake_base_turtle = """
@prefix schemaie: <https://schema-based-ie.org/icl#> .
<https://example.org/source> schemaie:propertyRecordId "rid-1" .
"""
    calls = {"count": 0}

    def fake_convert(*args, **kwargs):
        calls["count"] += 1
        instruction_cfg = kwargs["instruction_cfg"]
        debug_output_path = (
            instruction_cfg.debug_output_path
            if isinstance(instruction_cfg, ExtractionHelperCfg)
            else instruction_cfg["debug_output_path"]
        )
        Path(debug_output_path).write_text('{"graph": "invoke"}', encoding="utf-8")
        return [
            TurtleExport(
                source_name="Sample.aasx",
                ttl_name="Sample.ttl",
                status="Converted",
                pdf_name="Sample.pdf",
                turtle=fake_turtle,
                triple_count=1,
                property_record_count=2,
                instruction_count=1,
                linked_instruction_count=1,
                base_turtle=fake_base_turtle,
                debug_output_path=debug_output_path,
            )
        ]

    monkeypatch.setattr(icl_database_creation, "convert_paired_aasx_pdf_files_to_turtle", fake_convert)

    first = build_cached_icl_turtle_export_from_sample_dir(
        sample_dir,
        helper_artifact_id="experiment_5",
    )
    second = build_cached_icl_turtle_export_from_sample_dir(
        sample_dir,
        helper_artifact_id="experiment_10",
    )

    assert calls["count"] == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert Path(first.cache_path).exists()
    assert Path(first.metadata_path).exists()
    assert Path(first.base_cache_path).exists()
    graph_result_path = Path(first.debug_output_path)
    assert graph_result_path.parent == sample_dir / "icl_cache"
    assert graph_result_path.name.startswith("full.")
    assert graph_result_path.name.endswith(".graph-invoke.json")
    assert graph_result_path.read_text(encoding="utf-8") == '{"graph": "invoke"}'
    assert first.base_turtle == fake_base_turtle
    assert second.base_turtle == fake_base_turtle
    assert first.helper_generation_hash == extraction_helper_generation_hash()
    assert second.debug_output_path == str(graph_result_path)
    assert second.helper_artifact_ids == ["experiment_10"]
    assert "experiment_10" in second.turtle


def test_cached_icl_source_reruns_cached_instruction_errors(tmp_path: Path, monkeypatch):
    sample_dir = tmp_path / "Sample"
    sample_dir.mkdir()
    (sample_dir / "Sample.aasx").write_bytes(b"aasx")
    (sample_dir / "Sample.pdf").write_bytes(b"pdf")
    failed_turtle = """
@prefix schemaie: <https://schema-based-ie.org/icl#> .
<https://example.org/instruction/failed> a schemaie:ExtractionInstruction ;
    schemaie:evidence "No source evidence was generated." .
"""
    fresh_turtle = """
@prefix schemaie: <https://schema-based-ie.org/icl#> .
<https://example.org/instruction/fresh> a schemaie:ExtractionInstruction ;
    schemaie:evidence "Fresh helper evidence." .
"""
    fake_base_turtle = """
@prefix schemaie: <https://schema-based-ie.org/icl#> .
<https://example.org/source> schemaie:propertyRecordId "rid-1" .
"""
    calls = {"count": 0}

    def fake_convert(*args, **kwargs):
        calls["count"] += 1
        instruction_cfg = kwargs["instruction_cfg"]
        debug_output_path = (
            instruction_cfg.debug_output_path
            if isinstance(instruction_cfg, ExtractionHelperCfg)
            else instruction_cfg["debug_output_path"]
        )
        if calls["count"] == 1:
            Path(debug_output_path).write_text(
                json.dumps({"generation_errors": ["Batch 0 instruction generation failed: API error"]}),
                encoding="utf-8",
            )
            return [
                TurtleExport(
                    source_name="Sample.aasx",
                    ttl_name="Sample.ttl",
                    status="Converted",
                    pdf_name="Sample.pdf",
                    turtle=failed_turtle,
                    triple_count=1,
                    property_record_count=2,
                    instruction_count=1,
                    linked_instruction_count=1,
                    instruction_errors=["Batch 0 instruction generation failed: API error"],
                    base_turtle=fake_base_turtle,
                    debug_output_path=debug_output_path,
                )
            ]

        Path(debug_output_path).write_text(
            json.dumps({"generation_errors": [], "raw_generation_results": [{"text": "ok"}]}),
            encoding="utf-8",
        )
        return [
            TurtleExport(
                source_name="Sample.aasx",
                ttl_name="Sample.ttl",
                status="Converted",
                pdf_name="Sample.pdf",
                turtle=fresh_turtle,
                triple_count=1,
                property_record_count=2,
                instruction_count=1,
                linked_instruction_count=1,
                base_turtle=fake_base_turtle,
                debug_output_path=debug_output_path,
            )
        ]

    monkeypatch.setattr(icl_database_creation, "convert_paired_aasx_pdf_files_to_turtle", fake_convert)

    first = build_cached_icl_turtle_export_from_sample_dir(sample_dir)
    second = build_cached_icl_turtle_export_from_sample_dir(sample_dir, force_rebuild=False)
    third = build_cached_icl_turtle_export_from_sample_dir(sample_dir, force_rebuild=False)

    assert calls["count"] == 2
    assert first.cache_hit is False
    assert first.instruction_errors == ["Batch 0 instruction generation failed: API error"]
    assert second.cache_hit is False
    assert "Fresh helper evidence." in second.turtle
    assert "No source evidence was generated." not in second.turtle
    assert third.cache_hit is True
    assert third.instruction_errors == []
    assert "Fresh helper evidence." in third.turtle


def test_force_rebuild_removes_stale_graph_invoke_cache_before_conversion(tmp_path: Path, monkeypatch):
    sample_dir = tmp_path / "Sample"
    sample_dir.mkdir()
    aasx_path = sample_dir / "Sample.aasx"
    pdf_path = sample_dir / "Sample.pdf"
    aasx_path.write_bytes(b"aasx")
    pdf_path.write_bytes(b"pdf")

    source_cache_hash = icl_database_creation.stable_json_hash(
        {
            "aasx_sha256": icl_database_creation.file_sha256(aasx_path),
            "pdf_sha256": icl_database_creation.file_sha256(pdf_path),
            "helper_generation_hash": extraction_helper_generation_hash(),
            "embedding_config_hash": None,
            "generate_instructions": True,
            "generate_datasheet_embeddings": False,
            "technical_data_only": False,
            "technical_properties_only": False,
        }
    )
    graph_result_path = sample_dir / "icl_cache" / f"full.{source_cache_hash}.graph-invoke.json"
    graph_result_path.parent.mkdir()
    graph_result_path.write_text('{"stale": true}', encoding="utf-8")

    fake_turtle = """
@prefix schemaie: <https://schema-based-ie.org/icl#> .
<https://example.org/instruction/a> a schemaie:ExtractionInstruction .
"""
    fake_base_turtle = """
@prefix schemaie: <https://schema-based-ie.org/icl#> .
<https://example.org/source> schemaie:propertyRecordId "rid-1" .
"""

    def fake_convert(*args, **kwargs):
        instruction_cfg = kwargs["instruction_cfg"]
        debug_output_path = (
            instruction_cfg.debug_output_path
            if isinstance(instruction_cfg, ExtractionHelperCfg)
            else instruction_cfg["debug_output_path"]
        )
        assert debug_output_path == str(graph_result_path)
        assert not graph_result_path.exists()
        return [
            TurtleExport(
                source_name="Sample.aasx",
                ttl_name="Sample.ttl",
                status="Converted",
                pdf_name="Sample.pdf",
                turtle=fake_turtle,
                triple_count=1,
                base_turtle=fake_base_turtle,
            )
        ]

    monkeypatch.setattr(icl_database_creation, "convert_paired_aasx_pdf_files_to_turtle", fake_convert)

    result = build_cached_icl_turtle_export_from_sample_dir(sample_dir, force_rebuild=True)

    assert result.cache_hit is False
    assert result.debug_output_path is None
    assert not graph_result_path.exists()


def test_force_rebuild_reruns_extraction_helper_pipeline_when_full_cache_exists(tmp_path: Path, monkeypatch):
    sample_dir = tmp_path / "Sample"
    sample_dir.mkdir()
    aasx_path = sample_dir / "Sample.aasx"
    pdf_path = sample_dir / "Sample.pdf"
    aasx_path.write_bytes(b"aasx")
    pdf_path.write_bytes(b"pdf")

    property_record = RdfPropertyRecord(
        record_id="rid-voltage",
        name="Voltage",
        value="230 V",
        aas_type="Property",
        id_short="Voltage",
    )
    helper = ExtractionHelper(
        group_key="group",
        response_field_name="Voltage",
        property_record_id="rid-voltage",
        property_name="Voltage",
        grounding="exact",
        evidence="Fresh helper evidence.",
        extraction_rule="Select the fresh value.",
        formatting_rule="Return the fresh value.",
    )
    base_turtle = """
@prefix schemaie: <https://schema-based-ie.org/icl#> .
<https://example.org/source> schemaie:propertyRecordId "rid-voltage" .
"""
    helper_turtle = """
@prefix schemaie: <https://schema-based-ie.org/icl#> .
<https://example.org/source> schemaie:propertyRecordId "rid-voltage" .
<https://example.org/instruction/fresh> a schemaie:ExtractionInstruction ;
    schemaie:evidence "Fresh helper evidence." .
"""
    calls = {"pipeline": 0}

    monkeypatch.setattr(icl_database_creation, "load_aasx_object_store", lambda file_bytes: object())
    monkeypatch.setattr(icl_database_creation, "object_store_to_turtle", lambda object_store: base_turtle)
    monkeypatch.setattr(
        icl_database_creation,
        "collect_property_records_from_object_store",
        lambda object_store, source_name=None: [property_record],
    )
    monkeypatch.setattr(icl_database_creation, "add_property_record_ids_to_turtle", lambda turtle, records: turtle)
    monkeypatch.setattr(icl_database_creation, "load_pdf_text_from_bytes", lambda file_bytes: "PDF text")

    def fake_pipeline(pdf_text, property_groups, cfg=None, instruction=None):
        calls["pipeline"] += 1
        return icl_database_creation.ExtractionHelperRunResult(
            instructions=[helper],
            debug_output_path=cfg.get("debug_output_path") if isinstance(cfg, dict) else None,
        )

    def fake_add_helpers(**kwargs):
        assert kwargs["instructions"][0].evidence == "Fresh helper evidence."
        return helper_turtle, 1

    monkeypatch.setattr(icl_database_creation, "run_extraction_helper_pipeline", fake_pipeline)
    monkeypatch.setattr(icl_database_creation, "add_extraction_instructions_to_turtle", fake_add_helpers)

    first = build_cached_icl_turtle_export_from_sample_dir(sample_dir)
    assert calls["pipeline"] == 1
    assert "Fresh helper evidence." in first.turtle

    stale_ttl_path = Path(first.cache_path)
    stale_metadata_path = Path(first.metadata_path)
    stale_ttl_path.write_text(
        """
@prefix schemaie: <https://schema-based-ie.org/icl#> .
<https://example.org/instruction/stale> a schemaie:ExtractionInstruction ;
    schemaie:evidence "Stale cached helper." .
""",
        encoding="utf-8",
    )
    assert stale_ttl_path.exists()
    assert stale_metadata_path.exists()

    second = build_cached_icl_turtle_export_from_sample_dir(sample_dir, force_rebuild=True)

    assert calls["pipeline"] == 2
    assert second.cache_hit is False
    assert "Fresh helper evidence." in second.turtle
    assert "Stale cached helper." not in second.turtle


def test_extraction_helper_payloads_collect_record_and_artifact_ids():
    turtle = """
@prefix schemaie: <https://schema-based-ie.org/icl#> .
<https://example.org/instruction/a> a schemaie:ExtractionInstruction ;
    schemaie:propertyRecordId "rid-1", "rid-2" ;
    schemaie:helperArtifactId "experiment_5", "experiment_10" ;
    schemaie:extractionRule "Choose the exact voltage row" .
"""

    payloads = _extraction_helper_payloads_from_turtle(turtle)

    assert len(payloads) == 1
    assert sorted(payloads[0]["record_ids"]) == ["rid-1", "rid-2"]
    assert sorted(payloads[0]["artifact_ids"]) == ["experiment_10", "experiment_5"]
    assert payloads[0]["properties"]["schemaie__extractionRule"] == "Choose the exact voltage row"
    assert "schemaie__helperArtifactId" not in payloads[0]["properties"]


def test_upsert_extraction_helpers_links_by_source_tagged_property_record_id():
    turtle = """
@prefix schemaie: <https://schema-based-ie.org/icl#> .
<https://example.org/instruction/a> a schemaie:ExtractionInstruction ;
    schemaie:propertyRecordId "rid-1" ;
    schemaie:helperArtifactId "experiment_5" .
"""

    class FakeResult:
        def single(self):
            return {"linked": 2}

    class FakeSession:
        def __init__(self):
            self.calls = []

        def run(self, query, **params):
            self.calls.append((query, params))
            return FakeResult()

    session = FakeSession()
    result = _upsert_extraction_helpers(session, turtle, "Sample.aasx")

    assert result == {"helpers": 1, "links": 2}
    query, params = session.calls[0]
    assert "$source_name IN coalesce(element.iclSourceNames, [])" in query
    assert 'key ENDS WITH "__propertyRecordId"' in query
    assert "selfRel:schemaie__hasExtractionInstruction" in query
    assert 'label ENDS WITH "__Property"' in query
    assert 'label ENDS WITH "__MultiLanguageProperty"' in query
    assert 'label ENDS WITH "__Range"' in query
    assert "replace(replace(toString(instruction.`schemaie__helperArtifactId`)" not in query
    assert "toString(element[key]) CONTAINS recordId" not in query
    assert "toStringList(instruction.`schemaie__helperArtifactId`)" in query
    assert "toStringList(element[key])" in query
    assert params["record_ids"] == ["rid-1"]
    assert params["artifact_ids"] == ["experiment_5"]


def test_tag_imported_resources_does_not_require_resource_label():
    turtle = """
@prefix ex: <http://example.com/> .
ex:root ex:hasChild [ ex:value "blank" ] .
"""

    class FakeResult:
        def single(self):
            return {"tagged": 1}

    class FakeSession:
        def __init__(self):
            self.queries = []

        def run(self, query, **params):
            self.queries.append(query)
            return FakeResult()

    session = FakeSession()
    tagged = _tag_imported_resources(session, turtle, graph_name="Sample", source_name="Sample.aasx")

    assert tagged == 2
    assert "MATCH (resource {uri: uri})" in session.queries[0]
    assert "MATCH (resource:Resource" not in session.queries[0]
    assert "MATCH (root)" in session.queries[1]
    assert "MATCH (root:Resource" not in session.queries[1]


def test_cached_neo4j_import_checks_database_for_base_even_on_cache_hit(monkeypatch):
    monkeypatch.setitem(sys.modules, "neo4j", types.SimpleNamespace(GraphDatabase=object()))
    events = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeDriver:
        def session(self):
            return FakeSession()

        def close(self):
            events.append("close")

    monkeypatch.setattr(
        icl_database_creation,
        "connect_neo4j",
        lambda graph_database, uri, user, password: (FakeDriver(), "bolt://fake", "neo4j"),
    )
    monkeypatch.setattr(icl_database_creation, "_ensure_n10s_ready", lambda session: events.append("ready"))
    monkeypatch.setattr(
        icl_database_creation,
        "_source_base_graph_exists",
        lambda session, source_name: events.append(("checked", source_name)) or False,
    )
    monkeypatch.setattr(
        icl_database_creation,
        "_import_with_n10s",
        lambda session, turtle, turtle_file_uri: events.append(("imported", turtle)) or (3, "n10s.inline"),
    )
    monkeypatch.setattr(
        icl_database_creation,
        "_tag_imported_resources",
        lambda session, turtle, graph_name, source_name: events.append(("tagged", turtle)) or 1,
    )
    monkeypatch.setattr(
        icl_database_creation,
        "_upsert_extraction_helpers",
        lambda session, turtle, source_name: events.append(("upserted", turtle)) or {"helpers": 1, "links": 1},
    )

    result = icl_database_creation.import_cached_turtle_export_to_neo4j(
        TurtleExport(
            source_name="Sample.aasx",
            ttl_name="Sample.ttl",
            status="Cached",
            turtle="@prefix schemaie: <https://schema-based-ie.org/icl#> .\n",
            triple_count=1,
            cache_hit=True,
            base_turtle="@prefix schemaie: <https://schema-based-ie.org/icl#> .\n",
        )
    )

    assert result.success is True
    assert ("checked", "Sample.aasx") in events
    assert ("imported", "@prefix schemaie: <https://schema-based-ie.org/icl#> .\n") in events
    assert ("tagged", "@prefix schemaie: <https://schema-based-ie.org/icl#> .\n") in events
    assert "Imported base graph" in result.message


def test_import_turtle_exports_uses_cached_import_when_base_turtle_is_available(monkeypatch):
    calls = []

    def fake_cached_import(export, **kwargs):
        calls.append(export.source_name)
        return icl_database_creation.Neo4jImportResult(
            source_name=export.source_name,
            success=True,
            message="cached import",
        )

    monkeypatch.setattr(icl_database_creation, "import_cached_turtle_export_to_neo4j", fake_cached_import)
    monkeypatch.setattr(
        icl_database_creation,
        "import_turtle_to_neo4j",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full import should not run")),
    )

    results = icl_database_creation.import_turtle_exports_to_neo4j(
        [
            TurtleExport(
                source_name="Sample.aasx",
                ttl_name="Sample.ttl",
                status="Cached",
                turtle="@prefix schemaie: <https://schema-based-ie.org/icl#> .\n",
                triple_count=1,
                base_turtle="@prefix schemaie: <https://schema-based-ie.org/icl#> .\n",
            )
        ],
        remove_resource_label=False,
    )

    assert calls == ["Sample.aasx"]
    assert results[0].message == "cached import"


def test_discover_icl_example_sample_dirs_balances_limited_samples_by_company(tmp_path: Path):
    for sample_name in [
        "Festo_1",
        "Festo_2",
        "Festo_3",
        "Harting_1",
        "Rstahl_1",
        "Rstahl_2",
    ]:
        sample_dir = tmp_path / sample_name
        sample_dir.mkdir()
        (sample_dir / f"{sample_name}.aasx").write_bytes(b"aasx")
        (sample_dir / f"{sample_name}.pdf").write_bytes(b"pdf")

    selected = discover_icl_example_sample_dirs(tmp_path, sample_limit=5)

    assert [path.name for path in selected] == [
        "Festo_1",
        "Harting_1",
        "Rstahl_1",
        "Festo_2",
        "Rstahl_2",
    ]


def load_icl_query_module():
    app_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "streamlit-web-ui"
        / "tabs"
        / "icl_query.py"
    )
    spec = importlib.util.spec_from_file_location("icl_query_test_module", app_path)
    module = importlib.util.module_from_spec(spec)
    previous_streamlit = sys.modules.get("streamlit")
    previous_utils = sys.modules.get("utils")
    sys.modules["streamlit"] = types.SimpleNamespace()
    sys.modules["utils"] = types.SimpleNamespace(
        count_property_definitions=lambda definitions: len(definitions or []),
        make_json_safe=lambda value: value,
    )
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if previous_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = previous_streamlit
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils
    return module


def make_property_record(record_id: str, semantic_id: str | None = None) -> RdfPropertyRecord:
    return RdfPropertyRecord(
        record_id=record_id,
        name="Voltage",
        value="230",
        definition="Test property",
        aas_type="Property",
        id_short="Voltage",
        unit="V",
        datatype="string",
        path=["TechnicalData", "Electrical"],
        element_path=["Electrical", "Voltage"],
        submodel_id="http://example.com/submodel/technicaldata",
        submodel_id_short="TechnicalData",
        semantic_id=semantic_id,
        source_name="test-source",
    )


def test_group_property_records_collapses_same_semantic_id_records():
    records = [
        make_property_record("r1", semantic_id="id1"),
        make_property_record("r2", semantic_id="id1"),
    ]
    groups = group_property_records(records)
    assert len(groups) == 1
    assert groups[0].semantic_id == "id1"
    assert len(groups[0].records) == 2
    assert groups[0].response_field_name != ""


def test_is_technical_data_property_record_filters_by_submodel_or_path():
    technical_record = make_property_record("r1")
    assert is_technical_data_property_record(technical_record)

    nontechnical_record = make_property_record("r2")
    nontechnical_record.submodel_id_short = "ProductInformation"
    nontechnical_record.submodel_id = "http://example.com/submodel/productinformation"
    nontechnical_record.path = ["ProductInformation", "Name"]
    assert not is_technical_data_property_record(nontechnical_record)


def test_filter_property_records_to_technical_data_returns_only_matching_records():
    technical = make_property_record("r1")
    nontechnical = make_property_record("r2")
    nontechnical.submodel_id_short = "ProductInformation"
    nontechnical.submodel_id = "http://example.com/submodel/productinformation"
    nontechnical.path = ["ProductInformation", "Name"]

    filtered = filter_property_records_to_technical_data([technical, nontechnical])
    assert filtered == [technical]


def test_is_technical_properties_property_record_filters_by_path():
    technical_properties_record = make_property_record("r3")
    technical_properties_record.path = ["TechnicalData", "TechnicalProperties", "R_STAHL_Classification", "ZCL00009", "ZMN02967"]
    assert is_technical_properties_property_record(technical_properties_record)

    nontechnical_record = make_property_record("r4")
    nontechnical_record.path = ["TechnicalData", "Electrical", "Voltage"]
    assert not is_technical_properties_property_record(nontechnical_record)


def test_filter_property_records_to_technical_properties_returns_only_matching_records():
    technical_properties = make_property_record("r5")
    technical_properties.path = ["TechnicalData", "TechnicalProperties", "R_STAHL_Classification"]
    nontechnical = make_property_record("r6")
    nontechnical.path = ["TechnicalData", "Electrical", "Voltage"]

    filtered = filter_property_records_to_technical_properties([technical_properties, nontechnical])
    assert filtered == [technical_properties]


def test_add_unique_literals_writes_only_non_empty_distinct_literals():
    graph = Graph()
    subject = URIRef("http://example.org/instruction")
    predicate = URIRef("http://example.org/counterExample")
    _add_unique_literals(graph, subject, predicate, ["one", "", None, "one", "two"])

    objects = sorted(str(value) for value in graph.objects(subject, predicate))
    assert objects == ["one", "two"]


def test_prepare_fetch_file_uri_writes_turtle_and_returns_file_uri(tmp_path: Path):
    export = TurtleExport(
        source_name="test.aasx",
        ttl_name="test.ttl",
        status="Converted",
        turtle="@prefix : <http://example.org/> .\n",
        triple_count=0,
    )
    uri = _prepare_fetch_file_uri(export, tmp_path, "file:///var/lib/neo4j/import")

    assert uri == "file:///var/lib/neo4j/import/test.ttl"
    assert (tmp_path / "test.ttl").read_text(encoding="utf-8") == export.turtle


def test_safe_uri_ref_encodes_extra_hash_characters_in_fragments():
    uri = safe_uri_ref("file:///home/aas-rail/aas_rail/0173-1#02-AAQ325#003")
    assert str(uri) == "file:///home/aas-rail/aas_rail/0173-1#02-AAQ325%23003"

    uri2 = safe_uri_ref("https://example.com/path#frag#other")
    assert str(uri2) == "https://example.com/path#frag%23other"


def test_select_batch_and_has_more_batches_behave_as_expected():
    groups = [
        PropertyRecord(
            group_key="g1",
            response_field_name="field1",
            records=[PropertyRecordMember(record_id="r1", name="A")],
            id_short="A",
        ),
        PropertyRecord(
            group_key="g2",
            response_field_name="field2",
            records=[PropertyRecordMember(record_id="r2", name="B")],
            id_short="B",
        ),
        PropertyRecord(
            group_key="g3",
            response_field_name="field3",
            records=[PropertyRecordMember(record_id="r3", name="C")],
            id_short="C",
        ),
    ]
    state = State(cfg=ExtractionHelperCfg(batch_size=2), pdf_text="", property_groups=groups)

    select_batch(state)
    assert state.batch_start == 0
    assert state.batch_end == 2
    assert has_more_batches(state) == "has_more"

    state.batch_index = 1
    select_batch(state)
    assert state.batch_start == 2
    assert state.batch_end == 3
    assert has_more_batches(state) == "done"


def test_extraction_helper_cfg_accepts_openai_client_cfg():
    cfg = ExtractionHelperCfg(client_cfg=OpenAIClientCfg(model="gpt-5.4"))

    assert cfg.client_cfg.client_name == "openai"
    assert cfg.client_cfg.model == "gpt-5.4"


def test_helpers_from_response_uses_current_response_model_fields():
    group = PropertyRecord(
        group_key="group",
        response_field_name="Voltage",
        records=[PropertyRecordMember(record_id="r1", name="Voltage", value="230 V")],
        id_short="Voltage",
        semantic_id="semantic-voltage",
    )
    helpers = _helpers_from_response(
        {
            "Voltage": {
                "grounding": "exact",
                "evidence": "Technical table row for rated voltage.",
                "extraction_rule": "Select the value in the matching row.",
                "formatting_rule": "Return the value with unit as written.",
                "avoid": ["Ignore test voltage rows."],
            }
        },
        [group],
    )

    assert len(helpers) == 1
    helper = helpers[0]
    assert helper.group_key == "group"
    assert helper.semantic_id == "semantic-voltage"
    assert helper.evidence == "Technical table row for rated voltage."
    assert helper.extraction_rule == "Select the value in the matching row."
    assert helper.formatting_rule == "Return the value with unit as written."
    assert helper.avoid == ["Ignore test voltage rows."]


def test_fallback_helper_uses_current_response_model_fields():
    group = PropertyRecord(
        group_key="group",
        response_field_name="Voltage",
        records=[PropertyRecordMember(record_id="r1", name="Voltage", value="230 V")],
        id_short="Voltage",
    )

    helper = _fallback_helper_for_group(group)

    assert helper.grounding == "reference_only"
    assert helper.evidence
    assert helper.extraction_rule
    assert helper.formatting_rule
    assert helper.avoid == []


def test_extraction_helper_pipeline_writes_graph_invoke_result(tmp_path: Path, monkeypatch):
    from aas_rail.langgraphs.icl import extraction_helper_pipeline

    group = PropertyRecord(
        group_key="group",
        response_field_name="Voltage",
        records=[PropertyRecordMember(record_id="r1", name="Voltage", value="230 V")],
        id_short="Voltage",
    )

    class DummyGenerator:
        def __init__(self, cfg):
            self.cfg = cfg

    class DummyGraph:
        def invoke(self, graph_input):
            return {
                **graph_input,
                "batch_index": 1,
                "batch_start": 0,
                "batch_end": 1,
                "prompt_instance": "prompt",
                "extraction_helpers": [],
                "usage": [{"total_tokens": 3}],
                "raw_generation_results": [{"text": "raw"}],
                "generation_errors": [],
            }

    monkeypatch.setattr(extraction_helper_pipeline, "ExtractionHelperGenerator", DummyGenerator)
    monkeypatch.setattr(extraction_helper_pipeline, "build_graph", lambda generator: DummyGraph())

    output_path = tmp_path / "graph-invoke.json"
    result = extraction_helper_pipeline.run_extraction_helper_pipeline(
        pdf_text="PDF text",
        property_groups=[group],
        cfg={"debug_output_path": str(output_path)},
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert "raw_langgraph_state" not in saved
    assert saved["pdf_text"] == "PDF text"
    assert saved["batch_index"] == 1
    assert saved["property_groups"][0]["group_key"] == "group"
    assert result.debug_output_path == str(output_path)


def test_make_json_safe_converts_models_and_paths():
    class SampleModel(BaseModel):
        name: str
        count: int

    sample = SampleModel(name="test", count=1)
    payload = {
        "path": Path("/tmp/test"),
        "model": sample,
        "items": [1, None, {"inner": Path("/tmp/inner")}],
    }

    safe = _make_json_safe(payload)
    assert safe["path"] == str(Path("/tmp/test"))
    assert safe["model"] == {"name": "test", "count": 1}
    assert safe["items"][2]["inner"] == str(Path("/tmp/inner"))
