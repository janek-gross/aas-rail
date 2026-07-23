"""RDF persistence helpers for PDF-derived extraction instructions."""

import hashlib
import json
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from rdflib import Graph, Literal as RDFLiteral, Namespace, RDF, URIRef
from rdflib.namespace import XSD

from .extraction_helper_models import ExtractionHelper, PropertyRecord, RdfPropertyRecord
from .rdf_property_grouping import group_property_records
from .rdf_export import safe_uri_ref
from .rdf_serialization import NS_AAS


AAS = Namespace(NS_AAS)
SCHEMA_IE = Namespace("https://schema-based-ie.org/icl#")
SCHEMA_IE_INSTRUCTION_ROOT = "https://schema-based-ie.org/icl/instruction"
SCHEMA_IE_DATASHEET_EMBEDDING_ROOT = "https://schema-based-ie.org/icl/datasheet-embedding"


def add_datasheet_embedding_to_turtle(
    turtle_data: str,
    *,
    source_name: str,
    pdf_name: str | None,
    embedding: Sequence[float],
    embedding_model: str,
    embedding_provider: str,
    text_char_count: int,
    chunk_count: int,
    embedding_chunk_size: int | None = None,
    embedding_chunk_overlap: int | None = None,
    embedding_max_pdf_chars: int | None = None,
) -> str:
    """Add a source-level product datasheet embedding node to Turtle RDF."""
    graph = Graph()
    graph.parse(data=turtle_data, format="turtle")
    graph.bind("schemaie", SCHEMA_IE)

    embedding_uri = _datasheet_embedding_uri(source_name)
    graph.add((embedding_uri, RDF.type, SCHEMA_IE.ProductDatasheetEmbedding))
    _add_literal(graph, embedding_uri, SCHEMA_IE.sourceName, source_name)
    _add_literal(graph, embedding_uri, SCHEMA_IE.pdfName, pdf_name)
    _add_literal(graph, embedding_uri, SCHEMA_IE.embeddingProvider, embedding_provider)
    _add_literal(graph, embedding_uri, SCHEMA_IE.embeddingModel, embedding_model)
    _add_literal(graph, embedding_uri, SCHEMA_IE.embeddingChunkSize, embedding_chunk_size, XSD.integer)
    _add_literal(graph, embedding_uri, SCHEMA_IE.embeddingChunkOverlap, embedding_chunk_overlap, XSD.integer)
    _add_literal(graph, embedding_uri, SCHEMA_IE.embeddingMaxPdfChars, embedding_max_pdf_chars, XSD.integer)
    _add_literal(graph, embedding_uri, SCHEMA_IE.embeddingDimensions, len(embedding), XSD.integer)
    _add_literal(graph, embedding_uri, SCHEMA_IE.embeddingChunkCount, chunk_count, XSD.integer)
    _add_literal(graph, embedding_uri, SCHEMA_IE.textCharCount, text_char_count, XSD.integer)
    embedding_config = _embedding_config_metadata(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_chunk_size=embedding_chunk_size,
        embedding_chunk_overlap=embedding_chunk_overlap,
        embedding_max_pdf_chars=embedding_max_pdf_chars,
    )
    embedding_config_json = json.dumps(embedding_config, sort_keys=True, separators=(",", ":"))
    _add_literal(graph, embedding_uri, SCHEMA_IE.embeddingConfigJson, embedding_config_json)
    _add_literal(graph, embedding_uri, SCHEMA_IE.embeddingConfigHash, _stable_config_hash(embedding_config))
    _add_literal(graph, embedding_uri, SCHEMA_IE.embeddingJson, json.dumps(list(embedding), separators=(",", ":")))
    return graph.serialize(format="turtle")


def add_extraction_instructions_to_turtle(
    turtle_data: str,
    property_records: Sequence[RdfPropertyRecord | Mapping[str, Any]],
    instructions: Sequence[ExtractionHelper | Mapping[str, Any]],
    source_name: str,
    pdf_name: str | None = None,
    helper_generation_metadata: Mapping[str, Any] | None = None,
    helper_artifact_ids: Sequence[str] | None = None,
) -> tuple[str, int]:
    """Add extraction-instruction nodes and property links to Turtle RDF."""
    graph = Graph()
    graph.parse(data=turtle_data, format="turtle")
    graph.bind("schemaie", SCHEMA_IE)

    records = [
        record if isinstance(record, RdfPropertyRecord) else RdfPropertyRecord.model_validate(record)
        for record in property_records
    ]
    instructions = [
        instruction
        if isinstance(instruction, ExtractionHelper)
        else ExtractionHelper.model_validate(instruction)
        for instruction in instructions
    ]
    record_groups = group_property_records(records)
    records_by_id = {record.record_id: record for record in records}

    instructions_by_group_key = {
        instruction.group_key: instruction
        for instruction in instructions
        if instruction.group_key
    }
    instructions_by_field_name = {
        instruction.response_field_name: instruction
        for instruction in instructions
        if instruction.response_field_name
    }
    instructions_by_semantic_id = {
        instruction.semantic_id: instruction
        for instruction in instructions
        if instruction.semantic_id
    }
    instructions_by_record_id = {
        instruction.property_record_id: instruction
        for instruction in instructions
        if instruction.property_record_id
    }

    linked = 0
    for group in record_groups:
        group_records = [
            records_by_id[member.record_id]
            for member in group.records
            if member.record_id in records_by_id
        ]
        instruction = (
            instructions_by_group_key.get(group.group_key)
            or instructions_by_field_name.get(group.response_field_name)
            or (instructions_by_semantic_id.get(group.semantic_id) if group.semantic_id else None)
        )
        if instruction is None:
            instruction = next(
                (
                    instructions_by_record_id[record.record_id]
                    for record in group_records
                    if record.record_id in instructions_by_record_id
                ),
                None,
            )
        if instruction is None:
            continue

        instruction_uri = _instruction_uri(
            source_name,
            group.response_field_name,
            str((helper_generation_metadata or {}).get("helper_generation_hash") or ""),
        )
        for record in group_records:
            property_subject = find_property_subject(graph, record)
            if property_subject is not None:
                graph.add((property_subject, SCHEMA_IE.hasExtractionInstruction, instruction_uri))
                linked += 1

        _add_instruction_node(
            graph,
            instruction_uri,
            instruction,
            group,
            source_name=source_name,
            pdf_name=pdf_name,
            helper_generation_metadata=helper_generation_metadata,
            helper_artifact_ids=helper_artifact_ids,
        )

    return graph.serialize(format="turtle"), linked


def add_property_record_ids_to_turtle(
    turtle_data: str,
    property_records: Sequence[RdfPropertyRecord | Mapping[str, Any]],
) -> str:
    """Attach stable property record IDs to value-bearing AAS property nodes."""
    graph = Graph()
    graph.parse(data=turtle_data, format="turtle")
    graph.bind("schemaie", SCHEMA_IE)
    records = [
        record if isinstance(record, RdfPropertyRecord) else RdfPropertyRecord.model_validate(record)
        for record in property_records
    ]
    for record in records:
        property_subject = find_property_subject(graph, record)
        if property_subject is None:
            continue
        _add_unique_literals(graph, property_subject, SCHEMA_IE.propertyRecordId, [record.record_id])
    return graph.serialize(format="turtle")


def add_helper_artifact_ids_to_turtle(
    turtle_data: str,
    helper_artifact_ids: Sequence[str] | None,
) -> str:
    """Attach artifact IDs to all extraction-helper nodes in Turtle RDF."""
    clean_artifact_ids = _clean_sequence(helper_artifact_ids)
    if not clean_artifact_ids:
        return turtle_data

    graph = Graph()
    graph.parse(data=turtle_data, format="turtle")
    graph.bind("schemaie", SCHEMA_IE)
    for instruction_uri in graph.subjects(RDF.type, SCHEMA_IE.ExtractionInstruction):
        _add_unique_literals(graph, instruction_uri, SCHEMA_IE.helperArtifactId, clean_artifact_ids)
    return graph.serialize(format="turtle")


def find_property_subject(graph: Graph, record: RdfPropertyRecord):
    """Find the RDF node for a property record in an AAS RDF graph."""
    if record.submodel_id and record.element_path:
        current = safe_uri_ref(record.submodel_id)
        for segment in record.element_path:
            current = _find_child_by_id_short(graph, current, segment)
            if current is None:
                break
        if current is not None:
            return current

    return _find_property_subject_by_id_short(graph, record)


def _find_child_by_id_short(graph: Graph, parent, id_short: str):
    for predicate in (
        AAS["Submodel/submodelElements"],
        AAS["SubmodelElementCollection/value"],
        AAS["SubmodelElementList/value"],
    ):
        for child in graph.objects(parent, predicate):
            child_id_short = graph.value(child, AAS["Referable/idShort"])
            if child_id_short is not None and str(child_id_short) == id_short:
                return child
    return None


def _find_property_subject_by_id_short(graph: Graph, record: RdfPropertyRecord):
    candidates = []
    for rdf_type in (AAS["Property"], AAS["MultiLanguageProperty"], AAS["Range"]):
        for subject in graph.subjects(RDF.type, rdf_type):
            id_short = graph.value(subject, AAS["Referable/idShort"])
            if id_short is not None and str(id_short) == record.name:
                candidates.append(subject)

    if len(candidates) == 1:
        return candidates[0]

    value_matched = [
        subject
        for subject in candidates
        if record.value and _rdf_value_text(graph, subject).strip() == record.value.strip()
    ]
    if value_matched:
        return value_matched[0]
    return candidates[0] if candidates else None


def _rdf_value_text(graph: Graph, subject) -> str:
    values = []
    for predicate in (AAS["Property/value"], AAS["Range/min"], AAS["Range/max"]):
        values.extend(str(value) for value in graph.objects(subject, predicate))
    return ", ".join(values)


def _instruction_uri(source_name: str, instruction_key: str, helper_generation_hash: str = "") -> URIRef:
    stem = quote(source_name.rsplit(".", 1)[0], safe="")
    key = quote(instruction_key, safe="")
    if helper_generation_hash:
        generation = quote(helper_generation_hash, safe="")
        return URIRef(f"{SCHEMA_IE_INSTRUCTION_ROOT}/{stem}/{generation}/{key}")
    return URIRef(f"{SCHEMA_IE_INSTRUCTION_ROOT}/{stem}/{key}")


def _datasheet_embedding_uri(source_name: str) -> URIRef:
    stem = quote(source_name.rsplit(".", 1)[0], safe="")
    return URIRef(f"{SCHEMA_IE_DATASHEET_EMBEDDING_ROOT}/{stem}")


def _embedding_config_metadata(
    *,
    embedding_provider: str,
    embedding_model: str,
    embedding_chunk_size: int | None,
    embedding_chunk_overlap: int | None,
    embedding_max_pdf_chars: int | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": embedding_provider,
        "model": embedding_model,
    }
    if embedding_chunk_size is not None:
        metadata["chunk_size"] = int(embedding_chunk_size)
    if embedding_chunk_overlap is not None:
        metadata["chunk_overlap"] = int(embedding_chunk_overlap)
    if embedding_max_pdf_chars is not None:
        metadata["max_pdf_chars"] = int(embedding_max_pdf_chars)
    return metadata


def _stable_config_hash(metadata: Mapping[str, Any]) -> str:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _add_instruction_node(
    graph: Graph,
    instruction_uri: URIRef,
    instruction: ExtractionHelper,
    group: PropertyRecord,
    *,
    source_name: str,
    pdf_name: str | None,
    helper_generation_metadata: Mapping[str, Any] | None = None,
    helper_artifact_ids: Sequence[str] | None = None,
) -> None:
    records = group.records
    representative = records[0]
    graph.add((instruction_uri, RDF.type, SCHEMA_IE.ExtractionInstruction))
    _add_literal(graph, instruction_uri, SCHEMA_IE.groupKey, instruction.group_key or group.group_key)
    _add_literal(
        graph,
        instruction_uri,
        SCHEMA_IE.responseFieldName,
        instruction.response_field_name or group.response_field_name,
    )
    _add_literal(graph, instruction_uri, SCHEMA_IE.semanticId, instruction.semantic_id or group.semantic_id)
    _add_literal(graph, instruction_uri, SCHEMA_IE.idShort, group.id_short)
    _add_literal(graph, instruction_uri, SCHEMA_IE.unit, group.unit)
    _add_literal(graph, instruction_uri, SCHEMA_IE.datatype, group.datatype)
    _add_literal(graph, instruction_uri, SCHEMA_IE.propertyRecordCount, len(records), XSD.integer)
    _add_unique_literals(
        graph,
        instruction_uri,
        SCHEMA_IE.propertyRecordId,
        [record.record_id for record in records],
    )
    _add_unique_literals(
        graph,
        instruction_uri,
        SCHEMA_IE.propertyName,
        [instruction.property_name] + [record.name for record in records],
    )
    _add_unique_literals(
        graph,
        instruction_uri,
        SCHEMA_IE.knownPropertyValue,
        [record.value for record in records],
    )
    _add_unique_literals(
        graph,
        instruction_uri,
        SCHEMA_IE.propertyDefinition,
        [record.definition for record in records],
    )
    _add_unique_literals(
        graph,
        instruction_uri,
        SCHEMA_IE.propertyPath,
        [" / ".join(record.path) for record in records],
    )
    _add_literal(graph, instruction_uri, SCHEMA_IE.grounding, instruction.grounding)
    _add_literal(graph, instruction_uri, SCHEMA_IE.evidence, instruction.evidence)
    _add_literal(graph, instruction_uri, SCHEMA_IE.extractionRule, instruction.extraction_rule)
    _add_literal(graph, instruction_uri, SCHEMA_IE.formattingRule, instruction.formatting_rule)
    _add_unique_literals(graph, instruction_uri, SCHEMA_IE.avoid, instruction.avoid)
    _add_literal(graph, instruction_uri, SCHEMA_IE.sourceName, source_name)
    _add_literal(graph, instruction_uri, SCHEMA_IE.pdfName, pdf_name)
    _add_helper_generation_metadata(graph, instruction_uri, helper_generation_metadata)
    _add_unique_literals(graph, instruction_uri, SCHEMA_IE.helperArtifactId, _clean_sequence(helper_artifact_ids))


def _add_helper_generation_metadata(
    graph: Graph,
    instruction_uri: URIRef,
    metadata: Mapping[str, Any] | None,
) -> None:
    if not metadata:
        return

    _add_literal(graph, instruction_uri, SCHEMA_IE.helperGenerationHash, metadata.get("helper_generation_hash"))
    _add_literal(
        graph,
        instruction_uri,
        SCHEMA_IE.helperGenerationConfigJson,
        metadata.get("helper_generation_config_json"),
    )
    _add_literal(graph, instruction_uri, SCHEMA_IE.helperProvider, metadata.get("helper_provider"))
    _add_literal(graph, instruction_uri, SCHEMA_IE.helperModel, metadata.get("helper_model"))
    _add_literal(graph, instruction_uri, SCHEMA_IE.helperInstructionHash, metadata.get("helper_instruction_hash"))
    _add_literal(graph, instruction_uri, SCHEMA_IE.helperTemperature, metadata.get("helper_temperature"), XSD.double)
    _add_literal(graph, instruction_uri, SCHEMA_IE.helperBatchSize, metadata.get("helper_batch_size"), XSD.integer)
    _add_literal(graph, instruction_uri, SCHEMA_IE.helperMaxPdfChars, metadata.get("helper_max_pdf_chars"), XSD.integer)
    _add_literal(
        graph,
        instruction_uri,
        SCHEMA_IE.helperTechnicalDataOnly,
        metadata.get("helper_technical_data_only"),
        XSD.boolean,
    )
    _add_literal(
        graph,
        instruction_uri,
        SCHEMA_IE.helperTechnicalPropertiesOnly,
        metadata.get("helper_technical_properties_only"),
        XSD.boolean,
    )


def _clean_sequence(values: Sequence[Any] | None) -> list[str]:
    cleaned = []
    seen = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _add_literal(graph: Graph, subject, predicate, value: Any, datatype=None) -> None:
    if value is None or value == "":
        return
    graph.add((subject, predicate, RDFLiteral(value, datatype=datatype)))


def _add_unique_literals(graph: Graph, subject, predicate, values: Sequence[Any], datatype=None) -> None:
    seen = set()
    for value in values:
        text = str(value).strip() if value is not None else ""
        if not text or text in seen:
            continue
        seen.add(text)
        _add_literal(graph, subject, predicate, value, datatype)
