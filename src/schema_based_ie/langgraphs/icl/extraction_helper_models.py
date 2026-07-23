"""Models shared by extraction-helper generation, grouping, and RDF persistence."""

from __future__ import annotations

from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, create_model


class RdfPropertyRecord(BaseModel):
    """A value-bearing property used as input for helper generation."""

    record_id: str
    name: str
    value: str | None = None
    definition: str = ""
    aas_type: str
    id_short: str | None = None
    unit: str | None = None
    datatype: str | None = None
    path: list[str] = Field(default_factory=list)
    element_path: list[str] = Field(default_factory=list)
    submodel_id: str | None = None
    submodel_id_short: str | None = None
    semantic_id: str | None = None
    source_name: str | None = None


class PropertyRecordMember(BaseModel):
    """One source property inside a deduplicated extraction-helper group."""

    record_id: str
    name: str
    value: str | None = None
    definition: str = ""
    path: list[str] = Field(default_factory=list)
    element_path: list[str] = Field(default_factory=list)
    submodel_id: str | None = None
    submodel_id_short: str | None = None
    source_name: str | None = None


class PropertyRecord(BaseModel):
    """Deduplicated property records that should share one extraction helper."""

    group_key: str
    response_field_name: str
    records: list[PropertyRecordMember] = Field(min_length=1)
    semantic_id: str | None = None
    id_short: str
    value: str | None = None
    unit: str | None = None
    datatype: str | None = None


Grounding = Literal["exact", "inferred", "conflicting", "reference_only"]


class ResponseModel(BaseModel):
    """LLM-facing extraction helper payload for one deduplicated property group."""

    model_config = ConfigDict(extra="forbid")

    grounding: Grounding = Field(
        description=(
            "How the reference value is supported by the visible source. "
            "exact: directly visible in the source. "
            "inferred: derivable from visible evidence. "
            "conflicting: visible evidence suggests a different value. "
            "reference_only: not justified by visible evidence."
        )
    )
    evidence: str = Field(
        description=(
            "Visible source evidence that supports the extraction. "
            "Mention the relevant section, label, address, table, title, code, or pattern. "
            "If unsupported, say that the value could not be derived from visible source evidence."
        )
    )
    extraction_rule: str = Field(
        description=(
            "How to select or derive the value from the evidence. "
            "Do not invent labels or lookup paths. "
            "Do not repeat the target output value."
        )
    )
    formatting_rule: str = Field(
        description=(
            "How to format the extracted value like the reference. "
            "Mention units, casing, separators, precision, or whether to copy text as-is. "
            "Do not repeat the target output value."
        )
    )
    avoid: list[str] = Field(
        default_factory=list,
        description=(
            "Nearby misleading labels, values, or strategies to avoid. "
            "Do not include the correct target value."
        ),
    )


class ExtractionHelper(ResponseModel):
    """Internal extraction helper attached to one deduplicated property group."""

    model_config = ConfigDict(extra="forbid")

    group_key: str = ""
    response_field_name: str = ""
    semantic_id: str = ""
    property_record_id: str = ""
    property_name: str = ""


class ExtractionHelperRunResult(BaseModel):
    """Result returned by the extraction-helper graph."""
    
    instructions: list[ExtractionHelper] = Field(default_factory=list)
    usage: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    debug_output_path: str | None = None


def extraction_helper_response_factory(
    property_groups: Sequence[PropertyRecord],
    response_model: type[BaseModel] = ResponseModel,
    model_name: str = "ExtractionHelperResponse",
) -> type[BaseModel]:
    """Build a constrained response schema keyed by stable property group field names."""
    fields: dict[str, Any] = {}
    for group in property_groups:
        fields[group.response_field_name] = (
            response_model,
            Field(..., description=_group_description(group)),
        )
    return create_model(model_name, **fields, __config__=ConfigDict(extra="forbid"))


def property_groups_prompt_payload(groups: Sequence[PropertyRecord]) -> dict[str, dict[str, Any]]:
    """Return prompt data keyed exactly like the response schema fields."""
    payload: dict[str, dict[str, Any]] = {}
    for group in groups:
        representative = group.records[0]
        payload[group.response_field_name] = {
            "definition": ", ".join(unique_non_empty(record.definition for record in group.records)[:1]),
            "reference_value": ", ".join(unique_non_empty(record.value for record in group.records)[:1]),
            "unit": group.unit,
            "datatype": group.datatype,
        }
    return payload


def unique_non_empty(values) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        text = str(value).strip() if value is not None else ""
        if not text or text in seen:
            continue
        seen.add(text)
        unique_values.append(text)
    return unique_values


def _group_description(group: PropertyRecord) -> str:
    names = ", ".join(unique_non_empty(record.name for record in group.records)[:4])
    definitions = unique_non_empty(record.definition for record in group.records)
    definition = definitions[0] if definitions else ""
    return (
        f"Extraction helper for id_short '{group.id_short}'. "
        f"Property names in this group: {names or 'unknown'}. "
        f"Definition: {definition}"
    )
