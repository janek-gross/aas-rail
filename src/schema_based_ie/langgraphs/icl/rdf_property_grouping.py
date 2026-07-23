"""RDF property grouping for extraction-helper generation."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

from .extraction_helper_models import PropertyRecord, PropertyRecordMember, RdfPropertyRecord


def coerce_property_records(
    property_records: Sequence[RdfPropertyRecord | Mapping[str, Any]],
) -> list[RdfPropertyRecord]:
    return [
        record if isinstance(record, RdfPropertyRecord) else RdfPropertyRecord.model_validate(record)
        for record in property_records
    ]


def group_property_records(
    property_records: Sequence[RdfPropertyRecord | Mapping[str, Any]],
) -> list[PropertyRecord]:
    """Deduplicate property records for LLM extraction-helper generation.

    Semantic IDs are authoritative. Records without semantic IDs are grouped by
    id_short, value, unit, and datatype.
    """
    records = coerce_property_records(property_records)
    grouped_records: dict[str, list[RdfPropertyRecord]] = {}
    group_metadata: dict[str, dict[str, str | None]] = {}

    for record in records:
        group_key, metadata = _deduplication_key(record)
        grouped_records.setdefault(group_key, []).append(record)
        group_metadata.setdefault(group_key, metadata)

    groups = [
        PropertyRecord(
            group_key=group_key,
            response_field_name="",
            records=[_member_from_rdf_record(record) for record in grouped_records[group_key]],
            **group_metadata[group_key],
        )
        for group_key in grouped_records
    ]
    return _assign_response_field_names(groups)


def _member_from_rdf_record(record: RdfPropertyRecord) -> PropertyRecordMember:
    return PropertyRecordMember(
        record_id=record.record_id,
        name=record.name,
        value=record.value,
        definition=record.definition,
        path=record.path,
        element_path=record.element_path,
        submodel_id=record.submodel_id,
        submodel_id_short=record.submodel_id_short,
        source_name=record.source_name,
    )


def _deduplication_key(record: RdfPropertyRecord) -> tuple[str, dict[str, str | None]]:
    semantic_id = _clean_text(record.semantic_id)
    id_short = _record_id_short(record)
    unit = _clean_text(record.unit)
    datatype = _clean_text(record.datatype)
    value = _clean_text(record.value)

    if semantic_id:
        return (
            f"semantic:{semantic_id}",
            {
                "semantic_id": semantic_id,
                "id_short": id_short,
                "value": value,
                "unit": unit,
                "datatype": datatype,
            },
        )

    key_payload = "|".join(
        [
            _normalize_key_part(id_short),
            _normalize_key_part(value),
            _normalize_key_part(unit),
            _normalize_key_part(datatype),
        ]
    )
    return (
        f"fallback:{key_payload}",
        {
            "semantic_id": None,
            "id_short": id_short,
            "value": value,
            "unit": unit,
            "datatype": datatype,
        },
    )


def _assign_response_field_names(groups: Sequence[PropertyRecord]) -> list[PropertyRecord]:
    field_names = _response_field_names(groups)
    return [
        group.model_copy(update={"response_field_name": field_names[index]})
        for index, group in enumerate(groups)
    ]


def _response_field_names(groups: Sequence[PropertyRecord]) -> list[str]:
    if not groups:
        return []

    candidates_by_group = [_candidate_field_names(group) for group in groups]
    selected = [candidates[0] for candidates in candidates_by_group]

    for candidate_index in range(max(len(candidates) for candidates in candidates_by_group)):
        counts = _counts(selected)
        changed = False
        for index, name in enumerate(selected):
            if counts[name] == 1:
                continue
            candidates = candidates_by_group[index]
            replacement = candidates[min(candidate_index + 1, len(candidates) - 1)]
            if replacement != name:
                selected[index] = replacement
                changed = True
        if not changed and all(count == 1 for count in _counts(selected).values()):
            break

    counts = _counts(selected)
    return [
        name if counts[name] == 1 else f"{name}_{_short_hash(groups[index].group_key)}"
        for index, name in enumerate(selected)
    ]


def _candidate_field_names(group: PropertyRecord) -> list[str]:
    representative = group.records[0]
    id_short = _field_segment(group.id_short or representative.name or "property")
    ancestors = [_field_segment(item) for item in _path_ancestors(representative)]
    candidates = [id_short]
    for depth in range(1, len(ancestors) + 1):
        candidates.append("_".join(ancestors[-depth:] + [id_short]))
    candidates.append(f"{id_short}_{_short_hash(group.group_key)}")
    return _dedupe_preserve_order(candidates)


def _path_ancestors(record: PropertyRecordMember) -> list[str]:
    if len(record.element_path) > 1:
        return record.element_path[:-1]
    if len(record.path) > 1:
        return record.path[:-1]
    return []


def _record_id_short(record: RdfPropertyRecord) -> str:
    return (
        _clean_text(record.id_short)
        or (record.element_path[-1] if record.element_path else "")
        or _clean_text(record.name)
        or "property"
    )


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_key_part(value: Any) -> str:
    return _clean_text(value).casefold()


def _field_segment(value: Any) -> str:
    text = _clean_text(value)
    text = re.sub(r"\W+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "property"
    if text[0].isdigit():
        text = f"property_{text}"
    return text


def _counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
