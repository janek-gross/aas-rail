# -> streamlit

"""AASX loading and conversion helpers for ICL database preparation."""

import hashlib
import io
import json
import re
from decimal import Decimal
from typing import Any, Mapping

from basyx.aas import model
from basyx.aas.adapter import aasx
from basyx.aas.model.provider import DictObjectStore

from .extraction_helper_models import RdfPropertyRecord
from .rdf_export import object_store_to_turtle


def load_aasx_object_store(file_bytes: bytes) -> DictObjectStore:
    """Load an AASX file into a BaSyx object store."""
    object_store = DictObjectStore()
    file_store = aasx.DictSupplementaryFileContainer()
    with aasx.AASXReader(io.BytesIO(file_bytes)) as reader:
        reader.read_into(object_store, file_store)
    return object_store


def aasx_bytes_to_turtle(file_bytes: bytes) -> str:
    """Convert AASX bytes to Turtle RDF."""
    object_store = load_aasx_object_store(file_bytes)
    return object_store_to_turtle(object_store)


###################################################################################
##### Collect value-bearing AAS properties as extraction-helper input records. ####
###################################################################################

def collect_property_records_from_object_store(
    object_store: DictObjectStore,
    source_name: str | None = None,
) -> list[RdfPropertyRecord]:
    """Collect value-bearing AAS properties with names, values, and definitions."""
    concept_lookup = _concept_lookup(object_store)
    records: list[RdfPropertyRecord] = []

    for obj in object_store:
        if not isinstance(obj, model.Submodel):
            continue

        submodel_id = str(getattr(obj, "id", "") or "")
        submodel_id_short = str(getattr(obj, "id_short", "") or "")
        for element in _children(obj):
            _collect_property_records(
                element=element,
                concept_lookup=concept_lookup,
                records=records,
                source_name=source_name,
                submodel_id=submodel_id,
                submodel_id_short=submodel_id_short,
                element_path=[],
            )

    return records


def collect_property_records_from_aasx_bytes(
    file_bytes: bytes,
    source_name: str | None = None,
) -> list[RdfPropertyRecord]:
    """Load AASX bytes and collect value-bearing property records."""
    return collect_property_records_from_object_store(load_aasx_object_store(file_bytes), source_name=source_name)


def extract_technical_data_properties_from_aasx_bytes(
    file_bytes: bytes,
    class_id: str = "TechnicalData",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load AASX bytes and extract VAMOS-compatible TechnicalProperties definitions."""
    return extract_technical_data_properties(load_aasx_object_store(file_bytes), class_id=class_id)


def extract_product_metadata_from_aasx_bytes(file_bytes: bytes) -> dict[str, str]:
    """Load AASX bytes and extract product metadata used for ICL target ranking."""
    return extract_product_metadata(load_aasx_object_store(file_bytes))


def extract_product_metadata(object_store: Any) -> dict[str, str]:
    """Extract product-level eClass and manufacturer metadata from an AAS object store."""
    metadata_entries = _metadata_value_entries(object_store)
    eclass_values = _metadata_values_for_id_shorts(
        metadata_entries,
        ("ProductClassId", "ProductGroup", "ClassId"),
    )
    manufacturer_values = _metadata_values_for_id_shorts(
        metadata_entries,
        ("ManufacturerName", "Company", "NameOfSupplier", "Brand"),
    )
    return {
        "eclass_id": _select_eclass_id(eclass_values),
        "manufacturer_name": _first_non_empty(manufacturer_values),
    }


def extract_technical_data_properties(
    object_store: Any,
    class_id: str = "TechnicalData",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract VAMOS-compatible property definitions from TechnicalData/TechnicalProperties.

    This intentionally mirrors the VAMOS evaluation framework's
    ``extract_technical_data_properties`` output shape without importing that project.
    """
    submodel = find_technical_data_submodel(object_store)
    technical_properties = _get_referable(submodel, "TechnicalProperties")
    properties = _property_recursion(_children(technical_properties))
    can_resolve_concepts = hasattr(object_store, "get_identifiable") and object_store is not submodel

    property_definitions: dict[str, Any] = {}
    property_labels: dict[str, Any] = {}
    for prop in properties:
        concept_description = None
        semantic_reference = _semantic_reference_value(prop)
        if semantic_reference and can_resolve_concepts:
            try:
                concept_description = object_store.get_identifiable(semantic_reference)
            except Exception:
                concept_description = None

        prop_id, name, definition, unit = _definition_metadata_from_property(
            prop,
            concept_description,
        )
        value, value_type = _get_value_and_type(prop)

        property_definitions[prop_id] = {
            "id": prop_id,
            "name": name,
            "type": value_type,
            "definition": definition,
            "unit": unit,
            "values": [],
        }
        property_labels[prop_id] = {
            "name": _normalize_id_short(name.get("en", "")),
            "unit": unit,
            "type": value_type,
            "definition": definition.get("en", ""),
            "value": value,
        }

    class_template = {
        "Type": "CustomDictionary",
        "release": "0.0",
        "properties": property_definitions,
        "classes": {
            class_id: {
                "id": class_id,
                "name": _get_class_name(submodel),
                "description": "",
                "keywords": [],
                "properties": list(property_definitions.keys()),
            }
        },
    }
    return class_template, property_labels


def find_technical_data_submodel(object_store: Any):
    """Find the TechnicalData submodel using the same fallback order as VAMOS."""
    if isinstance(object_store, model.Submodel):
        return object_store

    submodels = [item for item in object_store if isinstance(item, model.Submodel)]
    candidates = [
        item for item in submodels
        if item.id_short and item.id_short.lower() == "technicaldata"
    ]
    if candidates:
        return candidates[0]

    candidates = [
        item for item in submodels
        if item.id_short and "technical" in item.id_short.lower()
    ]
    if candidates:
        return candidates[0]

    candidates = [
        item for item in submodels
        if item.semantic_id and "technicaldata" in str(item.semantic_id).lower()
    ]
    if candidates:
        return candidates[0]

    raise ValueError("TechnicalData submodel not found")


def _property_recursion(elements: Any) -> list[Any]:
    collected = []
    for elem in elements or []:
        if isinstance(elem, (model.SubmodelElementCollection, model.SubmodelElementList)):
            collected.extend(_property_recursion(_children(elem)))
        elif isinstance(elem, (model.MultiLanguageProperty, model.Property, model.Range)):
            collected.append(elem)
    return collected


def filter_property_records_to_technical_data(
    property_records: list[RdfPropertyRecord],
) -> list[RdfPropertyRecord]:
    """Keep only records that belong to a TechnicalData submodel."""
    return [record for record in property_records if is_technical_data_property_record(record)]


def filter_property_records_to_technical_properties(
    property_records: list[RdfPropertyRecord],
) -> list[RdfPropertyRecord]:
    """Keep only records that belong to a TechnicalProperties collection."""
    return [record for record in property_records if is_technical_properties_property_record(record)]


def is_technical_data_property_record(record: RdfPropertyRecord) -> bool:
    """Return whether a property record is located in a TechnicalData submodel."""
    candidates = [
        record.submodel_id_short,
        record.submodel_id,
        record.path[0] if record.path else None,
    ]
    return any("technicaldata" in _normalize_identifier(candidate) for candidate in candidates)


def is_technical_properties_property_record(record: RdfPropertyRecord) -> bool:
    """Return whether a property record is located inside a TechnicalProperties element collection."""
    candidates = [
        record.submodel_id_short,
        record.submodel_id,
    ] + (record.path or [])
    return any("technicalproperties" in _normalize_identifier(candidate) for candidate in candidates)


def _collect_property_records(
    *,
    element: Any,
    concept_lookup: dict[str, str],
    records: list[RdfPropertyRecord],
    source_name: str | None,
    submodel_id: str | None,
    submodel_id_short: str | None,
    element_path: list[str],
) -> None:
    element_id_short = str(getattr(element, "id_short", "") or "")
    next_path = element_path + ([element_id_short] if element_id_short else [])

    if isinstance(element, (model.Property, model.MultiLanguageProperty, model.Range)):
        name = _first_text(getattr(element, "display_name", None)) or element_id_short
        semantic_id = _first_reference_value(getattr(element, "semantic_id", None))
        definition = (
            _first_text(getattr(element, "description", None))
            or concept_lookup.get(semantic_id or "", "")
            or f"Value-bearing AAS property {name}."
        )
        value = _element_value_text(element)
        unit = _element_unit_text(element)
        datatype = _element_datatype_text(element)
        path = ([submodel_id_short] if submodel_id_short else []) + next_path
        records.append(
            RdfPropertyRecord(
                record_id=_record_id(source_name, submodel_id, next_path, name, value),
                name=name,
                value=value,
                definition=definition,
                aas_type=type(element).__name__,
                id_short=element_id_short,
                unit=unit,
                datatype=datatype,
                path=path,
                element_path=next_path,
                submodel_id=submodel_id,
                submodel_id_short=submodel_id_short,
                semantic_id=semantic_id,
                source_name=source_name,
            )
        )

    for child in _children(element):
        _collect_property_records(
            element=child,
            concept_lookup=concept_lookup,
            records=records,
            source_name=source_name,
            submodel_id=submodel_id,
            submodel_id_short=submodel_id_short,
            element_path=next_path,
        )


def _children(element: Any) -> list[Any]:
    child_container = None
    if isinstance(element, model.Submodel):
        child_container = getattr(element, "submodel_element", None)
    elif isinstance(element, (model.SubmodelElementCollection, model.SubmodelElementList)):
        child_container = getattr(element, "value", None)

    if not child_container:
        return []
    try:
        return list(child_container)
    except TypeError:
        return []


def _get_referable(element: Any, id_short: str) -> Any:
    if hasattr(element, "get_referable"):
        return element.get_referable(id_short)
    for child in _children(element):
        if getattr(child, "id_short", None) == id_short:
            return child
    raise KeyError(id_short)


def _semantic_reference_value(prop: Any) -> str | None:
    return _first_reference_value(getattr(prop, "semantic_id", None))


def _definition_metadata_from_property(
    prop: Any,
    concept_description: Any | None,
) -> tuple[str, dict[str, str], dict[str, str], str]:
    if concept_description is not None:
        prop_id = str(getattr(concept_description, "id", "") or "")
        embedded_spec = _first_embedded_data_specification_content(concept_description)
        display_name = _first_text(getattr(prop, "display_name", None))
        preferred_name = _first_text(getattr(embedded_spec, "preferred_name", None))
        concept_description_text = _first_text(getattr(concept_description, "description", None))
        embedded_definition = _first_text(getattr(embedded_spec, "definition", None))
        name_text = display_name or preferred_name or str(getattr(concept_description, "id_short", "") or prop_id)
        definition_text = concept_description_text or embedded_definition or name_text or prop_id
        unit = str(getattr(embedded_spec, "unit", "") or "")
        return prop_id, {"en": name_text}, {"en": definition_text}, unit

    prop_id = str(getattr(prop, "id_short", "") or "")
    name_text = prop_id
    definition_text = _first_text(getattr(prop, "description", None)) or prop_id
    return prop_id, {"en": name_text}, {"en": definition_text}, ""


def _first_embedded_data_specification_content(concept_description: Any) -> Any | None:
    embedded_specs = getattr(concept_description, "embedded_data_specifications", None) or []
    for embedded_spec in embedded_specs:
        content = getattr(embedded_spec, "data_specification_content", None)
        if content is not None:
            return content
    return None


def _get_class_name(submodel: Any) -> str:
    try:
        designation = _get_referable(
            _get_referable(submodel, "GeneralInformation"),
            "ManufacturerProductDesignation",
        ).value
    except Exception:
        return ""
    return _first_text(designation)


def _metadata_value_entries(object_store: Any) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    try:
        roots = list(object_store)
    except TypeError:
        roots = [object_store]

    for root in roots:
        _collect_metadata_value_entries(root, entries)
    return entries


def _collect_metadata_value_entries(element: Any, entries: list[tuple[str, str]]) -> None:
    id_short = str(getattr(element, "id_short", "") or "")
    if isinstance(element, (model.Property, model.MultiLanguageProperty, model.Range)):
        value = _element_value_text(element)
        if id_short and value:
            entries.append((id_short, value))

    for child in _children(element):
        _collect_metadata_value_entries(child, entries)


def _metadata_values_for_id_shorts(
    entries: list[tuple[str, str]],
    id_shorts: tuple[str, ...],
) -> list[str]:
    values: list[str] = []
    for target_id_short in id_shorts:
        normalized_target = _normalize_identifier(target_id_short)
        values.extend(
            value
            for id_short, value in entries
            if _normalize_identifier(id_short) == normalized_target
        )
    return values


def _select_eclass_id(values: list[str]) -> str:
    return _first_non_empty([value for value in values if _looks_like_eclass_id(value)]) or _first_non_empty(values)


def _looks_like_eclass_id(value: str) -> bool:
    text = str(value or "")
    return bool(
        re.search(r"(?<!\d)\d{8}(?!\d)", text)
        or re.search(r"(?<!\d)\d{2}[-.\s]\d{2}[-.\s]\d{2}[-.\s]\d{2}(?!\d)", text)
    )


def _first_non_empty(values: list[str]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalize_id_short(id_short: str | None) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]", "_", str(id_short or ""))
    if normalized and not normalized[0].isalpha():
        normalized = "ID_" + normalized
    return normalized[:128]


def _map_value_type(original_type: Any) -> str:
    if original_type in [str, "string", "<class 'str'>"]:
        return "string"
    if original_type in [bool, "bool", "<class 'bool'>"]:
        return "bool"
    if original_type in [model.datatypes.Float, Decimal, int, float]:
        return "numeric"
    return "string"


def _get_value_and_type(prop: Any) -> tuple[Any, str]:
    if isinstance(prop, model.Property):
        value_type = _map_value_type(prop.value_type)
        value = prop.value
    elif isinstance(prop, model.MultiLanguageProperty):
        value_type = "string"
        value = _first_text(prop.value)
    elif isinstance(prop, model.Range):
        value_type = "numeric" if _map_value_type(prop.value_type) == "numeric" else "string"
        value = {
            "min": _json_safe_value(prop.min),
            "max": _json_safe_value(prop.max),
        }
    else:
        value_type, value = "string", ""
    return _json_safe_value(value), value_type


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    return value


def _concept_lookup(object_store: DictObjectStore) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for obj in object_store:
        if not isinstance(obj, model.ConceptDescription):
            continue
        definition = _concept_definition(obj)
        if definition:
            lookup[str(obj.id)] = definition
    return lookup


def _concept_definition(concept: model.ConceptDescription) -> str:
    description = _first_text(getattr(concept, "description", None))
    if description:
        return description

    embedded_specs = getattr(concept, "embedded_data_specifications", None) or []
    for embedded_spec in embedded_specs:
        content = getattr(embedded_spec, "data_specification_content", None)
        definition = _first_text(getattr(content, "definition", None))
        if definition:
            return definition
    return ""


def _element_value_text(element: Any) -> str | None:
    if isinstance(element, model.Range):
        parts = []
        if getattr(element, "min", None) is not None:
            parts.append(f"min: {element.min}")
        if getattr(element, "max", None) is not None:
            parts.append(f"max: {element.max}")
        return ", ".join(parts) if parts else None
    return _first_text(getattr(element, "value", None)) or None


def _element_unit_text(element: Any) -> str | None:
    unit = getattr(element, "unit", None)
    if unit:
        return str(unit)
    return None


def _element_datatype_text(element: Any) -> str | None:
    value_type = getattr(element, "value_type", None)
    if value_type is None:
        return None
    return str(getattr(value_type, "name", None) or value_type)


def _first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        if value.get("en") is not None:
            return str(value["en"])
        for item in value.values():
            if item is not None:
                return str(item)
        return ""
    if hasattr(value, "items"):
        items = list(value.items())
        for language, text in items:
            if language == "en" and text is not None:
                return str(text)
        for _, text in items:
            if text is not None:
                return str(text)
        return ""
    return str(value)


def _first_reference_value(reference: Any) -> str | None:
    if reference is None:
        return None

    for attr_name in ("key", "keys"):
        keys = getattr(reference, attr_name, None)
        if keys is None:
            continue
        try:
            iterable = list(keys)
        except TypeError:
            iterable = [keys]
        for key in iterable:
            value = getattr(key, "value", None)
            if value:
                return str(value)
    return str(reference) if reference else None


def _normalize_identifier(value: Any) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _record_id(
    source_name: str | None,
    submodel_id: str | None,
    element_path: list[str],
    name: str,
    value: str | None,
) -> str:
    payload = json.dumps(
        {
            "source_name": source_name or "",
            "submodel_id": submodel_id or "",
            "element_path": element_path,
            "name": name,
            "value": value or "",
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
