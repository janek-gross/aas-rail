"""
ICL database query tab.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import streamlit as st

from schema_based_ie.langgraphs.icl.icl_database_creation import (
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USER,
)
from schema_based_ie.langgraphs.icl.rdf_queries import (
    normalize_property_definition_params,
    query_product_metadata,
    query_property_values_with_metadata,
)

from utils import count_property_definitions, make_json_safe


DEFINITION_DISPLAY_SCOPES = {
    "All queried definitions": "all",
    "Definitions with TechnicalData matches": "technical_data",
    "Definitions with TechnicalProperties matches": "technical_properties",
}


def render_icl_query_tab() -> None:
    """Render controls for querying uploaded definitions against the ICL database."""
    st.subheader("ICL Query")

    upload_state = st.session_state.get("upload_state", {})
    property_definitions = upload_state.get("property_definitions", [])
    has_definitions = bool(property_definitions)

    if not has_definitions:
        st.warning("Upload property definitions before querying the ICL database.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Definitions", count_property_definitions(property_definitions))
    with col2:
        st.metric("Datasheet", upload_state.get("datasheet_name") or "No datasheet")
    with col3:
        st.metric("Query source", "Neo4j ICL")

    st.write("#### Neo4j connection")
    conn_col1, conn_col2, conn_col3 = st.columns(3)
    with conn_col1:
        neo4j_uri = st.text_input("Neo4j URI", DEFAULT_NEO4J_URI, key="icl_query_neo4j_uri")
    with conn_col2:
        neo4j_user = st.text_input("Neo4j user", DEFAULT_NEO4J_USER, key="icl_query_neo4j_user")
    with conn_col3:
        neo4j_password = st.text_input(
            "Neo4j password",
            value=DEFAULT_NEO4J_PASSWORD,
            type="password",
            key="icl_query_neo4j_pass",
        )

    st.write("#### Similarity filters")
    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)
    with filter_col1:
        target_eclass_id = st.text_input("Target eClass ID", key="icl_query_target_eclass")
    with filter_col2:
        target_manufacturer_name = st.text_input("Target manufacturer", key="icl_query_target_manufacturer")
    with filter_col3:
        technical_data_only = st.checkbox(
            "TechnicalData only",
            value=True,
            key="icl_query_technical_data_only",
        )
    with filter_col4:
        technical_properties_only = st.checkbox(
            "TechnicalProperties path only",
            value=False,
            key="icl_query_technical_properties_only",
        )
    with filter_col5:
        query_limit = st.number_input(
            "Max raw matches",
            min_value=10,
            max_value=50000,
            value=5000,
            step=500,
            key="icl_query_limit",
        )

    display_scope_label = st.selectbox(
        "Displayed definitions",
        options=list(DEFINITION_DISPLAY_SCOPES),
        index=0,
        key="icl_query_display_scope",
    )
    display_scope = DEFINITION_DISPLAY_SCOPES[display_scope_label]

    with st.expander("Helper filters", expanded=False):
        helper_filter_col1, helper_filter_col2 = st.columns(2)
        with helper_filter_col1:
            helper_artifact_id = st.text_input("Artifact ID", key="icl_query_helper_artifact_id")
            helper_provider = st.text_input("Helper provider", key="icl_query_helper_provider")
        with helper_filter_col2:
            helper_generation_hash = st.text_input("Helper generation hash", key="icl_query_helper_generation_hash")
            helper_model = st.text_input("Helper model", key="icl_query_helper_model")

    metadata_col, query_col = st.columns([1, 2])
    with metadata_col:
        metadata_clicked = st.button("Load product metadata", key="icl_query_metadata_btn")
    with query_col:
        query_clicked = st.button("Query uploaded definitions", key="icl_query_run_btn")

    if metadata_clicked:
        with st.spinner("Loading ICL product metadata..."):
            try:
                st.session_state["icl_query_product_metadata"] = query_product_metadata(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password,
                )
            except Exception as exc:
                st.error(f"Unable to load product metadata: {exc}")

    product_metadata = st.session_state.get("icl_query_product_metadata", [])
    if product_metadata:
        with st.expander("Available ICL products", expanded=False):
            st.dataframe(product_metadata, use_container_width=True, hide_index=True)

    if query_clicked:
        with st.spinner("Querying ICL property values..."):
            try:
                query_rows = query_property_values_with_metadata(
                    property_definitions,
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password,
                    target_eclass_id=target_eclass_id,
                    target_manufacturer_name=target_manufacturer_name,
                    limit=int(query_limit),
                    technical_data_only=technical_data_only,
                    technical_properties_only=technical_properties_only,
                    helper_generation_hash=helper_generation_hash,
                    helper_artifact_id=helper_artifact_id,
                    helper_provider=helper_provider,
                    helper_model=helper_model,
                )
                st.session_state["icl_query_rows"] = query_rows
            except Exception as exc:
                st.error(f"ICL query failed: {exc}")

    query_rows = st.session_state.get("icl_query_rows", [])
    if not query_rows:
        return

    display_rows = build_definition_result_rows(property_definitions, query_rows, display_scope=display_scope)
    st.write("#### Retrieved property information")
    st.caption(
        "Raw matches are Neo4j result rows; unique value groups collapse repeated values per source; "
        "matched value rows are the rows retained inside those groups."
    )
    if display_rows:
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No retrieved property definitions match the current display filter.")

    with st.expander("Raw query rows", expanded=False):
        st.code(json.dumps(make_json_safe(query_rows), indent=2), language="json")

    st.download_button(
        "Download ICL query results",
        data=json.dumps(make_json_safe({"rows": query_rows, "by_definition": display_rows}), indent=2),
        file_name="icl_query_results.json",
        mime="application/json",
    )


def build_definition_result_rows(
    definitions: Any,
    query_rows: list[dict[str, Any]],
    display_scope: str = "all",
) -> list[dict[str, Any]]:
    """Build one table row per uploaded definition."""
    definition_params = normalize_property_definition_params(definitions)
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for query_row in query_rows:
        key = (
            str(query_row.get("requestedPropertyId") or ""),
            str(query_row.get("requestedPropertyName") or ""),
        )
        grouped_rows[key].append(query_row)

    display_rows = []
    for definition in definition_params:
        key = (definition["property_id"], definition["property_name"])
        matches = filter_matches_for_display_scope(grouped_rows.get(key, []), display_scope)
        if display_scope != "all" and not matches:
            continue
        deduplicated_matches = deduplicate_property_value_matches(matches)
        display_rows.append(
            {
                "Property ID": definition["property_id"],
                "Property name": definition["property_name"],
                "Raw matches": len(matches),
                "Unique value groups": len(deduplicated_matches),
                "Matched value rows": sum(match["sourceRowCount"] for match in deduplicated_matches),
                "Retrieved property information": [format_match(match) for match in deduplicated_matches],
            }
        )

    return display_rows


def deduplicate_property_value_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group repeated property values and prefer TechnicalData duplicates."""
    grouped_matches: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)

    for match in matches:
        values = clean_match_values(match.get("values"))
        if not values:
            continue

        key = (
            str(match.get("sourceName") or ""),
            tuple(normalize_value_for_deduplication(value) for value in values),
        )
        grouped_matches[key].append({**match, "values": values})

    deduplicated_matches = []
    for duplicate_group in grouped_matches.values():
        kept_group = duplicate_group
        if len(duplicate_group) > 1:
            technical_data_matches = [
                match for match in duplicate_group if is_technical_data_submodel(match)
            ]
            if technical_data_matches:
                kept_group = technical_data_matches
        deduplicated_matches.append(summarize_deduplicated_matches(kept_group))

    deduplicated_matches.sort(
        key=lambda match: (
            -(match.get("similarityScore") or 0),
            match.get("sourceName") or "",
            "; ".join(match.get("values") or []),
            "; ".join(match.get("submodels") or []),
        )
    )
    return deduplicated_matches


def summarize_deduplicated_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    first_match = matches[0]
    scores = [match.get("similarityScore") for match in matches if match.get("similarityScore") is not None]
    return {
        "sourceName": first_match.get("sourceName") or "",
        "values": first_match.get("values") or [],
        "sourceRowCount": len(matches),
        "propertyKeys": unique_non_empty(
            match.get("propertyIdShort") or match.get("semanticId") for match in matches
        ),
        "submodels": unique_non_empty(format_submodel(match) for match in matches),
        "paths": unique_non_empty(format_match_path(match) for match in matches),
        "eclassIds": unique_non_empty(match.get("eclassId") for match in matches),
        "manufacturers": unique_non_empty(match.get("manufacturerName") for match in matches),
        "similarityScore": max(scores) if scores else None,
        "extractionInstructions": unique_instruction_entries(matches),
    }


def format_match(match: dict[str, Any]) -> str:
    values = match.get("values") or []
    value_text = "; ".join(str(value) for value in values[:5])
    if len(values) > 5:
        value_text = f"{value_text}; ..."
    source = match.get("sourceName") or ""
    property_key = "; ".join(match.get("propertyKeys") or [])
    submodel = "; ".join(match.get("submodels") or [])
    path = "; ".join(match.get("paths") or [])
    eclass_id = "; ".join(match.get("eclassIds") or [])
    manufacturer = "; ".join(match.get("manufacturers") or [])
    score = match.get("similarityScore")
    source_row_count = match.get("sourceRowCount", 1)
    instructions = format_extraction_instructions(match.get("extractionInstructions") or [])
    parts = [
        source,
        property_key,
        f"values: {value_text}",
        f"instructions: {instructions}" if instructions else "",
        f"matched rows: {source_row_count}",
        f"submodel: {submodel}" if submodel else "",
        f"path: {path}" if path else "",
        f"eClass: {eclass_id}" if eclass_id else "",
        f"manufacturer: {manufacturer}" if manufacturer else "",
        f"score: {score}" if score is not None else "",
    ]
    return " | ".join(part for part in parts if part)


def unique_instruction_entries(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect distinct extraction-instruction maps from matched rows."""
    instructions = []
    seen = set()
    for match in matches:
        for instruction in match.get("extractionInstructions") or []:
            if not isinstance(instruction, dict):
                continue
            cleaned = {
                key: value
                for key, value in instruction.items()
                if has_instruction_value(value)
            }
            if not cleaned:
                continue
            key = json.dumps(cleaned, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            instructions.append(cleaned)
    return instructions


def format_extraction_instructions(instructions: list[dict[str, Any]]) -> str:
    """Format instruction maps compactly for the results table."""
    formatted = []
    for instruction in instructions[:3]:
        parts = [
            f"examples: {format_instruction_value(instruction.get('referenceExamples'))}"
            if instruction.get("referenceExamples")
            else "",
            f"grounding: {instruction.get('grounding')}" if instruction.get("grounding") else "",
            f"source: {instruction.get('sourceHint')}" if instruction.get("sourceHint") else "",
            f"context: {instruction.get('applicabilityContext')}" if instruction.get("applicabilityContext") else "",
            f"evidence: {instruction.get('sourceEvidencePattern')}" if instruction.get("sourceEvidencePattern") else "",
            f"select: {instruction.get('selectionRule')}" if instruction.get("selectionRule") else "",
            f"transform: {instruction.get('transformationRule')}" if instruction.get("transformationRule") else "",
            f"output: {instruction.get('outputRule')}" if instruction.get("outputRule") else "",
            f"avoid: {format_instruction_value(instruction.get('counterExamples'))}"
            if instruction.get("counterExamples")
            else "",
        ]
        formatted.append("; ".join(str(part) for part in parts if part))
    if len(instructions) > 3:
        formatted.append("...")
    return " || ".join(formatted)


def format_instruction_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def has_instruction_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return str(value).strip() != ""


def clean_match_values(values: Any) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    return [str(value).strip() for value in values if value is not None and str(value).strip()]


def filter_matches_for_display_scope(
    matches: list[dict[str, Any]],
    display_scope: str,
) -> list[dict[str, Any]]:
    if display_scope == "technical_data":
        return [match for match in matches if is_technical_data_match(match)]
    if display_scope == "technical_properties":
        return [match for match in matches if is_technical_properties_match(match)]
    return matches


def normalize_value_for_deduplication(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def is_technical_data_submodel(match: dict[str, Any]) -> bool:
    return is_technical_data_match(match)


def is_technical_data_match(match: dict[str, Any]) -> bool:
    return any(
        "technicaldata" in normalize_identifier(candidate)
        for candidate in path_scope_candidates(match)
    )


def is_technical_properties_match(match: dict[str, Any]) -> bool:
    return any(
        "technicalproperties" in normalize_identifier(candidate)
        for candidate in path_scope_candidates(match)
    )


def path_scope_candidates(match: dict[str, Any]) -> list[Any]:
    candidates: list[Any] = [
        match.get("submodelIdShort"),
        match.get("submodelUri"),
        match.get("elementPath"),
    ]
    candidates.append(format_match_path(match))

    for instruction in match.get("extractionInstructions") or []:
        if not isinstance(instruction, dict):
            continue
        candidates.extend(instruction.get("propertyPaths") or [])

    return candidates


def normalize_identifier(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item or "") for item in value)
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def format_match_path(match: dict[str, Any]) -> str:
    element_path = match.get("elementPath")
    if isinstance(element_path, list):
        return " / ".join(str(part).strip() for part in element_path if str(part or "").strip())

    if element_path:
        return str(element_path).strip()

    for instruction in match.get("extractionInstructions") or []:
        if not isinstance(instruction, dict):
            continue
        property_paths = instruction.get("propertyPaths") or []
        for property_path in property_paths:
            if str(property_path or "").strip():
                return str(property_path).strip()
    return ""


def format_submodel(match: dict[str, Any]) -> str:
    id_short = str(match.get("submodelIdShort") or "").strip()
    if id_short:
        return id_short
    return str(match.get("submodelUri") or "").strip()


def unique_non_empty(values: Any) -> list[str]:
    unique_values = []
    seen_values = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen_values:
            continue
        seen_values.add(text)
        unique_values.append(text)
    return unique_values
