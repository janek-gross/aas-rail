"""
Inferencing tab for running schema-based extraction.
"""

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from schema_based_ie.experiments.run_inference import run_inference
from schema_based_ie.langgraphs.generic_pipeline import Cfg, ICLCfg

from utils import (
    build_display_rows,
    build_generic_pipeline_dataset,
    count_property_definitions,
    make_json_safe,
    normalize_result_payload,
    parse_extraction_result,
    save_cached_inference_result,
)

DEFAULT_EXTRACTION_CONFIG = Cfg(icl_cfg=ICLCfg()).model_dump()

CACHE_KEY_LAST_RECORD = "last_inference_record"
CACHE_KEY_LAST_RESULT = "last_inference_result"


def render_inferencing_tab() -> None:
    """Render the inferencing tab for running extraction."""
    st.subheader("Inferencing")

    upload_state = st.session_state.get("upload_state", {})
    source_text = upload_state.get("source_text")
    property_definitions = upload_state.get("property_definitions", [])
    cached_record = st.session_state.get(CACHE_KEY_LAST_RECORD)

    has_inputs = bool(upload_state.get("datasheet_bytes")) and bool(property_definitions)

    if not has_inputs:
        display_cached_inference_result(cached_record)
        st.warning("Upload a datasheet and property definitions before running a new inference.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Datasheet", upload_state.get("datasheet_name") or "No file")
    with col2:
        num_props = count_property_definitions(property_definitions)
        st.metric("Properties", num_props)
    with col3:
        st.metric("Text length", f"{len(source_text)} chars" if source_text else "No text")

    with st.expander("Extraction configuration", expanded=False):
        config_json = st.text_area(
            "Extraction config JSON",
            value=json.dumps(DEFAULT_EXTRACTION_CONFIG, indent=2),
            height=300,
            key="config_json",
        )

    run_button = st.button("Run Inference", key="run_inference_btn")

    if not run_button:
        display_cached_inference_result(cached_record)
        return

    datasheet_name = upload_state.get("datasheet_name")
    datasheet_bytes = upload_state.get("datasheet_bytes")

    sample_dir = None
    temp_root = None
    try:
        with st.spinner("Running generic pipeline inference..."):
            sample_dir = build_generic_pipeline_dataset(datasheet_name, datasheet_bytes, property_definitions)
            temp_root = Path(sample_dir).parent

            inference_cfg = parse_config(config_json)
            raw_result = run_inference(input_path=sample_dir, cfg=inference_cfg)

            metadata = {
                "datasheet_name": datasheet_name,
                "property_count": count_property_definitions(property_definitions),
                "config": inference_cfg,
            }
            cached_record = save_cached_inference_result(raw_result, metadata)
            st.session_state[CACHE_KEY_LAST_RECORD] = cached_record
            st.session_state[CACHE_KEY_LAST_RESULT] = cached_record["result"]

        st.success("Inference completed.")
        display_inference_result(cached_record["result"])

    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        import traceback

        with st.expander("Traceback"):
            st.code(traceback.format_exc())
    finally:
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root)


def parse_config(config_json: str) -> Dict[str, Any]:
    """Parse the inference config textarea, falling back to the default Cfg model."""
    try:
        parsed_cfg = json.loads(config_json)
    except json.JSONDecodeError:
        parsed_cfg = None

    if not isinstance(parsed_cfg, dict):
        parsed_cfg = deepcopy(DEFAULT_EXTRACTION_CONFIG)

    parsed_cfg = expand_flat_extractor_config(parsed_cfg)

    if "id_cfg" not in parsed_cfg:
        parsed_cfg["id_cfg"] = deepcopy(DEFAULT_EXTRACTION_CONFIG["id_cfg"])

    return parsed_cfg


def expand_flat_extractor_config(parsed_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Support the older flat extractor config shape used by early app drafts."""
    flat_keys = {
        "client",
        "model",
        "prompt",
        "temperature",
        "prompt_degradation_intensity",
        "response_schema",
        "batch_size",
        "timeout",
        "retry",
    }
    if "schema_based_extractor_cfg" in parsed_cfg or not flat_keys.intersection(parsed_cfg):
        return parsed_cfg

    cfg = deepcopy(DEFAULT_EXTRACTION_CONFIG)
    extractor_cfg = cfg["schema_based_extractor_cfg"]
    extractor_cfg["client_cfg"] = {
        "client_name": parsed_cfg.get("client", extractor_cfg["client_cfg"]["client_name"]),
        "model": parsed_cfg.get("model", extractor_cfg["client_cfg"]["model"]),
    }

    for key in flat_keys - {"client", "model"}:
        if key in parsed_cfg:
            extractor_cfg[key] = parsed_cfg[key]

    if "id_cfg" in parsed_cfg:
        cfg["id_cfg"] = parsed_cfg["id_cfg"]
    if "icl_cfg" in parsed_cfg:
        cfg["icl_cfg"] = parsed_cfg["icl_cfg"]
    if "query_generator_cfg" in parsed_cfg:
        cfg["query_generator_cfg"] = parsed_cfg["query_generator_cfg"]

    return cfg


def display_cached_inference_result(cached_record: Optional[Dict[str, Any]]) -> None:
    """Display the persisted inference result, if available."""
    if not cached_record or "result" not in cached_record:
        return

    metadata = cached_record.get("metadata", {})
    saved_at = cached_record.get("saved_at")
    datasheet_name = metadata.get("datasheet_name")

    details = []
    if datasheet_name:
        details.append(str(datasheet_name))
    if saved_at:
        details.append(f"saved {saved_at}")
    suffix = f" ({', '.join(details)})" if details else ""

    st.info(f"Last inference result loaded from cache{suffix}.")
    display_inference_result(cached_record["result"])


def display_inference_result(raw_result: Any) -> None:
    """Display the inference result with extraction table and raw JSON."""
    parsed_result = parse_extraction_result(raw_result.get("parsed", raw_result) if isinstance(raw_result, dict) else raw_result)
    normalized = normalize_result_payload(parsed_result)

    st.write("#### Extraction Results")

    extraction_data = raw_result.get("extraction_result") if isinstance(raw_result, dict) else None
    if extraction_data is None:
        extraction_data = normalized

    parsed_extraction = parse_extraction_result(extraction_data)
    extraction_rows = build_display_rows(parsed_extraction)

    if extraction_rows:
        st.write("##### Extracted Properties")
        st.table(extraction_rows[:50])
    else:
        st.json(make_json_safe(parsed_extraction))

    with st.expander("Raw result JSON", expanded=False):
        st.code(json.dumps(make_json_safe(raw_result), indent=2), language="json")

    st.download_button(
        "Download results as JSON",
        data=json.dumps(make_json_safe({"result": normalized}), indent=2),
        file_name="aas_rail_results.json",
        mime="application/json",
    )
