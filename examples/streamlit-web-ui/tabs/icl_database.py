"""
ICL database preparation tab.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from aas_rail.langgraphs.icl.aasx_io import (
    collect_property_records_from_aasx_bytes,
    filter_property_records_to_technical_data,
    filter_property_records_to_technical_properties,
)
from aas_rail.langgraphs.icl.rdf_property_grouping import group_property_records
from aas_rail.langgraphs.icl.icl_database_creation import (
    DEFAULT_NEO4J_IMPORT_DIR,
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USER,
    convert_aasx_files_to_turtle,
    convert_paired_aasx_pdf_files_to_turtle,
    import_turtle_exports_to_neo4j,
    reset_neo4j_database,
    write_turtle_exports,
)
from aas_rail.model_clients.client_configs import EmbeddingCfg
from aas_rail.langgraphs.icl.extraction_helper_pipeline import (
    ExtractionHelperCfg,
    ExtractionHelperClientCfg,
)

from utils import CACHE_DIR


TTL_OUTPUT_DIR = CACHE_DIR / "ttl"
DEBUG_OUTPUT_DIR = CACHE_DIR / "debug"
EMBEDDING_PROVIDER_OPTIONS = ["llama", "ollama", "openai", "google", "sentence_transformer"]


def render_icl_database_tab() -> None:
    """Render the ICL database preparation tab."""
    st.subheader("ICL Database")

    aasx_file_uploads = st.file_uploader(
        "AASX files",
        type=["aasx"],
        accept_multiple_files=True,
        key="icl_aasx_files",
    )
    pdf_file_uploads = st.file_uploader(
        "Matching PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="icl_pdf_files",
    )
    folder_uploads = st.file_uploader(
        "ICL examples folder",
        type=["aasx", "pdf"],
        accept_multiple_files="directory",
        key="icl_examples_folder",
        help="Select a folder containing paired AASX and PDF files. Subfolders are included.",
    )
    folder_aasx_files, folder_pdf_files = split_icl_examples_folder_uploads(folder_uploads or [])
    aasx_files = dedupe_uploaded_files([*(aasx_file_uploads or []), *folder_aasx_files])
    pdf_files = dedupe_uploaded_files([*(pdf_file_uploads or []), *folder_pdf_files])

    if aasx_files or pdf_files:
        st.write(f"{len(aasx_files or [])} AASX file(s), {len(pdf_files or [])} PDF file(s) selected")
        if aasx_files:
            pdf_by_stem = {Path(file.name).stem: file.name for file in pdf_files or []}
            st.table(
                {
                    "AASX": [file.name for file in aasx_files],
                    "Matched PDF": [pdf_by_stem.get(Path(file.name).stem, "") for file in aasx_files],
                }
            )

    with st.expander("Extraction helper generation", expanded=False):
        generate_instructions = st.checkbox(
            "Generate extraction helpers",
            value=True,
            key="icl_generate_extraction_instructions",
        )
        technical_data_only = st.checkbox(
            "Restrict to TechnicalData submodels",
            value=True,
            key="icl_instruction_technical_data_only",
            disabled=not generate_instructions,
        )
        technical_properties_only = st.checkbox(
            "Restrict to TechnicalProperties collection",
            value=False,
            key="icl_instruction_technical_properties_only",
            disabled=not (generate_instructions and technical_data_only),
        )
        provider = st.selectbox(
            "LLM provider",
            options=["llama", "openai", "google", "ollama", "anthropic"],
            index=0,
            key="icl_instruction_provider",
            disabled=not generate_instructions,
        )
        default_models = {
            "llama": "ggml-org/gpt-oss-20b-GGUF:MXFP4",
            "openai": "gpt-4o-mini",
            "google": "gemini-2.5-flash-lite",
            "ollama": "gpt-oss:20b",
            "anthropic": "claude-sonnet-4-5",
        }
        model = st.text_input(
            "Model",
            value=default_models[provider],
            key="icl_instruction_model",
            disabled=not generate_instructions,
        )
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=100,
            value=24,
            step=1,
            key="icl_instruction_batch_size",
            disabled=not generate_instructions,
        )
        max_pdf_chars = st.number_input(
            "Max PDF text chars",
            min_value=1000,
            max_value=200000,
            value=200000,
            step=5000,
            key="icl_instruction_max_pdf_chars",
            disabled=not generate_instructions,
        )
        helper_artifact_id = st.text_input(
            "Artifact ID",
            value="adhoc",
            key="icl_helper_artifact_id",
            disabled=not generate_instructions,
        )

    with st.expander("Datasheet embeddings", expanded=False):
        default_embedding_cfg = EmbeddingCfg()
        default_embedding_provider = default_embedding_cfg.client_cfg.client_name
        embedding_provider_options = [default_embedding_provider] + [
            provider for provider in EMBEDDING_PROVIDER_OPTIONS if provider != default_embedding_provider
        ]
        generate_datasheet_embeddings = st.checkbox(
            "Generate product datasheet embeddings",
            value=True,
            key="icl_generate_datasheet_embeddings",
        )
        embedding_provider = st.selectbox(
            "Embedding provider",
            options=embedding_provider_options,
            index=0,
            key="icl_embedding_provider",
            disabled=not generate_datasheet_embeddings,
        )
        embedding_model = st.text_input(
            "Embedding model",
            value=default_embedding_cfg.client_cfg.model,
            key="icl_embedding_model",
            disabled=not generate_datasheet_embeddings,
        )
        embed_col1, embed_col2, embed_col3 = st.columns(3)
        with embed_col1:
            embedding_chunk_size = st.number_input(
                "Embedding chunk size",
                min_value=100,
                max_value=20000,
                value=default_embedding_cfg.chunk_size,
                step=50,
                key="icl_embedding_chunk_size",
                disabled=not generate_datasheet_embeddings,
            )
        with embed_col2:
            embedding_chunk_overlap = st.number_input(
                "Embedding chunk overlap",
                min_value=0,
                max_value=5000,
                value=default_embedding_cfg.chunk_overlap,
                step=20,
                key="icl_embedding_chunk_overlap",
                disabled=not generate_datasheet_embeddings,
            )
        with embed_col3:
            embedding_max_pdf_chars = st.number_input(
                "Embedding max PDF chars",
                min_value=1000,
                max_value=500000,
                value=default_embedding_cfg.max_pdf_chars,
                step=10000,
                key="icl_embedding_max_pdf_chars",
                disabled=not generate_datasheet_embeddings,
            )

    if generate_instructions and aasx_files:
        preview_payloads = tuple((uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in aasx_files)
        with st.spinner("Preparing extraction helper preview..."):
            preview = build_extraction_helper_preview(
                preview_payloads,
                technical_data_only,
                technical_properties_only,
            )

        st.write("#### Extraction helper preview")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Value properties", preview["selected_property_count"])
        with metric_col2:
            st.metric("Helpers to extract", preview["helper_count"])
        with metric_col3:
            st.metric("All value properties", preview["total_property_count"])

        if preview["rows"]:
            st.dataframe(preview["rows"], use_container_width=True, hide_index=True)
        if preview["errors"]:
            with st.expander("Preview errors", expanded=False):
                st.dataframe(preview["errors"], use_container_width=True, hide_index=True)

    needs_pdf = generate_instructions or generate_datasheet_embeddings
    process_disabled = not aasx_files or (needs_pdf and not pdf_files)
    if st.button("Convert to Turtle", key="process_aasx_btn", disabled=process_disabled):
        with st.spinner("Processing AASX files..."):
            aasx_payloads = [(uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in aasx_files]
            if needs_pdf:
                pdf_payloads = [(uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in pdf_files]
                instruction_cfg = ExtractionHelperCfg(
                    client_cfg=ExtractionHelperClientCfg(client_name=provider, model=model),
                    batch_size=int(batch_size),
                    max_pdf_chars=int(max_pdf_chars),
                )
                embedding_cfg = EmbeddingCfg(
                    client_cfg={
                        "client_name": embedding_provider,
                        "model": embedding_model,
                    },
                    chunk_size=int(embedding_chunk_size),
                    chunk_overlap=int(embedding_chunk_overlap),
                    max_pdf_chars=int(embedding_max_pdf_chars),
                )
                turtle_results = convert_paired_aasx_pdf_files_to_turtle(
                    aasx_payloads,
                    pdf_payloads,
                    instruction_cfg=instruction_cfg,
                    embedding_cfg=embedding_cfg,
                    generate_instructions=generate_instructions,
                    generate_datasheet_embeddings=generate_datasheet_embeddings,
                    technical_data_only=technical_data_only,
                    technical_properties_only=technical_properties_only,
                    helper_artifact_id=helper_artifact_id,
                    debug_output_dir=DEBUG_OUTPUT_DIR,
                )
            else:
                turtle_results = convert_aasx_files_to_turtle(aasx_payloads)
            turtle_results = write_turtle_exports(turtle_results, TTL_OUTPUT_DIR)
            st.session_state["icl_turtle_results"] = [result.model_dump() for result in turtle_results]

    turtle_results = st.session_state.get("icl_turtle_results", [])
    if turtle_results:
        st.write("#### Turtle exports")
        st.table(
            {
                "File": [result["source_name"] for result in turtle_results],
                "PDF": [result.get("pdf_name") or "" for result in turtle_results],
                "Status": [result["status"] for result in turtle_results],
                "Triples": [result["triple_count"] for result in turtle_results],
                "Properties": [result.get("property_record_count", 0) for result in turtle_results],
                "Instructions": [result.get("instruction_count", 0) for result in turtle_results],
                "Linked": [result.get("linked_instruction_count", 0) for result in turtle_results],
                "Instruction status": [result.get("instruction_status") or "" for result in turtle_results],
                "Embedding status": [result.get("embedding_status") or "" for result in turtle_results],
                "Embedding dims": [result.get("embedding_dimension", 0) for result in turtle_results],
                "Embedding chunks": [result.get("embedding_chunk_count", 0) for result in turtle_results],
                "Helper hash": [result.get("helper_generation_hash") or "" for result in turtle_results],
                "Artifact IDs": [", ".join(result.get("helper_artifact_ids") or []) for result in turtle_results],
                "Instruction errors": [len(result.get("instruction_errors") or []) for result in turtle_results],
                "Embedding error": [result.get("embedding_error") or "" for result in turtle_results],
                "Debug state": [result.get("debug_output_path") or "" for result in turtle_results],
                "Error": [result.get("error") or "" for result in turtle_results],
                "TTL bytes": [len(result.get("turtle") or "") for result in turtle_results],
            }
        )

        for result in turtle_results:
            if result.get("turtle"):
                with st.expander(f"Preview: {result['ttl_name']}", expanded=False):
                    st.code(result["turtle"][:4000], language="turtle")

                embedding_snippet = _embedding_turtle_snippet(result["turtle"])
                if embedding_snippet:
                    with st.expander(f"Datasheet embedding RDF: {result['ttl_name']}", expanded=False):
                        st.code(embedding_snippet, language="turtle")

                st.download_button(
                    f"Download {result['ttl_name']}",
                    data=result["turtle"],
                    file_name=result["ttl_name"],
                    mime="text/turtle",
                    key=f"download_{result['ttl_name']}",
                )
            if result.get("instruction_errors"):
                with st.expander(f"Instruction generation errors: {result['source_name']}", expanded=False):
                    st.code("\n\n".join(result["instruction_errors"]))

    st.write("#### Neo4j import")
    col1, col2, col3 = st.columns(3)
    with col1:
        neo4j_uri = st.text_input("Neo4j URI", DEFAULT_NEO4J_URI, key="neo4j_uri")
    with col2:
        neo4j_user = st.text_input("Neo4j user", DEFAULT_NEO4J_USER, key="neo4j_user")
    with col3:
        neo4j_password = st.text_input(
            "Neo4j password",
            value=DEFAULT_NEO4J_PASSWORD,
            type="password",
            key="neo4j_pass",
        )

    neo4j_import_dir = st.text_input(
        "Neo4j import directory for n10s.fetch (optional)",
        value=DEFAULT_NEO4J_IMPORT_DIR,
        key="neo4j_import_dir",
    )

    reset_col, load_col = st.columns([1, 2])
    with reset_col:
        confirm_reset = st.checkbox("Confirm reset", key="confirm_neo4j_reset")
        if st.button("Reset Neo4j database", key="reset_neo4j_btn", disabled=not confirm_reset):
            with st.spinner("Resetting Neo4j database..."):
                reset_result = reset_neo4j_database(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
            if reset_result.success:
                st.success(
                    f"Deleted {reset_result.nodes_deleted} node(s) and "
                    f"{reset_result.relationships_deleted} relationship(s)."
                )
            else:
                st.error(reset_result.message)

    with load_col:
        load_clicked = st.button("Load Turtle into Neo4j", key="load_neo4j_btn", disabled=not turtle_results)

    if load_clicked:
        with st.spinner("Importing Turtle into Neo4j with n10s..."):
            import_results = import_turtle_exports_to_neo4j(
                turtle_results,
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                neo4j_import_dir=neo4j_import_dir or None,
            )

        st.table(
            {
                "File": [result.source_name for result in import_results],
                "Status": ["Imported" if result.success else "Failed" for result in import_results],
                "Mode": [result.mode for result in import_results],
                "Triples": [result.triples_loaded for result in import_results],
                "Message": [result.message for result in import_results],
            }
        )

    if TTL_OUTPUT_DIR.exists():
        st.caption(f"Turtle files are written to {Path(TTL_OUTPUT_DIR)}")


@st.cache_data(show_spinner=False)
def build_extraction_helper_preview(
    aasx_payloads: tuple[tuple[str, bytes], ...],
    technical_data_only: bool,
    technical_properties_only: bool,
) -> dict[str, Any]:
    """Build a no-LLM preview of extraction-helper grouping."""
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    total_property_count = 0
    selected_property_count = 0

    for source_name, file_bytes in aasx_payloads:
        try:
            property_records = collect_property_records_from_aasx_bytes(file_bytes, source_name=source_name)
            total_property_count += len(property_records)
            if technical_data_only:
                if technical_properties_only:
                    property_records = filter_property_records_to_technical_properties(property_records)
                else:
                    property_records = filter_property_records_to_technical_data(property_records)
            selected_property_count += len(property_records)

            for group in group_property_records(property_records):
                rows.append(
                    {
                        "AASX": source_name,
                        "Response field": group.response_field_name,
                        "idShort": group.id_short,
                        "Semantic ID": group.semantic_id or "",
                        "Known values": "; ".join(_unique_non_empty(record.value for record in group.records)[:4]),
                        "Unit": group.unit or "",
                        "Datatype": group.datatype or "",
                        "Properties sharing helper": len(group.records),
                        "Submodels": "; ".join(
                            _unique_non_empty(
                                record.submodel_id_short or record.submodel_id for record in group.records
                            )[:4]
                        ),
                    }
                )
        except Exception as exc:
            errors.append({"AASX": source_name, "Error": str(exc)})

    return {
        "rows": rows,
        "errors": errors,
        "total_property_count": total_property_count,
        "selected_property_count": selected_property_count,
        "helper_count": len(rows),
    }


def _unique_non_empty(values: Any) -> list[str]:
    unique_values = []
    seen_values = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen_values:
            continue
        seen_values.add(text)
        unique_values.append(text)
    return unique_values


def split_icl_examples_folder_uploads(uploaded_files: list[Any]) -> tuple[list[Any], list[Any]]:
    """Split directory-uploaded ICL example files into AASX and PDF lists."""
    aasx_files = []
    pdf_files = []
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".aasx":
            aasx_files.append(uploaded_file)
        elif suffix == ".pdf":
            pdf_files.append(uploaded_file)
    return aasx_files, pdf_files


def dedupe_uploaded_files(uploaded_files: list[Any]) -> list[Any]:
    """Keep one uploaded file per browser-provided path/name."""
    deduped_files = []
    seen_names = set()
    for uploaded_file in uploaded_files:
        if uploaded_file.name in seen_names:
            continue
        seen_names.add(uploaded_file.name)
        deduped_files.append(uploaded_file)
    return deduped_files


def _embedding_turtle_snippet(turtle: str) -> str:
    marker = "schemaie:ProductDatasheetEmbedding"
    marker_index = turtle.find(marker)
    if marker_index < 0:
        return ""

    subject_start = turtle.rfind("\n\n", 0, marker_index)
    subject_start = 0 if subject_start < 0 else subject_start + 2
    subject_end = turtle.find("\n\n", marker_index)
    if subject_end < 0:
        subject_end = len(turtle)

    snippet = turtle[subject_start:subject_end].strip()
    embedding_start = snippet.find("schemaie:embeddingJson")
    if embedding_start < 0:
        return snippet

    embedding_end = snippet.find(";", embedding_start)
    if embedding_end < 0:
        embedding_end = snippet.find(".", embedding_start)
    if embedding_end < 0:
        return snippet

    return (
        snippet[:embedding_start]
        + 'schemaie:embeddingJson "[embedding vector omitted in preview]"'
        + snippet[embedding_end:]
    )
