"""
Upload tab for datasheet and property definitions.
"""

from typing import Any

import streamlit as st

from utils import (
    count_property_definitions,
    get_property_definition_items,
    load_pdf_text,
    load_property_definitions,
)


def render_upload_tab() -> None:
    """Render the upload tab for datasheet and property definitions."""
    st.subheader("Upload")

    col1, col2 = st.columns([1, 1])

    source_text = None
    datasheet_bytes = None
    datasheet_name = None
    datasheet_error = None

    with col1:
        st.write("#### Datasheet")
        datasheet_file = st.file_uploader("Datasheet file", type=["pdf", "txt"], key="datasheet")

        if datasheet_file is not None:
            datasheet_name = datasheet_file.name
            datasheet_bytes = datasheet_file.getvalue()
            try:
                if datasheet_file.type == "text/plain" or datasheet_file.name.lower().endswith(".txt"):
                    source_text = datasheet_bytes.decode("utf-8")
                    st.success("TXT file uploaded.")
                    st.text_area("Processed text", source_text[:8192], height=360, disabled=True)
                else:
                    source_text = load_pdf_text(datasheet_bytes)
                    st.success("PDF file uploaded.")
                    st.pdf(datasheet_bytes, height=640)
                    with st.expander("Processed text", expanded=False):
                        st.text_area("Extracted text", source_text[:8192], height=260, disabled=True)
            except Exception as exc:
                datasheet_error = str(exc)
                st.error(f"Unable to read datasheet: {datasheet_error}")

    with col2:
        st.write("#### Property Definitions")
        definitions_file = st.file_uploader(
            "Definitions file",
            type=["json", "txt", "aasx"],
            key="definitions",
        )

        property_definitions: Any = []
        definitions_error = None

        if definitions_file is not None:
            try:
                property_definitions = load_property_definitions(definitions_file)
                definition_items = get_property_definition_items(property_definitions)
                num_properties = count_property_definitions(property_definitions)
                st.success(f"Loaded {num_properties} property definitions.")

                with st.expander("Definition preview", expanded=False):
                    st.json(definition_items[:5])
            except Exception as exc:
                definitions_error = str(exc)
                st.error(f"Unable to read definitions file: {definitions_error}")

    st.session_state["upload_state"] = {
        "source_text": source_text,
        "datasheet_name": datasheet_name,
        "datasheet_bytes": datasheet_bytes,
        "property_definitions": property_definitions,
        "datasheet_error": datasheet_error,
        "definitions_error": definitions_error,
    }
