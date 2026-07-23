"""
aas-rail Streamlit Web UI - Main Application

Modular multi-tab application for:
1. Upload: Datasheet and property definitions
2. Inferencing: AAS information extraction with optional retrieval and ICL
3. ICL Database: In-context learning database preparation
"""

import streamlit as st

from tabs.icl_database import render_icl_database_tab
from tabs.icl_query import render_icl_query_tab
from tabs.inferencing import render_inferencing_tab
from tabs.upload import render_upload_tab
from utils import load_cached_inference_result



def init_session_state():
    """Initialize session state variables."""
    if "upload_state" not in st.session_state:
        st.session_state["upload_state"] = {}
    if "last_inference_record" not in st.session_state:
        cached_record = load_cached_inference_result()
        st.session_state["last_inference_record"] = cached_record
        st.session_state["last_inference_result"] = cached_record["result"] if cached_record else None
    if "icl_turtle_results" not in st.session_state:
        st.session_state["icl_turtle_results"] = []
    if "icl_query_rows" not in st.session_state:
        st.session_state["icl_query_rows"] = []


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="aas-rail",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    init_session_state()

    # Header
    st.title("aas-rail")
    st.markdown(
        """
    **Asset Administration Shell Retrieval-Augmented In-context Learning for Information Extraction**

    A Streamlit application for:
    - **Upload**: Datasheet and property definition files
    - **Inferencing**: Run schema-based information extraction on datasheets
    - **ICL Database**: Prepare in-context learning databases from AASX files
    """
    )

    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(["Upload", "Inferencing", "ICL Database", "ICL Query"])

    with tab1:
        render_upload_tab()

    with tab2:
        render_inferencing_tab()

    with tab3:
        render_icl_database_tab()

    with tab4:
        render_icl_query_tab()

    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
    **About this application:**
    
    aas-rail integrates schema-guided information extraction with retrieval-augmented
    in-context learning for Asset Administration Shell data.
    
    - **Upload Tab**: Prepare your datasheet and property definitions
    - **Inferencing Tab**: Extract values from datasheets using property schemas
    - **ICL Tab**: Build knowledge graphs from AASX files for improved context
    """
    )


if __name__ == "__main__":
    main()
