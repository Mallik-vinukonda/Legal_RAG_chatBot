# Sidebar UI logic will go here.

import streamlit as st
from legal.utils import get_legal_domains

def sidebar_ui(data_dir, vector_db_dir):
    st.image("https://api.dicebear.com/6.x/identicon/svg?seed=VadheelAI", width=100)
    st.title("VadheelAI")
    st.markdown("#### Your Indian Legal Assistant")
    st.markdown("---")
    st.subheader("📄 Document Analysis")
    uploaded_files = st.file_uploader(
        "Upload legal documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload contracts, judgments, or any legal document for instant analysis and Q&A"
    )
    process_col1, process_col2 = st.columns(2)
    selected_domain = st.selectbox("Select a legal domain", get_legal_domains())
    return uploaded_files, process_col1, process_col2, selected_domain
