import os
import tempfile

import streamlit as st

from rag_engine import (
    get_vector_store,
    get_answer,
)

st.set_page_config(
    page_title="AI Tutor Dashboard",
    layout="centered"
)

st.title("📚 AI Tutor Dashboard")
st.write("Upload a PDF and ask questions about it.")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    try:
        with st.spinner("Processing PDF..."):

            st.session_state.vector_store = get_vector_store(
                temp_path
            )

        st.success("PDF processed successfully!")

    except Exception as e:
        st.error(f"PDF processing failed: {e}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if st.session_state.vector_store:

    st.divider()

    user_query = st.text_input(
        "Ask a question about your document"
    )

    if user_query:

        try:
            with st.spinner("Generating answer..."):

                answer = get_answer(
                    st.session_state.vector_store,
                    user_query
                )

            st.markdown("### 🤖 Answer")
            st.write(answer)

        except Exception as e:
            st.error(f"Error: {e}")
