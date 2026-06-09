import os
import tempfile
import streamlit as st

from rag_engine import get_vector_store, get_answer

st.set_page_config(
    page_title="AI Tutor Dashboard",
    layout="centered"
)

st.title("📚 AI Tutor Dashboard")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:
        vector_store = get_vector_store(temp_path)

        st.success("PDF processed successfully!")

        user_query = st.text_input(
            "Ask a question about the PDF"
        )

        if user_query:

            with st.spinner("Thinking..."):

                answer = get_answer(
                    vector_store,
                    user_query
                )

            st.markdown("### 🤖 Answer")
            st.write(answer)

    except Exception as e:
        st.error(str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
