import streamlit as st
from rag_engine import get_vector_store, get_answer

st.title("AI Tutor Dashboard")
pdf = st.file_uploader("Upload your PDF", type="pdf")

# Cache the vector store so it doesn't re-run every time you ask a question
@st.cache_resource
def load_vector_store(file_path):
    return get_vector_store(file_path)

if pdf:
    # Save file
    with open("temp.pdf", "wb") as f:
        f.write(pdf.getbuffer())
    
    # Load once and store in session
    vs = load_vector_store("temp.pdf")
    
    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Analyzing..."): # Added a spinner for better UX
            response = get_answer(vs, query)
            st.write(response)