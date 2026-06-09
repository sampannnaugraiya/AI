import streamlit as st
import os
from rag_engine import get_vector_store, get_answer

# Set up clean page configuration
st.set_page_config(page_title="AI Tutor Dashboard", layout="centered")
st.title("📚 AI Tutor Dashboard")
st.write("Upload a PDF and ask questions directly to Gemini without any server lag.")

# 1. File Upload Box
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily to pass to our engine
    temp_filename = "temp_uploaded_doc.pdf"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process the PDF using native Python text extraction
    if "vector_store" not in st.session_state:
        with st.spinner("Processing PDF cleanly... Hang tight!"):
            try:
                st.session_state.vector_store = get_vector_store(temp_filename)
                st.success("PDF processed successfully!")
            except Exception as e:
                st.error(f"Failed to process document: {e}")
                
    # Clean up the temporary file from the disk after loading
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    # 2. Chat Interface (Only displays if the text data is ready in memory)
    if "vector_store" in st.session_state:
        st.write("---")
        user_query = st.text_input("Ask a question about your document:")
        
        if user_query:
            with st.spinner("Gemini is thinking..."):
                try:
                    # Pass the text list and question straight to Google
                    answer = get_answer(st.session_state.vector_store, user_query)
                    st.markdown(f"### 🤖 Answer:\n{answer}")
                except Exception as e:
                    st.error(f"Error fetching answer: {e}")
