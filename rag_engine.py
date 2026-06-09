import os
import google.generativeai as genai
from pypdf import PdfReader

# Configure the official, native Google client tool
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

def get_vector_store(pdf_path):
    """Extracts all text directly from the PDF file chunks."""
    reader = PdfReader(pdf_path)
    text_chunks = []
    
    # Read the text page by page cleanly
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_chunks.append(page_text)
            
    # Return the raw text pages as our simple data store
    return text_chunks

def get_answer(vector_store, query):
    """Passes the entire document context directly to Gemini's massive context window."""
    # Combine the pages into a single block of context text
    context = "\n\n".join(vector_store)
    
    # Call the native model directly without any endpoint confusion
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"Use the following document text to answer the question accurately.\n\nDocument Text:\n{context}\n\nQuestion: {query}"
    
    response = model.generate_content(prompt)
    return response.text
