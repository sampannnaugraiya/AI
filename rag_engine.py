import os
from docling.document_converter import DocumentConverter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NOTE: We removed the hardcoded key line completely. 
# LangChain will automatically find the GOOGLE_API_KEY secret 
# that Streamlit Cloud handles behind the scenes!

def get_vector_store(pdf_path):
    # This runs the parsing and database creation
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    text = result.document.export_to_markdown()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_texts(splits, embeddings)

def get_answer(vector_store, query):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    docs = vector_store.similarity_search(query, k=8)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"Analyze the text: {context}\n\nQuestion: {query}"
    return llm.invoke(prompt).content