import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NOTE: No hardcoded keys. Streamlit will find GOOGLE_API_KEY in Secrets!

def get_vector_store(pdf_path):
    # Pure Python PDF text reader. No graphics, no cv2, no errors.
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Split the documents into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    
    # Generate embeddings and save to the vector database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(splits, embeddings)

def get_answer(vector_store, query):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    docs = vector_store.similarity_search(query, k=8)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"Analyze the text: {context}\n\nQuestion: {query}"
    return llm.invoke(prompt).content
