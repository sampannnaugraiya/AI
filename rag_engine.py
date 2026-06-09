from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_vector_store(pdf_path):
    # Pure Python PDF reader
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    
    # Pure Python memory storage. No databases, 100% crash-proof.
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    return InMemoryVectorStore.from_documents(splits, embeddings)
def get_answer(vector_store, query):
    # Added '-latest' to force LangChain to skip the broken v1beta endpoint mapping
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    docs = vector_store.similarity_search(query, k=8)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"Analyze the text: {context}\n\nQuestion: {query}"
    return llm.invoke(prompt).content
