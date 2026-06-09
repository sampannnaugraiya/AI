from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_vector_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    
    # Using the native 'v1:' prefix layout instead of client_options dicts
    embeddings = GoogleGenerativeAIEmbeddings(model="v1:text-embedding-004")
    return InMemoryVectorStore.from_documents(splits, embeddings)

def get_answer(vector_store, query):
    # This explicit string tells LangChain exactly how to route to production safely
    llm = ChatGoogleGenerativeAI(model="v1:gemini-1.5-flash")
    
    docs = vector_store.similarity_search(query, k=8)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"Analyze the text: {context}\n\nQuestion: {query}"
    return llm.invoke(prompt).content
