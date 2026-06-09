import os
import google.genai as genai
from google.genai import types
from pypdf import PdfReader

def get_vector_store(pdf_path):
    """
    Extract text from PDF pages.
    """
    reader = PdfReader(pdf_path)

    text_chunks = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_chunks.append(text)

    return text_chunks

def get_answer(vector_store, query):

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set."
        )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            api_version="v1"
        )
    )

    context = "\n\n".join(vector_store)

    prompt = f"""
You are an AI tutor.

Use ONLY the information contained in the document below.

DOCUMENT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text  
