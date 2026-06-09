import os
import google.genai as genai
from google.genai import types
from pypdf import PdfReader


def get_vector_store(pdf_path):
    reader = PdfReader(pdf_path)

    text_parts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)

    return text_parts


def get_answer(vector_store, query):
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise Exception("GOOGLE_API_KEY is missing.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            api_version="v1"
        )
    )

    document_text = "\n\n".join(vector_store)

    prompt = f"""
Answer the user's question using the document below.

DOCUMENT:
{document_text}

QUESTION:
{query}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text
