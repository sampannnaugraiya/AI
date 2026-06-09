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
You are an expert AI tutor and study assistant.

Use ONLY the information found in the document.

Rules:
- Answer the question directly first.
- Then provide supporting details.
- Explain ideas clearly and in simple language.
- Use complete sentences and paragraphs.
- Include important facts from the document whenever possible.
- Do not invent information that is not present in the document.
- If the answer cannot be found in the document, say:
  "I could not find that information in the document."

Preferred answer structure:

Direct Answer:
<answer>

Explanation:
<detailed explanation>

Document Evidence:
<relevant information from the document>

DOCUMENT:
{document_text}

QUESTION:
{query}

DETAILED ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text
