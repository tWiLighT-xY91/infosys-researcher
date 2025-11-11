# utils/gemini_client.py
import os
import json
from dotenv import load_dotenv
load_dotenv()

# google-generativeai usage (must be authenticated via GOOGLE_API env or service account)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# If using a service account file, ensure GOOGLE_APPLICATION_CREDENTIALS is set
# or use genai.configure(api_key=...) if using API Key (whichever your setup supports).

if genai is None:
    raise ImportError("google-generativeai not installed. pip install google-generativeai")

# configure: this will use GOOGLE_API_KEY or service account env variable if set by google library
# Optionally: genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = open("prompts/research_prompt.txt").read()

def gemini_synthesize(date: str, query: str, search_results_text: str, max_output_tokens: int = 1024):
    """
    Call Gemini (Generative AI) to synthesize a structured JSON research report.
    Returns a dict (parsed JSON) or a dict with 'text' if unable to parse.
    """
    # Compose the prompt
    system = SYSTEM_PROMPT
    user_message = f"Date: {date}\nUser Query: {query}\n\nSearch results:\n{search_results_text}\n\nProduce a JSON report as specified."

    # Example: using text-bison or a Gemini-compatible model name; change to the model you have access to
    model = "models/text-bison-001"  # adjust to your available model

    # Call the model
    resp = genai.generate_text(
        model=model,
        input=f"{system}\n\n{user_message}",
        temperature=0.2,
        max_output_tokens=max_output_tokens
    )
    text_out = resp.text or resp.candidates[0].content

    # Try parse JSON out of text if the model returns JSON
    try:
        parsed = json.loads(text_out)
        return parsed
    except Exception:
        # fallback — return text in a structured envelope
        return {"query": query, "date": date, "text": text_out}
