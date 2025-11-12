import os, json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


SYSTEM_PROMPT = """
You are a rigorous research assistant. Using the search results below,
write a well-structured JSON report with these fields:
- query
- date
- summary (overview)
- top_picks: list of {name, description, rank, pros, cons, best_for, confidence, sources}
- method (how you derived results)
- confidence_overall (0.0 - 1.0)
- notes (uncertain or conflicting data)
Return only valid JSON.
"""

def gemini_synthesize(date, query, search_results_text):
    """
    Send research evidence to Gemini and get a JSON report back.
    """
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"{SYSTEM_PROMPT}\n\nDate: {date}\nQuery: {query}\n\nSearch results:\n{search_results_text}"
    response = model.generate_content(prompt)

   
    text_out = response.text.strip()
    try:
        parsed = json.loads(text_out)
        return parsed
    except json.JSONDecodeError:
        return {"query": query, "date": date, "text": text_out}
