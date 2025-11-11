import os
from dotenv import load_dotenv
load_dotenv()
from tavily import TavilyClient

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY missing from .env")

client = TavilyClient(api_key=TAVILY_API_KEY)

def tavily_search(query: str, max_results: int = 5):
    """
    Returns a list of dicts: [{"title":..., "snippet":..., "url":...}, ...]
    """
    resp = client.search(query=query, max_results=max_results)
    hits = []
    for r in resp.get("results", [])[:max_results]:
        hits.append({
            "title": r.get("title"),
            "snippet": (r.get("content") or "")[:600],
            "url": r.get("url")
        })
    return hits
