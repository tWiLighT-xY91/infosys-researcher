import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from datetime import datetime
import json

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TAVILY_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Please set both TAVILY_API_KEY and GEMINI_API_KEY in your .env file")

tavily = TavilyClient(api_key=TAVILY_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

def tavily_search(query, max_results=5):
    print(f"🔎 Searching Tavily for: {query}")
    results = tavily.search(query=query, max_results=max_results)
    snippets = []
    for item in results.get("results", []):
        snippets.append(f"{item['title']}: {item['content']} ({item['url']})")
    return "\n\n".join(snippets)

def research_agent(question):
    clarify_prompt = [
        SystemMessage(content="You are a helpful research assistant."),
        HumanMessage(content=f"The user asked: '{question}'. If the question lacks context, respond with a short clarification question.")
    ]
    clarify_response = llm.invoke(clarify_prompt).content.strip()

    if "?" in clarify_response:
        print(f"🗣️ Clarification needed: {clarify_response}")
        user_input = input("💬 Your answer: ")
        full_query = f"{question} ({user_input})"
    else:
        full_query = question

    search_results = tavily_search(full_query)

    report_prompt = [
        SystemMessage(content="You are an expert research analyst. Write a structured report with pros, cons, and conclusions."),
        HumanMessage(content=f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\nResearch data:\n{search_results}\n\nNow create a detailed report answering: {full_query}")
    ]

    print("🧠 Generating report using Gemini...")
    response = llm.invoke(report_prompt)
    report = response.content.strip()

    try:
        json.loads(report)
        return json.dumps(json.loads(report), indent=2)
    except json.JSONDecodeError:
        return report

if __name__ == "__main__":
    user_query = input("❓ Enter your research question: ")
    report = research_agent(user_query)
    print("\n📄 === RESEARCH REPORT ===\n")
    print(report)
