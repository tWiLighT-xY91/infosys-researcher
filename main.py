import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage


load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TAVILY_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Please set both TAVILY_API_KEY and GEMINI_API_KEY in your .env file")


tavily = TavilyClient(api_key=TAVILY_API_KEY)
gemini = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)


def tavily_search(query, max_results=5):
    print(f"🔎 Searching Tavily for: {query}")
    results = tavily.search(query=query, max_results=max_results)
    snippets = []
    for item in results.get("results", []):
        snippets.append(f"{item['title']}: {item['content']} ({item['url']})")
    return "\n\n".join(snippets)


def research_agent(question):
    """
    Performs a full deep-research workflow:
    1. Clarifies incomplete queries.
    2. Performs Tavily web search.
    3. Synthesizes a Gemini report.
    """
    # Step 1: Clarify topic
    clarify_prompt = [
        SystemMessage(content="You are a helpful research assistant."),
        HumanMessage(content=f"The user asked: '{question}'. If the question lacks context, respond with a short clarification question.")
    ]
    clarify_response = gemini.invoke(clarify_prompt).content.strip()

    if "?" in clarify_response:
        print(f"🗣️ Clarification needed: {clarify_response}")
        user_input = input("💬 Your answer: ")
        full_query = f"{question} ({user_input})"
    else:
        full_query = question

    # Step 2: Performing Tavily Search
    search_results = tavily_search(full_query)

    # Step 3: report prep
    report_prompt = [
        SystemMessage(content="You are an expert research analyst. Write a structured report with pros, cons, and conclusions."),
        HumanMessage(content=f"Research data:\n{search_results}\n\nNow create a detailed report answering: {full_query}")
    ]

    print("🧠 Generating report using Gemini...")
    response = gemini.invoke(report_prompt)
    report = response.content.strip()
    return report

if __name__ == "__main__":
    user_query = input("❓ Enter your research question: ")
    report = research_agent(user_query)
    print("\n📄 === RESEARCH REPORT ===\n")
    print(report)
