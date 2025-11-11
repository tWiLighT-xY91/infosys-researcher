# main.py
import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_react_agent  # (works in 0.1.x)
from langchain_community.llms import HuggingFacePipeline
from langchain_community.tools import Tool
from transformers import pipeline

from utils.date_utils import get_current_date
from utils.search_utils import tavily_search
from utils.gemini_client import gemini_synthesize

# --- LLM setup (we will not use a heavy local LLM for Gemini step — Gemini will do final synth)
# But LangChain require an LLM object to run the agent's internal reasoning; use a small lightweight local model
# You can also set this to a dummy local LLM if you don't want local inference here.

# For safety, we will use a tiny HF model for chain-of-thought / agent reasoning if GPU available:
device = 0 if (os.environ.get("CUDA_VISIBLE_DEVICES") or False) else -1
local_pipeline = pipeline("text-generation", model="gpt2", device=device, max_new_tokens=128, temperature=0.0)
llm_local = HuggingFacePipeline(pipeline=local_pipeline)

# Define LangChain Tools
date_tool = Tool(
    name="get_current_date",
    func=lambda x=None: get_current_date(),
    description="Returns the current date as YYYY-MM-DD. No input required."
)

def _search_tool_fn(query: dict):
    # the agent will call this with a string; ensure we accept raw string input
    q = query if isinstance(query, str) else str(query.get("query", query))
    hits = tavily_search(q, max_results=5)
    # format succinctly
    return json.dumps(hits)

search_tool = Tool(
    name="tavily_search",
    func=_search_tool_fn,
    description="Search the web using Tavily. Input is a search query string. Returns JSON list of hits."
)

tools = [date_tool, search_tool]

# Prompt: instruct the agent to call date then search then return
from langchain.prompts import PromptTemplate
template = """
You are a research agent. ALWAYS call get_current_date() first to retrieve today's date.
Then, construct a search query and call tavily_search(query).
Finally, aggregate results and provide a concise answer OR hand off to Gemini for final synthesis.
User query: {input}
{agent_scratchpad}
"""
prompt = PromptTemplate.from_template(template)

agent = create_react_agent(llm_local, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_agent(user_query: str):
    # 1) Run the agent (this will call the date and the search tool)
    result = agent_executor.invoke({"input": user_query})
    # agent returns string output; agent_executor also logs tool calls in its trace; we assume tavily_search returned JSON string
    # Extract the date by calling get_current_date() directly (reliable)
    date = get_current_date()
    # Build search queries and gather evidence (you may parse tool outputs from agent trace instead)
    # For simplicity, run a final set of searches here:
    search_queries = [
        f"{user_query} {date} reviews",
        f"best {user_query} near me",
        f"{user_query} recommendations 2025"
    ]
    evidence_blocks = []
    for q in search_queries:
        hits = tavily_search(q, max_results=4)
        evidence_blocks.append({"query": q, "hits": hits})

    # Format evidence for Gemini prompt
    search_text = ""
    for b in evidence_blocks:
        search_text += f"=== {b['query']} ===\n"
        for h in b['hits']:
            search_text += f"- {h.get('title','')}\n  {h.get('snippet','')}\n  {h.get('url')}\n"

    print("Calling Gemini for synthesis...")
    report = gemini_synthesize(date=date, query=user_query, search_results_text=search_text)
    # Save & print
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/last_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    q = input("Ask a research question: ").strip()
    run_agent(q)
