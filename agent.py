from langgraph.graph import StateGraph, END, START
from shared_store import url_time
import time
from langgraph.prebuilt import ToolNode
from tools import (
    get_rendered_html,
    download_file,
    post_request,
    run_code,
    add_dependencies,
    ocr_image_tool,
    transcribe_audio,
    encode_image_to_base64,
)
from typing import TypedDict, Annotated, List
from langchain_core.messages import trim_messages, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

MAX_TOKENS = 3500
RECURSION_LIMIT = 120


# ---------------- STATE ----------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


TOOLS = [
    run_code,
    get_rendered_html,
    download_file,
    post_request,
    add_dependencies,
    ocr_image_tool,
    transcribe_audio,
    encode_image_to_base64,
]


# ---------------- LLM ----------------
llm = init_chat_model(
    model_provider="huggingface",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0,
    max_tokens=512,
).bind_tools(TOOLS)


# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Rules:
- Read the page carefully.
- Compute the correct answer.
- NEVER explain.
- NEVER use markdown.
- ALWAYS submit answers using the post_request tool.
- If unsure, submit "SKIP".

When submitting:
- Use the exact endpoint shown on the page
- Payload must contain: answer, email, secret

Always include:
email = {EMAIL}
secret = {SECRET}
"""


# ---------------- AGENT NODE ----------------
def agent_node(state: AgentState):
    cur_time = time.time()
    cur_url = os.getenv("url")

    # timeout protection
    prev_time = url_time.get(cur_url)
    if prev_time and (cur_time - float(prev_time)) >= 180:
        print("⏱️ Timeout exceeded, skipping")
        return {"messages": []}

    trimmed = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=llm,
    )

    if not any(msg.type == "human" for msg in trimmed):
        trimmed.append(
            HumanMessage(
                content=f"Continue solving URL: {os.getenv('url', 'UNKNOWN')}"
            )
        )

    print(f"--- INVOKING AGENT ({len(trimmed)} messages) ---")
    result = llm.invoke(trimmed)
    return {"messages": [result]}


# ---------------- ROUTING ----------------
def route(state):
    last = state["messages"][-1]

    # If tool call exists → execute tool
    if getattr(last, "tool_calls", None):
        print("Route → tools")
        return "tools"

    # If server returned a next URL → continue agent loop
    content = getattr(last, "content", None)
    if isinstance(content, dict) and content.get("url"):
        print("Route → agent (next URL)")
        return "agent"

    print("Route → END")
    return END


# ---------------- GRAPH ----------------
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")

graph.add_conditional_edges(
    "agent",
    route,
    {
        "tools": "tools",
        "agent": "agent",
        END: END,
    },
)

app = graph.compile()


# ---------------- RUNNER ----------------
def run_agent(url: str):
    url_time[url] = time.time()
    os.environ["url"] = url

    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url},
    ]

    app.invoke(
        {"messages": initial_messages},
        config={"recursion_limit": RECURSION_LIMIT},
    )

    print("✅ Agent run completed")
