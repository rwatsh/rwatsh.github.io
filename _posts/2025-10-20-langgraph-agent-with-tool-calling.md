---
layout: post
title: LangGraph Agent with Tool Calling
date: 2025-10-20 08:15 -0500
author: rwatsh
tags: [langgraph, langchain, agent, ollama, tools]
categories: [Langgraph, Langchain]
---

# LangGraph Agent with Tool Calling Example
This example demonstrates a LangGraph agent that can intelligently decide when to call external tools during its reasoning process. The agent uses Groq to access meta Llama3 LLM and three tools: Google Knowledge Graph Search, Organic Google Search, and Firecrawl for web scraping.

Code is given below:

```python
import os
from typing import Literal

from dotenv import load_dotenv
from firecrawl import Firecrawl  # Firecrawl SDK
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from serpapi import GoogleSearch
from langsmith import traceable

# ----------------- ENV -----------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_base_url = os.getenv("GROQ_BASE_URL")
serp_api_key = os.getenv("SERP_API_KEY")
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
# LangSmith env setup
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")
if not groq_base_url:
    raise ValueError("GROQ_BASE_URL not found in .env file")
if not serp_api_key:
    raise ValueError("SERP_API_KEY not found in .env file")
if not firecrawl_api_key:
    raise ValueError("FIRECRAWL_API_KEY not found in .env file")

# ----------------- LLM -----------------
llm = ChatOpenAI(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=groq_api_key,
    base_url=groq_base_url,
    temperature=0.5,
)

# ----------------- Firecrawl -----------------
firecrawl = Firecrawl(api_key=firecrawl_api_key)

# ----------------- TOOLS -----------------
@tool
def knowledge_graph_search(query: str) -> dict:
    """Extract specific information from Google's Knowledge Graph for reliable facts."""
    params = {"engine": "google", "q": query, "api_key": serp_api_key}
    search = GoogleSearch(params)
    results = search.get_dict()
    knowledge_graph = results.get("knowledge_graph", None)
    if knowledge_graph:
        return {"knowledge_graph": knowledge_graph}
    else:
        return {"error": "No Knowledge Graph data found."}

@tool
def organic_google_search(query: str) -> dict:
    """Perform a general organic Google search. Returns top 6 results if found."""
    params = {"engine": "google", "q": query, "api_key": serp_api_key}
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", None)

    if organic_results:
        return {"results": organic_results[:6]}
    else:
        return {"error": "No general search results found."}

@tool
def firecrawl_scrape(url: str) -> dict:
    """Scrape webpage content into a concise summary using Firecrawl."""
    try:
        doc = firecrawl.scrape(url, formats=["summary"])
        return {
            "url": url,
            "summary": getattr(doc, "summary", None),
            "metadata": getattr(doc, "metadata", None),
        }
    except Exception as e:
        return {"error": str(e)}

# Register tools
tools = [knowledge_graph_search, organic_google_search, firecrawl_scrape]
tools_by_name = {t.name: t for t in tools}

# ----------------- LLM WITH TOOLS -----------------
llm_with_tools = llm.bind_tools(tools)

# ----------------- GRAPH NODES -----------------
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or stop."""
    messages = [SystemMessage(content="""You are a research assistant.

Workflow:
1. Always attempt `knowledge_graph_search` first.  
2. If knowledge graph content is available, use it directly.  
3. If not available or an error occurs, perform `organic_google_search`.  
4. From the search results, carefully select the **most relevant** URL (not just the first one).  
5. Call `firecrawl_scrape` on the chosen URL to extract content.  

Use this workflow strictly in order. Do not skip steps unless explicitly instructed.
""")] + state["messages"]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: dict):
    """Executes tool calls made by the LLM."""
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": results}

def should_continue(state: MessagesState) -> Literal["Action", END]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "Action"
    return END

# ----------------- GRAPH -----------------
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, {"Action": "environment", END: END})
agent_builder.add_edge("environment", "llm_call")

# Attach memory
memory = InMemorySaver()
agent = agent_builder.compile(checkpointer=memory)


# Choose a thread id for this interactive run
THREAD_ID = "chat-session"

# ----------------- INTERACTIVE LOOP -----------------
@traceable(name="OuterWorkflowRun1")
def chat_loop():
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # Build state
        state = {"messages": [HumanMessage(content=user_input)]}

        # 🔑 Invoke with thread_id so memory sticks
        result = agent.invoke(
            state,
            config={"configurable": {"thread_id": THREAD_ID}},
        )

        # ✅ Clean terminal output: only final AI response
        final_response = result["messages"][-1].content
        print("\nAI:", final_response, "\n")


if __name__ == "__main__":
    print("🤖 Type your research query (or 'quit' to exit).")
    chat_loop()
```

### Testing

To test the LangGraph agent with tool calling, run the script and input a research query when prompted. The agent will decide whether to call the appropriate tools based on the workflow defined in the system message. You can exit the loop by typing 'quit', 'exit', or 'q'.

```bash
User: Who is Donald Trump?

AI: Donald John Trump is an American politician, media personality, and businessman who served as the 45th president of the United States from 2017 to 2021 and is the 47th president. He was born on June 14, 1946, in Jamaica Hospital Medical Center, New York, NY. He is a member of the Republican Party. His parents are Fred Trump and Mary Anne MacLeod Trump. He has been married three times to Ivana Trump, Marla Maples, and Melania Trump, with whom he has seven children: Ivanka Trump, Donald Trump Jr., Eric Trump, Tiffany Trump, Barron Trump, and others. He attended the Wharton School of the University of Pennsylvania. 


User: Who is Watsh Rajneesh?

AI: The search results did not provide information on a person named Watsh Rajneesh. However, they did provide information on Rajneesh, a spiritual leader and the founder of the Rajneesh movement. 


User: Tell me about Watsh Rajneesh?

AI: I couldn't find any information on a person named Watsh Rajneesh. It is possible that Watsh Rajneesh is a private individual and not a public figure, or that the name is misspelled or not well-known. Can you provide more context or details about who Watsh Rajneesh is or what they are known for? This will help me provide a more accurate response. 


User: Watsh Rajneesh is my name and i am an individual

AI: I'm glad to hear that I can help you with information about yourself.

As Watsh Rajneesh, you seem to have a LinkedIn profile with over 860 followers, and you've completed your MS in Software Engineering with a GPA of 3.418. Your project sources are available on GitHub.

If you'd like to share more about yourself or ask a specific question, I'm here to help. Otherwise, I can simply confirm that I've found some online presence associated with your name. 


User: Tell me about the certifications that i have acquired so far.

AI: Based on the search results, I found that you, Watsh Rajneesh, have an Oracle Cloud Infrastructure Certified Architect Associate certification. You also have experience working at Apple and have completed your MS in Software Engineering with a GPA of 3.418. Your project sources are available on GitHub.

If you're looking for more information on your certifications or want to share more about your experience, feel free to ask! 


User: quit
Goodbye!
```