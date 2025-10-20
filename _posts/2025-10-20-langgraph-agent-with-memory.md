---
layout: post
title: LangGraph Agent with Memory
date: 2025-10-20 01:50 -0500
author: rwatsh
tags: [langgraph, langchain, agent, ollama, memory]
categories: [Langgraph, Langchain]
---

Example of a LangGraph agent with memory that allows iterative refinement of LLM responses based on user feedback. The agent uses Ollama LLM and LangGraph's InMemorySaver for state persistence across interactions.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph = StateGraph(State)



# LangSmith env setup
#langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

# Initialize Ollama LLM
llm = OllamaLLM(
    base_url="http://localhost:11434",
    model="gpt-oss:120b-cloud"
)

# -------- Nodes  -------- #
def FirstResponseNode(state: State):
    """Node1: Get raw LLM reply"""
    response = llm.invoke(state["messages"])
    print("\n[Raw Node1 Response]:\n", response, "\n")
    return {"messages": [response]}


def RefineNode(state: State):
    """Node2a: Iterative refinement until user is satisfied"""
    last_ai_msg = state["messages"][-1]

    while True:
        refine_instr = input("Refine Node1 response (enter instructions, or press Enter if satisfied): ")

        if not refine_instr.strip():
            print("\n[Refinement Complete] Carrying forward final response...\n")
            return {"messages": [last_ai_msg]}  # exit loop with final message

        # Otherwise, do refinement
        user_msg = {"role": "user", "content": f"{refine_instr}\n\n{last_ai_msg.content}"}
        refined = llm.invoke([user_msg])
        print("\n[Refined Response]:\n", refined, "\n")

        # Update the latest response so next loop starts from here
        last_ai_msg = refined


def TransformNode(state: State):
    """Node2b: Transform (summarize, bulletize, translate, etc.)"""
    last_ai_msg = state["messages"][-1]
    # Show the final refined response before transformation
    print("\n[Final Refined Response to Transform]:\n", last_ai_msg.content, "\n")
    transform_instr = input("Transform response (summarize...etc.): ")
    user_msg = {"role": "user", "content": f"{transform_instr}\n\n{last_ai_msg.content}"}
    transformed = llm.invoke([user_msg])
    print("\n[Transformed Response]:\n", transformed, "\n")
    return {"messages": [transformed]}


def ExtractKeywordsNode(state: State):
    """Node3: Extract 4 keywords into JSON"""
    last_ai_msg = state["messages"][-1]
    user_msg = {
        "role": "user",
        "content": f"Extract 4 important keywords from this text and return as JSON aclearray:\n\n{last_ai_msg.content}"
    }
    keywords = llm.invoke([user_msg])
    print("\n[Extracted Keywords JSON]:\n", keywords, "\n")
    return {"messages": [keywords]}

# -------- Graph -------- #
graph.add_node("node1_response", FirstResponseNode)
graph.add_node("node2_refine", RefineNode)
graph.add_node("node2_transform", TransformNode)
graph.add_node("node3_keywords", ExtractKeywordsNode)

# -------- Edges -------- #
graph.add_edge(START, "node1_response")

# Node1 always flows to refinement (but refinement internally may loop)
graph.add_edge("node1_response", "node2_refine")

# Once refinement is done → go to transform
graph.add_edge("node2_refine", "node2_transform")

# Then → keywords → END
graph.add_edge("node2_transform", "node3_keywords")
graph.add_edge("node3_keywords", END)


# Attach memory
memory = InMemorySaver()
build = graph.compile(checkpointer=memory)

# ---------- Outer Loop (traceable) ----------
def run_workflow(user_input: str, thread_id: str):
    """Wraps one complete graph execution in a LangSmith trace"""
    result = build.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result

# ---------- REPL Loop ----------
thread_id = "demo-session"
while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")

        # 🔥 Show graph when quitting
        print("\n[Graph Execution Flow]\n")
        print(build.get_graph().draw_ascii())
        break

    # 🔥 One trace per user input → full workflow
    result = run_workflow(user_input, thread_id)

    # ✅ Clean terminal output: show only last AI response
    if "messages" in result and result["messages"]:
        print("\nAI:", result["messages"][-1].content, "\n")
```

### Output

```bash
User: When will humans be able to meet ETs?

[Raw Node1 Response]:
 The short answer is **we don’t know**—there’s no scientifically grounded date we can point to for when (or even if) humanity will have a face‑to‑face meeting with an extraterrestrial civilization.  What we can do, however, is outline the key factors that drive the odds and the timelines that researchers and futurists often discuss.

---

Refine Node1 response (enter instructions, or press Enter if satisfied): Be more specific with timeline

[Refined Response]:
 Below is a **more granular, date‑based timeline** that builds on the categories I outlined earlier.
I have broken each “meeting” scenario into three confidence bands—**optimistic, median, and conservative**—and anchored them to concrete projects, instrumentation milestones, and the physics of signal/propulsion travel.
All dates are **relative to 2025** (the year of this answer).

---



Refine Node1 response (enter instructions, or press Enter if satisfied):

[Refinement Complete] Carrying forward final response...


[Final Refined Response to Transform]:
 Below is a **more granular, date‑based timeline** that builds on the categories I outlined earlier.

Transform response (summarize...etc.): Summarize in 4 bullet points each with not more than 10 words.

[Transformed Response]:
 - Passive technosignature detection likely 2026‑2045.
- Two‑way communication earliest 2040‑2050, limited by light travel.
- Interstellar probes (alien or human) possible 2030‑2070 via Starshot/LSST.
- Manned interstellar visitation >2100, needs breakthrough propulsion.


[Extracted Keywords JSON]:
[
  "Passive technosignature detection",
  "Two-way communication",
  "Interstellar probes",
  "Manned interstellar visitation"
]

AI:
[
  "Passive technosignature detection",
  "Two-way communication",
  "Interstellar probes",
  "Manned interstellar visitation"
]


User: quit
Goodbye!

[Graph Execution Flow]

   +-----------+
   | __start__ |
   +-----------+
          *
          *
          *
+----------------+
| node1_response |
+----------------+
          *
          *
          *
  +--------------+
  | node2_refine |
  +--------------+
          *
          *
          *
+-----------------+
| node2_transform |
+-----------------+
          *
          *
          *
+----------------+
| node3_keywords |
+----------------+
          *
          *
          *
    +---------+
    | __end__ |
    +---------+
```
