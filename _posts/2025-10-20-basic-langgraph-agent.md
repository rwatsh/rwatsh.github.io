---
layout: post
title: Basic LangGraph agent
date: 2025-10-20 00:09 -0500
author: rwatsh
tags: [langgraph, langchain, agent, ollama]
categories: [Langgraph,Langchain]
---

# LangGraph Basic Agent Example

LangGraph is a new orchestration framework from LangChain for building, managing, and deploying long-running, stateful agents. It is part of the LangChain stack which includes:

- Orchestration - LangGraph
- Integrations - LangChain
- Evals + Observability - LangSmith
- Deployment - LangGraph Platform

Each stack component/product can be used independently of other components shown in the stack above. So we could replace LangChain with DSPy for e.g. and use LangGraph for orchestration of agents. For Evals + Observability we could use LangFuse or MLFlow instead of LangSmith and we could deploy our agents in other agent hosting platforms instead of LangGraph platform.
Other agent frameworks - CrewAI, MetaGPT and AutoGen.

At its core, LangGraph uses the power of graph-based architectures to model and manage the intricate relationships between various components of an AI agent workflow. LangGraph is a low-level orchestration framework for building, managing, and deploying long-running, stateful agents.
https://github.com/langchain-ai/langgraph 


## Features of LangGraph framework

1. Graph-based workflows
    * Nodes - Represent discrete units of work, which can be any function, a large language model (LLM), or a tool call.
    * Edges - Define the transitions between nodes, controlling the flow of the application. They can be simple, fixed paths or conditional, with the next step determined by the current state.
    * Cycles - A key feature that allows workflows to have loops. This is crucial for agents that need to reflect on their output, retry a failed action, or refine their reasoning before proceeding. 
￼
2. Advanced State and Memory management 
	* Persistent state - An explicit, user-defined state object is passed between nodes and automatically updated. This allows the graph to maintain context throughout long-running or multi-turn interactions, which is a major enhancement over earlier, often stateless, agent executors. **A shared state keeps evolving as AI workflow progresses and each Node agent as it executes will have access to the shared state store as context.**
	* Durable execution - An explicit, user-defined state object is passed between nodes and automatically updated. This allows the graph to maintain context throughout long-running or multi-turn interactions, which is a major enhancement over earlier, often stateless, agent executors.
    * Long-term memory - Supports persistent storage across multiple sessions, allowing agents to remember user preferences or past interactions over extended periods.
	* Time travel - Allows developers to inspect, rewind, and modify the application's state at any point during execution, which significantly simplifies debugging. 

3. Human-in-the-loop (HITL) workflow - Developers can add "human nodes" to the workflow where the process pauses to wait for human feedback or approval. The workflow can then resume based on the human's input. This is invaluable for sensitive tasks or those requiring expert review. 

4. First-class Streaming - Native support for streaming provides real-time visibility into an agent's reasoning process. It can stream intermediate steps and token-by-token output, which improves the user experience and helps with debugging.

5. Multi-agent coordination - The graph structure is well-suited for orchestrating collaboration between multiple specialized agents. A "supervisor" agent can route tasks to various worker agents to handle complex workloads.

6. Parallelization - Using the Send API, developers can run multiple branches of a graph concurrently to speed up the processing of independent subtasks. 
￼
## Basic Principles of LangGraph

1. LangGraph is *Typed* meaning it expects a `State` schema that must be defined using python typing
2. Each node and edge in the graph expects and returns values conforming to the schema.
3. Because it is stateful, LangGraph uses `Annotated` to add rules and metadata functions like reducers to state schemas.
4. `StateGraph`, or `Graph` defines the structure of Nodes and Edges.

```python
from typing import Annotated
from langchain_ollama import OllamaLLM
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Replace with your cloud IP or DNS name
ollama_url = "http://localhost:11434"


# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph = StateGraph(State)

llm = OllamaLLM(
    base_url=ollama_url,
    model="gpt-oss:120b-cloud",  # Or any model you've loaded in Ollama
)

# -------- Nodes -------- #

# Node 1: first LLM response to user input
def FirstResponseNode(state: State):
    response = llm.invoke(state["messages"])
    print(response)
    print("\n[Raw Node1 Response]:\n", response, "\n")  # show raw reply
    return {"messages": [response]}

# Node 2: user chooses how to transform Node1’s response
def TransformNode(state: State):
    last_ai_msg = state["messages"][-1]  # last AI message (Node1 output)

    # Ask user in terminal what to do with Node1's reply
    transform_instr = input("Transform Node1 response: ")

    # Feed instruction + Node1 output together
    user_msg = {
        "role": "user",
        "content": f"{transform_instr}\n\n{last_ai_msg.content}"
    }
    transformed = llm.invoke([user_msg])
    return {"messages": [transformed]}

# -------- Graph -------- #
graph.add_node("node1_response", FirstResponseNode)
graph.add_node("node2_transform", TransformNode)

graph.add_edge(START, "node1_response")
graph.add_edge("node1_response", "node2_transform")
graph.add_edge("node2_transform", END)

build = graph.compile()

# -------- Loop -------- #
while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # Run through graph
    result = build.invoke({"messages": [{"role": "user", "content": user_input}]})
    final_msg = result["messages"][-1]

    if final_msg.type == "human":
        print("\n[Transformed Response]:\n", final_msg.content, "\n")
```

### Output Example

```bash
User: Whatever happened to Metaverse idea by Meta?
[Some text…. ]
Transform Node1 response: Summarize this in bullet points
[Transformed Response: Summary… ]
```
