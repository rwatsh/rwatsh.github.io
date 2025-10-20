---
layout: post
title: Langchain with Ollama Cloud Models
date: 2025-10-19 10:41 -0500
author: rwatsh
tags: [langchain, ollama, llm]
categories: [Langchain,Ollama]
---

Ollama provides a way to run large language models (LLMs) locally or in the cloud. It supports various models including open-source ones. Langchain, a popular framework for building applications with LLMs, can be integrated with Ollama to leverage these models.

Below is a simple example of how to use Langchain with Ollama cloud models. 
Pre-requisites:
1. Install Ollama and set up your cloud instance. Follow the instructions on the [Ollama website](https://ollama.com/docs/cloud/getting-started).
2. Install the Langchain community package that includes the Ollama integration:


```bash
pip install langchain-community langchain-ollama

ollama signin
ollama run gpt-oss:120b-cloud
```


```python
from langchain_ollama import OllamaLLM 
# Replace with your cloud IP or DNS name
ollama_url = "http://localhost:11434"

llm = OllamaLLM(
    base_url=ollama_url,
    model="gpt-oss:120b-cloud",  # Or any model you've loaded in Ollama
)

# Now use the LLM in LangChain
response = llm.invoke("Explain the difference between CPU and GPU.")
print(response)
```