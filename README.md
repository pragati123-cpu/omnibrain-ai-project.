# OmniBrain - Evaluation Metrics & Langfuse Integration

## Overview

This project implements evaluation metrics and observability for an
LLM application using Ollama and Langfuse.

The project measures:

- LLM execution latency
- Input token usage
- Output token usage
- Total token usage
- Response correctness
- Response relevance
- Possible hallucinations
- LLM execution traces through Langfuse

The project is completely independent from the previous OmniBrain tasks.

---

## Architecture

```text
User Question
      |
      v
Ollama (Llama 3.2:3b)
      |
      v
Langfuse OpenAI Wrapper
      |
      v
LLM Response
      |
      +-------------------+
      |                   |
      v                   v
Metrics Calculation    Langfuse Trace
      |                   |
      +---------+---------+
                |
                v
          Evaluation
                |
       +--------+--------+
       |        |        |
       v        v        v
  Correctness Relevance Hallucination
       |        |        |
       +--------+--------+
                |
                v
        Langfuse Scores