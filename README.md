
# OmniBrain – Task 3: Hallucination & Output Boundary Check

## Overview

This task implements a hallucination and output boundary checking mechanism for the OmniBrain document-based question answering system.

The purpose of this task is to ensure that the AI generates answers only from the information available in the retrieved document context.

If the generated answer is not sufficiently supported by the retrieved context, the system blocks the answer and returns a safe fallback response.

---

## Objective

The main objectives of this task are:

- Prevent AI hallucinations.
- Restrict LLM responses to the provided document context.
- Prevent the model from using outside knowledge.
- Detect answers that are not sufficiently supported by the retrieved context.
- Return a standardized safe fallback message when the answer cannot be supported.

---

## Project Structure

```text
omnibrain-task4/
│
├── src/
│   ├── __init__.py
│   ├── hallucination_check.py
│   └── prompt_boundary.py
│
├── tests/
│   ├── test_hallucination.py
│   └── test_prompt_boundary.py
│
├── .gitignore
├── README.md
└── requirements
