import time

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import OpenAI

from src.evaluation.metrics import (
    calculate_latency,
    calculate_token_usage,
    evaluate_correctness,
    evaluate_relevance,
    evaluate_hallucination,
)

load_dotenv()

# Initialize Langfuse
langfuse = get_client()

# Ollama through Langfuse's OpenAI-compatible wrapper
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


def main():

    question = "What does RAG mean in AI?"

    print("Sending request to Ollama...")

    # Create an active Langfuse observation
    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-evaluation",
        input=question,
    ) as span:

        # -----------------------------
        # Start timing
        # -----------------------------

        start_time = time.time()

        # -----------------------------
        # LLM request
        # -----------------------------

        response = client.chat.completions.create(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant. "
                        "Answer technical questions accurately."
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )

        # -----------------------------
        # Calculate latency
        # -----------------------------

        latency = calculate_latency(start_time)

        # -----------------------------
        # Extract response
        # -----------------------------

        answer = response.choices[0].message.content

        # -----------------------------
        # Calculate token usage
        # -----------------------------

        token_usage = calculate_token_usage(response)

        # -----------------------------
        # Evaluation: Correctness
        # -----------------------------

        correctness = evaluate_correctness(
            answer,
            "retrieval-augmented generation",
        )

        # -----------------------------
        # Evaluation: Relevance
        # -----------------------------

        relevance = evaluate_relevance(
            answer,
            [
                "RAG",
                "retrieval",
                "generation",
            ],
        )

        # -----------------------------
        # Evaluation: Hallucination
        # -----------------------------

        hallucination = evaluate_hallucination(
             answer,
            "Retrieval-Augmented Generation",
)

        # -----------------------------
        # Store metrics in Langfuse
        # -----------------------------

        span.update(
            output=answer,
            metadata={
                "latency_seconds": latency,
                "input_tokens": token_usage["input_tokens"],
                "output_tokens": token_usage["output_tokens"],
                "total_tokens": token_usage["total_tokens"],
                "correctness": correctness["score"],
                "relevance": relevance["score"],
                "hallucination": hallucination["score"],
            },
        )

        # -----------------------------
        # Send correctness score
        # -----------------------------

        span.score_trace(
            name="response_correctness",
            value=float(correctness["score"]),
            data_type="NUMERIC",
            comment=correctness["label"],
        )

        # -----------------------------
        # Send relevance score
        # -----------------------------

        span.score_trace(
            name="response_relevance",
            value=float(relevance["score"]),
            data_type="NUMERIC",
            comment=(
                f"Matched "
                f"{relevance['matched_keywords']}/"
                f"{relevance['total_keywords']} keywords"
            ),
        )

        # -----------------------------
        # Send hallucination score
        # -----------------------------

        span.score_trace(
            name="hallucination_check",
            value=float(hallucination["score"]),
            data_type="NUMERIC",
            comment=hallucination["label"],
        )

    # The with-block automatically ends the span.
    # Do NOT call span.end() here.

    # Make sure all Langfuse data is uploaded
    langfuse.flush()

    # -----------------------------
    # Console output
    # -----------------------------

    print("\n========== RESPONSE ==========")
    print(answer)

    print("\n========== METRICS ==========")

    print(f"Latency: {latency} seconds")
    print(f"Input tokens: {token_usage['input_tokens']}")
    print(f"Output tokens: {token_usage['output_tokens']}")
    print(f"Total tokens: {token_usage['total_tokens']}")

    print("\n========== EVALUATION ==========")

    print(
        f"Correctness: {correctness['score']} "
        f"({correctness['label']})"
    )

    print(
        f"Relevance: {relevance['score']} "
        f"({relevance['matched_keywords']}/"
        f"{relevance['total_keywords']} keywords)"
    )

    print(
        f"Hallucination: {hallucination['score']} "
        f"({hallucination['label']})"
    )

    if hallucination["detected_claims"]:
        print(
            "Detected claims:",
            hallucination["detected_claims"],
        )

    print("\n✅ Evaluation data sent to Langfuse.")


if __name__ == "__main__":
    main()