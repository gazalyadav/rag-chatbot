# tests/evaluate_rag.py
"""
RAGAS Evaluation Script for RAG Chatbot.
Measures: Faithfulness, Answer Relevancy, Context Precision, Context Recall.

RAGAS needs an LLM to judge answers. We'll use Ollama locally.
Run with: python tests/evaluate_rag.py
"""

import sys
import json
sys.path.insert(0, ".")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.rag_chain      import run_rag_query
from src.memory_manager import ConversationMemory
from src.vector_store   import vector_store_exists


# ── Evaluation Questions ───────────────────────────────────────────────────────
# These are questions whose answers ARE in your indexed document.
# ground_truth = the ideal answer (written by you, used as reference).
# Tailor these to YOUR document content.

EVAL_DATASET = [
    {
        "question": "What is the RAG chatbot project about?",
        "ground_truth": (
            "The RAG chatbot project is a domain-specific intelligent chatbot "
            "that processes custom document sets including PDF, TXT, and DOCX files, "
            "indexes them into a vector database, and uses a local LLM to generate "
            "grounded, hallucination-free answers with source citations."
        )
    },
    {
        "question": "What technologies are used in the RAG chatbot?",
        "ground_truth": (
            "The RAG chatbot uses Python, Streamlit for the frontend, "
            "LangChain for orchestration, Sentence Transformers for embeddings, "
            "FAISS or ChromaDB as the vector database, and Llama 3 or Mistral "
            "running locally via Ollama as the language model."
        )
    },
    {
        "question": "What is the purpose of the vector database in the RAG system?",
        "ground_truth": (
            "The vector database stores document embeddings and enables fast "
            "similarity search to retrieve the most relevant document chunks "
            "when a user asks a question."
        )
    },
    {
        "question": "What document formats does the RAG chatbot support?",
        "ground_truth": (
            "The RAG chatbot supports PDF, TXT, and DOCX document formats "
            "for ingestion and processing."
        )
    },
    {
        "question": "How does the RAG system prevent hallucinations?",
        "ground_truth": (
            "The RAG system prevents hallucinations by grounding every answer "
            "in retrieved document context. The LLM is instructed to only use "
            "the provided context and cite sources, refusing to answer from "
            "outside knowledge."
        )
    },
]


def collect_rag_outputs(eval_dataset: list) -> dict:
    """
    Run each evaluation question through the RAG pipeline
    and collect questions, answers, and retrieved contexts.
    """
    print("\n" + "="*60)
    print("   COLLECTING RAG OUTPUTS FOR EVALUATION")
    print("="*60)

    questions = []
    answers   = []
    contexts  = []
    grounds   = []

    for i, item in enumerate(eval_dataset):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n  [{i+1}/{len(eval_dataset)}] {question[:60]}...")

        # Fresh memory for each question (isolated evaluation)
        memory = ConversationMemory()

        try:
            # Get answer from RAG pipeline
            answer, sources = run_rag_query(
                query  = question,
                memory = memory,
                k      = 4
            )

            # Get raw chunk texts for context evaluation
            from src.retriever import retrieve_relevant_chunks
            chunks       = retrieve_relevant_chunks(question, k=4)
            context_list = [chunk.page_content for chunk in chunks]

            questions.append(question)
            answers.append(answer)
            contexts.append(context_list)
            grounds.append(ground_truth)

            print(f"  ✅ Answer: {answer[:100]}...")
            print(f"  📎 Contexts retrieved: {len(context_list)}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

    return {
        "question"    : questions,
        "answer"      : answers,
        "contexts"    : contexts,
        "ground_truth": grounds,
    }


def run_evaluation():
    """Run the full RAGAS evaluation pipeline."""

    # Guard: check vector store exists
    if not vector_store_exists():
        print("❌ No documents indexed!")
        print("   Please upload and index documents via the Streamlit UI first.")
        sys.exit(1)

    print("\n🔍 Collecting RAG outputs...")
    raw_data = collect_rag_outputs(EVAL_DATASET)

    if not raw_data["question"]:
        print("❌ No data collected — check your vector store and Ollama.")
        sys.exit(1)

    # Build HuggingFace Dataset (required by RAGAS)
    dataset = Dataset.from_dict(raw_data)

    print("\n" + "="*60)
    print("   RUNNING RAGAS EVALUATION")
    print("   (This takes 2-5 minutes — RAGAS calls LLM per metric)")
    print("="*60)

    try:
        results = evaluate(
            dataset = dataset,
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        # ── Print Results ──────────────────────────────────────────────────────
        print("\n" + "="*60)
        print("   📊 RAGAS EVALUATION RESULTS")
        print("="*60)

        scores = {
            "Faithfulness"      : round(results["faithfulness"], 4),
            "Answer Relevancy"  : round(results["answer_relevancy"], 4),
            "Context Precision" : round(results["context_precision"], 4),
            "Context Recall"    : round(results["context_recall"], 4),
        }

        for metric, score in scores.items():
            bar    = "█" * int(score * 20)
            rating = "🟢 Excellent" if score >= 0.8 else \
                     "🟡 Good"      if score >= 0.6 else \
                     "🔴 Needs Work"
            print(f"  {metric:<22} {score:.4f}  {bar:<20} {rating}")

        overall = sum(scores.values()) / len(scores)
        print(f"\n  {'Overall Score':<22} {overall:.4f}")
        print("="*60)

        # ── Interpretation ─────────────────────────────────────────────────────
        print("\n  📝 INTERPRETATION:")
        print(f"  Faithfulness {scores['Faithfulness']:.2f}    → "
              f"{'Low hallucination ✅' if scores['Faithfulness'] >= 0.8 else 'Some hallucination ⚠️'}")
        print(f"  Ans Relevancy {scores['Answer Relevancy']:.2f}   → "
              f"{'Answers match questions ✅' if scores['Answer Relevancy'] >= 0.8 else 'Answers off-topic ⚠️'}")
        print(f"  Ctx Precision {scores['Context Precision']:.2f}   → "
              f"{'Retrieved chunks relevant ✅' if scores['Context Precision'] >= 0.8 else 'Noisy retrieval ⚠️'}")
        print(f"  Ctx Recall    {scores['Context Recall']:.2f}   → "
              f"{'Good coverage ✅' if scores['Context Recall'] >= 0.8 else 'Missing some info ⚠️'}")

        # ── Save results to JSON ───────────────────────────────────────────────
        output = {
            "scores"      : scores,
            "overall"     : round(overall, 4),
            "num_questions": len(raw_data["question"]),
            "model"       : "llama3",
            "embedding"   : "all-MiniLM-L6-v2",
            "chunk_size"  : 500,
            "chunk_overlap": 50,
            "top_k"       : 4,
        }

        with open("data/evaluation_results.json", "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n  💾 Results saved to: data/evaluation_results.json")
        print("\n  ✅ Evaluation complete!\n")

        return scores

    except Exception as e:
        print(f"\n  ❌ RAGAS evaluation failed: {e}")
        print("\n  Common causes:")
        print("  1. Ollama not running — start with: ollama serve")
        print("  2. RAGAS needs OpenAI by default — see fix below")
        print("\n  Fix: RAGAS uses OpenAI to judge by default.")
        print("  Run this to use a local judge instead:")
        print("  pip install ragas[all]")
        raise


if __name__ == "__main__":
    run_evaluation()