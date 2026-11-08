
import faiss
import numpy as np
import ollama

from sentence_transformers import SentenceTransformer


def validate_ml_answer(question, answer):
    """Correct a common classification/clustering mix-up."""
    q = question.lower()
    a = answer.lower()

    asks_comparison = (
        "classification" in q
        and "clustering" in q
        and any(word in q for word in ["difference", "compare", "versus", " vs "])
    )

    if not asks_comparison:
        return answer

    classification_wrong = (
        "classification is an unsupervised" in a
        or "classification is a unsupervised" in a
    )
    clustering_wrong = (
        "clustering is a supervised" in a
        or "clustering is supervised" in a
    )

    if classification_wrong or clustering_wrong:
        return (
            "Classification is generally a supervised learning technique: "
            "it learns from labeled examples to predict predefined classes. "
            "Clustering is generally an unsupervised learning technique: "
            "it groups similar data points without predefined class labels. "
            "These are standard machine-learning definitions. Check the "
            "retrieved PDF evidence below to see whether your document "
            "explains this distinction."
        )

    return answer


class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        """Create a searchable FAISS index from PDF chunks."""
        if not chunks:
            raise ValueError("No text chunks were provided.")

        texts = [
            chunk["text"].strip()
            for chunk in chunks
            if chunk.get("text", "").strip()
        ]

        valid_chunks = [
            chunk for chunk in chunks
            if chunk.get("text", "").strip()
        ]

        if not texts:
            raise ValueError("The PDF contains no usable text.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Inner product on normalized vectors gives cosine similarity.
        new_index = faiss.IndexFlatIP(embeddings.shape[1])
        new_index.add(embeddings)

        self.index = new_index
        self.chunks = valid_chunks

        print(f"Index created with {len(self.chunks)} chunks.")

    def retrieve(self, question, top_k=3):
        """Find passages most relevant to the user's question."""
        if self.index is None:
            raise RuntimeError("Build the index before searching.")

        if not question or not question.strip():
            return []

        question_embedding = self.model.encode(
            [question.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        question_embedding = np.asarray(
            question_embedding,
            dtype=np.float32,
        )

        k = min(max(1, top_k), len(self.chunks))
        scores, indices = self.index.search(question_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            result = self.chunks[int(idx)].copy()
            result["score"] = float(score)
            results.append(result)

        return results

    def generate_answer(self, question, top_k=3):
        """Generate a grounded answer and return its source passages."""
        results = self.retrieve(question, top_k=top_k)

        if not results:
            return (
                "I could not find relevant information in the PDF.",
                [],
            )

        context_parts = []

        for item in results:
            page = item.get("page", "Unknown")
            text = item["text"].strip()

            context_parts.append(
                f"[Source: PDF page {page}]\n{text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""
You are a careful assistant that answers questions about a user's PDF.

Answer the QUESTION using only the EVIDENCE provided below.


RULES:
1. Answer the exact question asked.
2. Use only the supplied PDF evidence for document-specific facts.
3. Keep answers concise and avoid unrelated topics.
4. If evidence is insufficient, say so clearly.
5. Cite supporting PDF pages using [Page N].
6. Never invent facts, quotations, or page numbers.
7. For comparisons, explain each item separately.
8. Avoid repeating the same information.
9. For machine learning questions, check that your
   explanation is internally consistent.
10. Classification is normally supervised learning:
    it learns from labeled examples to predict classes.
11. Clustering is normally unsupervised learning:
    it groups data without predefined class labels.
12. Do not describe classification as unsupervised.
13. If the PDF conflicts with standard terminology,
    identify the discrepancy instead of silently repeating it.
14. Do not claim the PDF supports a fact unless the
    supplied evidence actually contains that information.

EVIDENCE FROM THE PDF:
{context}

QUESTION:
{question.strip()}

GROUNDED ANSWER:
"""

        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise PDF question-answering "
                        "assistant. Follow the supplied evidence "
                        "and citation rules."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.1,
                "num_predict": 350,
            },
        )

        answer = response["message"]["content"].strip()
        answer = validate_ml_answer(question, answer)
        return answer, results
