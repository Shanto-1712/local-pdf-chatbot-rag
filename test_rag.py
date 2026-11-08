
from pdf_processor import extract_text_from_pdf, split_into_chunks
from rag_engine import RAGEngine

pdf_path = "data/sample.pdf"

pages = extract_text_from_pdf(pdf_path)
chunks = split_into_chunks(pages)

engine = RAGEngine()
engine.build_index(chunks)

question = input("\nAsk a question about your PDF: ")
results = engine.retrieve(question, top_k=3)

print("\nRelevant passages:\n")

for i, result in enumerate(results, start=1):
    print(f"Result {i} | Page {result['page']}")
    print(f"Similarity score: {result['score']:.3f}")
    print(result["text"])
    print("-" * 50)
