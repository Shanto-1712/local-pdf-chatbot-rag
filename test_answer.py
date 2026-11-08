
from pdf_processor import extract_text_from_pdf, split_into_chunks
from rag_engine import RAGEngine

pages = extract_text_from_pdf("data/sample.pdf")
chunks = split_into_chunks(
    pages,
    chunk_size=180,
    overlap=40
)

engine = RAGEngine()
engine.build_index(chunks)

while True:
    question = input("\nAsk about your PDF (type exit to quit): ")

    if question.strip().lower() == "exit":
        break

    if not question.strip():
        continue

    answer, sources = engine.generate_answer(question)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCE PAGES:")
    page_numbers = sorted({item["page"] for item in sources})
    print(", ".join(map(str, page_numbers)))
