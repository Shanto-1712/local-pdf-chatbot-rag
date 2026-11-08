
import pymupdf


def extract_text_from_pdf(pdf_path):
    """Extract text from every page of a PDF."""
    extracted_pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                extracted_pages.append({
                    "page": page_number,
                    "text": text
                })

    return extracted_pages


def split_into_chunks(pages, chunk_size=500, overlap=100):
    """Split page text into overlapping word chunks."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Use chunk_size > overlap >= 0.")

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        words = page["text"].split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                continue

            chunks.append({
                "page": page["page"],
                "text": " ".join(chunk_words)
            })

            if start + chunk_size >= len(words):
                break

    return chunks
