
from pdf_processor import extract_text_from_pdf, split_into_chunks

pdf_path = "data/sample.pdf"

pages = extract_text_from_pdf(pdf_path)
chunks = split_into_chunks(pages)

print("Pages with text:", len(pages))
print("Chunks created:", len(chunks))

if chunks:
    print("\nFirst chunk:")
    print(chunks[0]["text"][:500])
    print("\nSource page:", chunks[0]["page"])
else:
    print("No text found. The PDF may be scanned or empty.")
