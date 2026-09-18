from .models import DocumentChunk

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[DocumentChunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    words = text.split()
    chunks = []
    start = 0
    index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(DocumentChunk(f"{source}:{index}", source, chunk))

        if end == len(words):
            break

        start = end - overlap
        index += 1

    return chunks
