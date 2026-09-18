from research_assistant.chunking import chunk_text
from research_assistant.models import DocumentChunk

def test_chunking_preserves_source_and_content():
    chunks = chunk_text("one two three four five six", "demo.md", chunk_size=4, overlap=1)
    assert chunks
    assert all(c.source == "demo.md" for c in chunks)
    assert chunks[0].text == "one two three four"


def test_document_chunk_supports_metadata():
    chunk = DocumentChunk(
        chunk_id="test:0",
        source="test.md",
        text="Example text",
        metadata={"topic": "python"},
    )

    assert chunk.metadata == {"topic": "python"}


def test_chunk_text_preserves_metadata():
    chunks = chunk_text(
        "one two three four",
        "test.md",
        chunk_size=2,
        overlap=0,
        metadata={
            "source": "test.md",
            "file_type": "md",
        },
    )

    assert len(chunks) == 2
    assert all(
        chunk.metadata == {
            "source": "test.md",
            "file_type": "md",
        }
        for chunk in chunks
    )


def test_chunk_text_defaults_to_empty_metadata():
    chunks = chunk_text(
        "one two three",
        "test.md",
        chunk_size=2,
        overlap=0,
    )

    assert len(chunks) == 2
    assert all(chunk.metadata == {} for chunk in chunks)