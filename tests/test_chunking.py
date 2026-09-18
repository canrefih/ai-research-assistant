from research_assistant.chunking import chunk_text

def test_chunking_preserves_source_and_content():
    chunks = chunk_text("one two three four five six", "demo.md", chunk_size=4, overlap=1)
    assert chunks
    assert all(c.source == "demo.md" for c in chunks)
    assert chunks[0].text == "one two three four"
