from research_assistant.ingestion import load_directory

def test_load_directory(tmp_path):
    (tmp_path / "a.md").write_text("# Hello\nSemantic retrieval is useful.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Python is widely used for ML.", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("a,b", encoding="utf-8")
    chunks = load_directory(tmp_path)
    assert len(chunks) == 2
