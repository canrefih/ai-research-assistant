from research_assistant.ingestion import load_directory

def test_load_directory(tmp_path):
    (tmp_path / "a.md").write_text("# Hello\nSemantic retrieval is useful.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Python is widely used for ML.", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("a,b", encoding="utf-8")
    chunks = load_directory(tmp_path)
    assert len(chunks) == 2


def test_load_directory_adds_file_metadata(tmp_path):
    document = tmp_path / "python.md"
    document.write_text(
        "# Python\n\nPython is a programming language.",
        encoding="utf-8",
    )

    chunks = load_directory(tmp_path)

    assert len(chunks) == 1
    assert chunks[0].metadata == {
        "source": "python.md",
        "file_type": "md",
    }


def test_load_directory_adds_txt_file_metadata(tmp_path):
    document = tmp_path / "notes.txt"
    document.write_text(
        "Python is useful for research.",
        encoding="utf-8",
    )

    chunks = load_directory(tmp_path)

    assert len(chunks) == 1
    assert chunks[0].metadata == {
        "source": "notes.txt",
        "file_type": "txt",
    }