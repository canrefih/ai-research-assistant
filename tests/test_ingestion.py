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


def test_load_directory_deduplicates_identical_files(tmp_path):
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"

    content = "# Python\n\nPython is a programming language."

    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")

    chunks = load_directory(tmp_path)

    assert len(chunks) == 1


def test_load_directory_keeps_different_files(tmp_path):
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"

    first.write_text(
        "# Python\n\nPython is a programming language.",
        encoding="utf-8",
    )
    second.write_text(
        "# Java\n\nJava is a programming language.",
        encoding="utf-8",
    )

    chunks = load_directory(tmp_path)

    assert len(chunks) == 2


def test_load_directory_deduplicates_same_content_with_different_metadata(
    tmp_path,
):
    first = tmp_path / "first.md"
    second = tmp_path / "second.txt"

    content = "Python is a programming language."

    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")

    chunks = load_directory(tmp_path)

    assert len(chunks) == 1
    assert chunks[0].metadata == {
        "source": "first.md",
        "file_type": "md",
    }