import pytest

from research_assistant.cli import main


class FakePipeline:
    def __init__(self, *args, **kwargs):
        pass

    def load_index(self):
        return 2

    def ask(self, question, top_k=8):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        return "fake answer"


def test_ask_rejects_invalid_top_k(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["research-assistant", "ask", "test", "--top-k", "0"],
    )

    monkeypatch.setattr(
        "research_assistant.cli.ResearchPipeline",
        FakePipeline,
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert captured.err.strip() == "Error: top_k must be at least 1"


def test_ask_rejects_missing_index(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "research-assistant",
            "ask",
            "What is RAG?",
            "--index-dir",
            "does-not-exist",
        ],
    )

    class MissingIndexPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def load_index(self):
            raise FileNotFoundError(
                "No index found at does-not-exist. "
                "Run 'research-assistant index <directory>' first."
            )

    monkeypatch.setattr(
        "research_assistant.cli.ResearchPipeline",
        MissingIndexPipeline,
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert (
        captured.err.strip()
        == "Error: No index found at does-not-exist. "
        "Run 'research-assistant index <directory>' first."
    )


def test_index_rejects_empty_directory(monkeypatch, capsys, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    monkeypatch.setattr(
        "sys.argv",
        [
            "research-assistant",
            "index",
            str(empty_dir),
        ],
    )

    class EmptyDirectoryPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def index(self, directory):
            raise ValueError(
                f"No .md or .txt documents found in {directory}"
            )

    monkeypatch.setattr(
        "research_assistant.cli.ResearchPipeline",
        EmptyDirectoryPipeline,
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert (
        captured.err.strip()
        == f"Error: No .md or .txt documents found in {empty_dir}"
    )


def test_ask_handles_runtime_error(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "research-assistant",
            "ask",
            "What is RAG?",
        ],
    )

    class RuntimeErrorPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def load_index(self):
            return 2

        def ask(self, question, top_k=8):
            raise RuntimeError("Retriever is not fitted")

    monkeypatch.setattr(
        "research_assistant.cli.ResearchPipeline",
        RuntimeErrorPipeline,
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert captured.err.strip() == "Error: Retriever is not fitted"


def test_ask_prints_loaded_count_and_answer(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "research-assistant",
            "ask",
            "What is RAG?",
        ],
    )

    class SuccessfulPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def load_index(self):
            return 2

        def ask(self, question, top_k=8):
            assert question == "What is RAG?"
            assert top_k == 8
            return "fake answer"

    monkeypatch.setattr(
        "research_assistant.cli.ResearchPipeline",
        SuccessfulPipeline,
    )

    main()

    captured = capsys.readouterr()

    assert captured.out == (
        "Loaded 2 chunks from data/index.\n"
        "fake answer\n"
    )
    assert captured.err == ""
