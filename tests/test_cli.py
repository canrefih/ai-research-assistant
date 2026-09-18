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