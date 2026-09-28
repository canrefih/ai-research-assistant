import pytest
import requests
import sys

from research_assistant.cli import main
import research_assistant.cli as cli


class FakePipeline:
    def __init__(self, *args, **kwargs):
        pass

    def load_index(self):
        return 2

    def ask(self, question, top_k=8, use_web_search=False):
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

        def ask(self, question, top_k=8, use_web_search=False):
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

        def ask(self, question, top_k=8, use_web_search=False):
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


def test_ask_passes_model_names_to_pipeline(monkeypatch):
    pipeline_args = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            pipeline_args.update(kwargs)

        def load_index(self):
            return 1

        def ask(self, question, top_k=8, use_web_search=False):
            return "answer"

    monkeypatch.setattr(
        "research_assistant.cli.ResearchPipeline",
        FakePipeline,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "research-assistant",
            "ask",
            "What is Python?",
            "--no-reranker",
            "--embedding-model",
            "test-embedding-model",
            "--reranker-model",
            "test-reranker-model",
        ],
    )

    from research_assistant.cli import main

    main()

    assert pipeline_args == {
        "use_reranker": False,
        "index_dir": "data/index",
        "embedding_model": "test-embedding-model",
        "reranker_model": "test-reranker-model",
    }


def test_index_passes_embedding_model_to_pipeline(monkeypatch):
    pipeline_args = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            pipeline_args.update(kwargs)

        def index(self, directory):
            return 3

    monkeypatch.setattr(
        "research_assistant.cli.ResearchPipeline",
        FakePipeline,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "research-assistant",
            "index",
            "data/sample",
            "--embedding-model",
            "test-embedding-model",
        ],
    )

    from research_assistant.cli import main

    main()

    assert pipeline_args == {
        "index_dir": "data/index",
        "embedding_model": "test-embedding-model",
    }


def test_index_url_passes_url_and_options_to_pipeline(monkeypatch, capsys):
    pipeline_args = {}
    call_args = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            pipeline_args.update(kwargs)

        def index_url(self, url):
            call_args["url"] = url
            return 4

    monkeypatch.setattr(
        "research_assistant.cli.ResearchPipeline",
        FakePipeline,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "research-assistant",
            "index-url",
            "https://example.com/research",
            "--index-dir",
            "data/web-index",
            "--embedding-model",
            "test-embedding-model",
        ],
    )

    main()

    assert pipeline_args == {
        "index_dir": "data/web-index",
        "embedding_model": "test-embedding-model",
    }
    assert call_args == {
        "url": "https://example.com/research",
    }

    captured = capsys.readouterr()

    assert captured.out == (
        "Indexed 4 chunks from https://example.com/research "
        "into data/web-index.\n"
    )
    assert captured.err == ""


def test_index_url_handles_request_error(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "research-assistant",
            "index-url",
            "https://example.com/research",
        ],
    )

    class RequestErrorPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def index_url(self, url):
            raise requests.Timeout("request timed out")

    monkeypatch.setattr(
        "research_assistant.cli.ResearchPipeline",
        RequestErrorPipeline,
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert captured.err.strip() == "Error: request timed out"


def test_ask_with_web_enables_web_search(monkeypatch):
	pipeline_args = {}
	ask_args = {}

	class FakePipeline:
		def __init__(self, **kwargs):
			pipeline_args.update(kwargs)

		def load_index(self):
			return 1

		def ask(self, question, top_k=8, use_web_search=False):
			ask_args.update(
				question=question,
				top_k=top_k,
				use_web_search=use_web_search,
			)
			return "answer"

	class FakeWebSearchProvider:
		pass

	monkeypatch.setattr(
		"research_assistant.cli.ResearchPipeline",
		FakePipeline,
	)
	monkeypatch.setattr(
		"research_assistant.cli.TavilySearchProvider",
		FakeWebSearchProvider,
	)
	monkeypatch.setattr(
		"sys.argv",
		[
			"research-assistant",
			"ask",
			"What is RAG?",
			"--web",
		],
	)

	from research_assistant.cli import main

	main()

	assert isinstance(
		pipeline_args["web_search_provider"],
		FakeWebSearchProvider,
	)
	assert ask_args == {
		"question": "What is RAG?",
		"top_k": 8,
		"use_web_search": True,
	}

def test_ask_with_verify_sources_enables_source_verification(monkeypatch):
        pipeline_args = {}
        ask_args = {}

        class FakePipeline:
                def __init__(self, **kwargs):
                        pipeline_args.update(kwargs)

                def load_index(self):
                        return 1

                def ask(
                        self,
                        question,
                        top_k=8,
                        use_web_search=False,
                        verify_sources=False,
                ):
                        ask_args.update(
                                question=question,
                                top_k=top_k,
                                use_web_search=use_web_search,
                                verify_sources=verify_sources,
                        )
                        return "answer"

        class FakeWebSearchProvider:
                pass

        monkeypatch.setattr(
                "research_assistant.cli.ResearchPipeline",
                FakePipeline,
        )
        monkeypatch.setattr(
                "research_assistant.cli.TavilySearchProvider",
                FakeWebSearchProvider,
        )
        monkeypatch.setattr(
                "research_assistant.cli.HttpSourceVerifier",
                lambda: "fake-verifier",
        )
        monkeypatch.setattr(
                "sys.argv",
                [
                        "research-assistant",
                        "ask",
                        "What is RAG?",
                        "--web",
                        "--verify-sources",
                ],
        )

        from research_assistant.cli import main

        main()

        assert pipeline_args["source_verifier"] == "fake-verifier"
        assert ask_args == {
                "question": "What is RAG?",
                "top_k": 8,
                "use_web_search": True,
                "verify_sources": True,
        }

def test_ask_without_web_disables_web_search(monkeypatch):
	pipeline_args = {}
	ask_args = {}

	class FakePipeline:
		def __init__(self, **kwargs):
			pipeline_args.update(kwargs)

		def load_index(self):
			return 1

		def ask(
			self,
			question,
			top_k=8,
			use_web_search=False,
			use_query_expansion=False,
		):
			ask_args.update(
				question=question,
				top_k=top_k,
				use_web_search=use_web_search,
				use_query_expansion=use_query_expansion,
			)
			return "answer"

	monkeypatch.setattr(
		"research_assistant.cli.ResearchPipeline",
		FakePipeline,
	)
	monkeypatch.setattr(
		"sys.argv",
		[
			"research-assistant",
			"ask",
			"What is RAG?",
		],
	)

	from research_assistant.cli import main

	main()

	assert "web_search_provider" not in pipeline_args
	assert ask_args == {
		"question": "What is RAG?",
		"top_k": 8,
		"use_web_search": False,
		"use_query_expansion": False,
	}


def test_cli_crawl_url(monkeypatch, capsys):
    calls = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def crawl_url(self, url, max_pages):
            calls["crawl"] = (url, max_pages)
            return 3

    monkeypatch.setattr(
        "research_assistant.cli.ResearchPipeline",
        FakePipeline,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "research-assistant",
            "crawl-url",
            "https://example.com",
            "--max-pages",
            "7",
        ],
    )

    from research_assistant.cli import main

    main()

    assert calls["init"] == {
        "index_dir": "data/index",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }
    assert calls["crawl"] == (
        "https://example.com",
        7,
    )

    captured = capsys.readouterr()

    assert (
        "Crawled 3 chunks from https://example.com "
        "into data/index."
    ) in captured.out


def test_ask_with_query_expansion_enables_expansion(monkeypatch):
	pipeline_args = {}
	ask_args = {}

	class FakeExpander:
		pass

	class FakePipeline:
		def __init__(self, **kwargs):
			pipeline_args.update(kwargs)

		def load_index(self):
			return 2

		def ask(
			self,
			question,
			top_k=8,
			use_web_search=False,
			use_query_expansion=False,
		):
			ask_args.update(
				question=question,
				top_k=top_k,
				use_web_search=use_web_search,
				use_query_expansion=use_query_expansion,
			)
			return "answer"

	monkeypatch.setattr(
		"research_assistant.cli.ResearchPipeline",
		FakePipeline,
	)
	monkeypatch.setattr(
		"research_assistant.cli.LLMQueryExpander",
		FakeExpander,
	)
	monkeypatch.setattr(
		"sys.argv",
		[
			"research-assistant",
			"ask",
			"What is RAG?",
			"--query-expansion",
		],
	)

	from research_assistant.cli import main

	main()

	assert isinstance(
		pipeline_args["query_expander"],
		FakeExpander,
	)
	assert ask_args == {
		"question": "What is RAG?",
		"top_k": 8,
		"use_web_search": False,
		"use_query_expansion": True,
	}


def test_benchmark_command_prints_metrics(monkeypatch, capsys):
    class FakePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_index(self):
            return 2

    class FakeMetrics:
        recall_at_k = 0.75
        mean_reciprocal_rank = 0.6
        ndcg_at_k = 0.8

    monkeypatch.setattr(
        cli,
        "ResearchPipeline",
        FakePipeline,
    )
    monkeypatch.setattr(
        cli,
        "benchmark_dataset",
        lambda pipeline, dataset, k: FakeMetrics(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research-assistant",
            "benchmark",
            "data/evaluation/golden.jsonl",
            "--top-k",
            "5",
        ],
    )

    cli.main()

    output = capsys.readouterr().out

    assert "Recall@5: 0.7500" in output
    assert "MRR: 0.6000" in output
    assert "NDCG@5: 0.8000" in output


def test_benchmark_with_faithfulness(capsys, monkeypatch):
    class FakeMetrics:
        recall_at_k = 1.0
        mean_reciprocal_rank = 1.0
        ndcg_at_k = 1.0

    class FakeFaithfulnessMetrics:
        supported_claim_ratio = 0.75

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        def load_index(self):
            return 1

    def fake_benchmark_dataset(pipeline, dataset, k):
        return FakeMetrics()

    def fake_load_evaluation_dataset(path):
        return ["fake-case"]

    def fake_benchmark_faithfulness(pipeline, cases):
        return FakeFaithfulnessMetrics()

    monkeypatch.setattr(
        "research_assistant.cli.ResearchPipeline",
        FakePipeline,
    )
    monkeypatch.setattr(
        "research_assistant.cli.benchmark_dataset",
        fake_benchmark_dataset,
    )
    monkeypatch.setattr(
        "research_assistant.cli.load_evaluation_dataset",
        fake_load_evaluation_dataset,
    )
    monkeypatch.setattr(
        "research_assistant.cli.benchmark_faithfulness",
        fake_benchmark_faithfulness,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research_assistant",
            "benchmark",
            "dataset.jsonl",
            "--faithfulness",
        ],
    )

    main()

    output = capsys.readouterr().out

    assert "Recall@5: 1.0000" in output
    assert "MRR: 1.0000" in output
    assert "NDCG@5: 1.0000" in output
    assert "Supported Claim Ratio: 0.7500" in output


def test_benchmark_answer_quality(capsys, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research-assistant",
            "benchmark",
            "tests/data/golden_dataset.jsonl",
            "--answer-quality",
        ],
    )

    main()

    captured = capsys.readouterr()

    assert "Answer Overlap Score:" in captured.out