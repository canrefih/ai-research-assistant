from research_assistant.evaluation import EvaluationCase, EvaluationMetrics, evaluate_retrieval, load_evaluation_dataset, mean_reciprocal_rank, ndcg_at_k, recall_at_k, benchmark_pipeline, benchmark_dataset
from pathlib import Path
import pytest
from research_assistant.models import ResearchSource
from research_assistant.pipeline import ResearchPipeline


def test_evaluation_case_stores_question_and_relevant_sources():
	case = EvaluationCase(
		question="What is retrieval-augmented generation?",
		relevant_sources=["docs/rag.md"],
	)

	assert case.question == "What is retrieval-augmented generation?"
	assert case.relevant_sources == ["docs/rag.md"]


def test_load_evaluation_dataset(tmp_path):
	dataset = tmp_path / "golden.jsonl"
	dataset.write_text(
		'{"question": "What is RAG?", "relevant_sources": ["docs/rag.md"]}\n'
		'{"question": "What is BM25?", "relevant_sources": ["docs/bm25.md"]}\n',
		encoding="utf-8",
	)

	cases = load_evaluation_dataset(dataset)

	assert len(cases) == 2
	assert cases[0].question == "What is RAG?"
	assert cases[0].relevant_sources == ["docs/rag.md"]
	assert cases[1].question == "What is BM25?"


def test_golden_dataset_file_loads():
	dataset_path = (
		Path(__file__).parent
		/ "data"
		/ "golden_dataset.jsonl"
	)

	cases = load_evaluation_dataset(dataset_path)

	assert len(cases) == 2
	assert cases[0].question == (
		"What is retrieval-augmented generation?"
	)
	assert cases[1].question == "What is information retrieval?"


def test_recall_at_k():
	retrieved = [
		"docs/rag.md",
		"docs/other.md",
		"docs/retrieval.md",
	]
	relevant = [
		"docs/rag.md",
		"docs/retrieval.md",
	]

	assert recall_at_k(retrieved, relevant, 3) == 1.0
	assert recall_at_k(retrieved, relevant, 1) == 0.5


def test_recall_at_k_returns_zero_for_empty_relevant_sources():
	assert recall_at_k(
		["docs/rag.md"],
		[],
		5,
	) == 0.0


def test_mean_reciprocal_rank():
	retrieved = [
		[
			"docs/other.md",
			"docs/rag.md",
		],
		[
			"docs/bm25.md",
			"docs/other.md",
		],
	]
	relevant = [
		["docs/rag.md"],
		["docs/bm25.md"],
	]

	assert mean_reciprocal_rank(
		retrieved,
		relevant,
	) == 0.75


def test_mean_reciprocal_rank_returns_zero_when_no_relevant_result():
	assert mean_reciprocal_rank(
		[["docs/other.md"]],
		[["docs/rag.md"]],
	) == 0.0


def test_mean_reciprocal_rank_returns_zero_for_empty_input():
	assert mean_reciprocal_rank([], []) == 0.0


def test_ndcg_at_k():
	retrieved = [
		"docs/other.md",
		"docs/rag.md",
		"docs/bm25.md",
	]
	relevant = [
		"docs/rag.md",
		"docs/bm25.md",
	]

	assert ndcg_at_k(
		retrieved,
		relevant,
		3,
	) == pytest.approx(0.6934264036172708)


def test_ndcg_at_k_returns_zero_for_no_relevant_results():
	assert ndcg_at_k(
		["docs/other.md"],
		["docs/rag.md"],
		1,
	) == 0.0


def test_ndcg_at_k_returns_zero_for_empty_relevant_sources():
	assert ndcg_at_k(
		["docs/rag.md"],
		[],
		1,
	) == 0.0


def test_evaluation_metrics_stores_scores():
	metrics = EvaluationMetrics(
		recall_at_k=0.8,
		mean_reciprocal_rank=0.75,
		ndcg_at_k=0.7,
	)

	assert metrics.recall_at_k == 0.8
	assert metrics.mean_reciprocal_rank == 0.75
	assert metrics.ndcg_at_k == 0.7


def test_evaluate_retrieval():
	retrieved = [
		[
			"docs/other.md",
			"docs/rag.md",
		],
		[
			"docs/bm25.md",
			"docs/other.md",
		],
	]
	relevant = [
		["docs/rag.md"],
		["docs/bm25.md"],
	]

	metrics = evaluate_retrieval(
		retrieved,
		relevant,
		2,
	)

	assert metrics.recall_at_k == 1.0
	assert metrics.mean_reciprocal_rank == 0.75
	assert metrics.ndcg_at_k > 0.8


def test_evaluate_retrieval_rejects_mismatched_inputs():
	with pytest.raises(
		ValueError,
		match="retrieved_sources and relevant_sources must have the same length",
	):
		evaluate_retrieval(
			[["docs/rag.md"]],
			[],
			1,
		)


def test_benchmark_pipeline():
	class FakePipeline:
		def _build_research_context(
			self,
			question,
			top_k,
		):
			if "information retrieval" in question:
				return "", [
					ResearchSource(source="data/sample/retrieval.md"),
				]

			return "", [
				ResearchSource(source="data/sample/rag.md"),
			]

	cases = [
		EvaluationCase(
			question="What is retrieval-augmented generation?",
			relevant_sources=["data/sample/rag.md"],
		),
		EvaluationCase(
			question="What is information retrieval?",
			relevant_sources=["data/sample/retrieval.md"],
		),
	]

	metrics = benchmark_pipeline(
		FakePipeline(),
		cases,
		k=1,
	)

	assert metrics.recall_at_k == 1.0
	assert metrics.mean_reciprocal_rank == 1.0
	assert metrics.ndcg_at_k == 1.0


def test_benchmark_dataset(tmp_path):
    dataset_path = tmp_path / "golden.jsonl"

    dataset_path.write_text(
        '{"question":"What is retrieval-augmented generation?",'
        '"relevant_sources":["data/sample/rag.md"]}\n'
        '{"question":"What is information retrieval?",'
        '"relevant_sources":["data/sample/retrieval.md"]}\n',
        encoding="utf-8",
    )

    class FakePipeline:
        def _build_research_context(
            self,
            question,
            top_k,
        ):
            if "information retrieval" in question:
                return "", [
                    ResearchSource(
                        source="data/sample/retrieval.md"
                    ),
                ]

            return "", [
                ResearchSource(
                    source="data/sample/rag.md"
                ),
            ]

    metrics = benchmark_dataset(
        FakePipeline(),
        dataset_path,
        k=1,
    )

    assert metrics.recall_at_k == 1.0
    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.ndcg_at_k == 1.0