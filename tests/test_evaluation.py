from research_assistant.evaluation import (
	EvaluationCase,
	EvaluationMetrics,
	evaluate_retrieval,
	load_evaluation_dataset,
	mean_reciprocal_rank,
	ndcg_at_k,
	recall_at_k,
	benchmark_pipeline,
	benchmark_dataset,
	supported_claim_ratio,
	evaluate_faithfulness,
	benchmark_faithfulness,
	_content_words,
	answer_overlap_score,
	evaluate_answer_quality,
	benchmark_answer_quality
	)
from pathlib import Path
import pytest
from research_assistant.models import ResearchSource
from research_assistant.pipeline import ResearchPipeline


def test_evaluation_case_stores_question_and_relevant_sources():
	case = EvaluationCase(
		question="What is retrieval-augmented generation?",
		relevant_sources=["docs/rag.md"],
		reference_answer="",
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
			reference_answer="",
		),
		EvaluationCase(
			question="What is information retrieval?",
			relevant_sources=["data/sample/retrieval.md"],
			reference_answer="",
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


def test_supported_claim_ratio():
	answer = "RAG uses retrieved documents. It improves factual grounding."
	evidence = "RAG uses retrieved documents to provide additional context."

	assert supported_claim_ratio(answer, evidence) == 0.5


def test_supported_claim_ratio_returns_zero_for_empty_answer():
	assert supported_claim_ratio("", "Some evidence.") == 0.0


def test_supported_claim_ratio_returns_zero_when_claims_are_unsupported():
	answer = "RAG completely eliminates hallucinations."
	evidence = "RAG retrieves relevant documents."

	assert supported_claim_ratio(answer, evidence) == 0.0


def test_evaluate_faithfulness():
	answers = [
		"RAG uses retrieved documents.",
		"It improves factual grounding.",
	]
	evidence = [
		"RAG uses retrieved documents.",
		"It improves factual grounding and context.",
	]

	assert evaluate_faithfulness(
		answers,
		evidence,
	).supported_claim_ratio == 1.0


def test_evaluate_faithfulness_returns_zero_for_empty_input():
    assert (
        evaluate_faithfulness([], []).supported_claim_ratio
        == 0.0
    )


def test_evaluate_faithfulness_requires_matching_lengths():
	with pytest.raises(ValueError):
		evaluate_faithfulness(
			["RAG uses retrieved documents."],
			[],
		)


def test_benchmark_faithfulness():
	class FakeLLM:
		def answer(self, question, evidence):
			return "RAG uses retrieved documents."

	class FakePipeline:
		def __init__(self):
			self.llm = FakeLLM()

		def _build_research_context(self, question):
			return (
				"RAG uses retrieved documents to provide context.",
				[],
			)

	cases = [
		EvaluationCase(
			question="What is RAG?",
			relevant_sources=["data/sample/rag.md"],
			reference_answer="",
		)
	]

	score = benchmark_faithfulness(
		FakePipeline(),
		cases,
	)

	assert score.supported_claim_ratio == 1.0


def test_content_words_removes_stop_words():
	assert _content_words(
		"RAG uses retrieved documents for context."
	) == {
		"rag",
		"uses",
		"retrieved",
		"documents",
		"context",
	}


def test_supported_claim_ratio_accepts_fifty_percent_word_overlap():
	answer = "RAG uses retrieved documents."
	evidence = "RAG uses context."

	assert supported_claim_ratio(answer, evidence) == 1.0


def test_benchmark_faithfulness_returns_zero_for_empty_cases():
    class FakePipeline:
        pass

    metrics = benchmark_faithfulness(
        FakePipeline(),
        [],
    )

    assert metrics.supported_claim_ratio == 0.0


def test_answer_overlap_score_full_match():
    score = answer_overlap_score(
        "Retrieval uses relevant documents.",
        "Retrieval uses relevant documents.",
    )

    assert score == 1.0


def test_answer_overlap_score_partial_match():
    score = answer_overlap_score(
        "Retrieval uses documents.",
        "Retrieval uses documents context.",
    )

    assert score == 3 / 4


def test_answer_overlap_score_no_match():
    score = answer_overlap_score(
        "Cats sleep often.",
        "Retrieval uses documents.",
    )

    assert score == 0.0


def test_answer_overlap_score_empty_reference():
    score = answer_overlap_score(
        "Retrieval uses documents.",
        "",
    )

    assert score == 0.0


def test_evaluate_answer_quality_empty():
    metrics = evaluate_answer_quality(
        [],
        [],
    )

    assert metrics.answer_overlap_score == 0.0


def test_evaluate_answer_quality_average():
    metrics = evaluate_answer_quality(
        [
            "Retrieval uses documents.",
            "Cats sleep.",
        ],
        [
            "Retrieval uses documents.",
            "Dogs run.",
        ],
    )

    assert metrics.answer_overlap_score == 0.5


def test_evaluate_answer_quality_rejects_length_mismatch():
    with pytest.raises(ValueError):
        evaluate_answer_quality(
            ["Retrieval uses documents."],
            [],
        )


def test_benchmark_answer_quality():
    class FakeLLM:
        def answer(self, question, evidence):
            return "RAG uses retrieved documents."

    class FakePipeline:
        def __init__(self):
            self.llm = FakeLLM()

        def _build_research_context(self, question):
            return (
                "RAG uses retrieved documents to provide context.",
                [],
            )

    cases = [
        EvaluationCase(
            question="What is RAG?",
            relevant_sources=["data/sample/rag.md"],
            reference_answer="RAG uses retrieved documents.",
        )
    ]

    metrics = benchmark_answer_quality(
        FakePipeline(),
        cases,
    )

    assert metrics.answer_overlap_score == 1.0