import json
import math
from dataclasses import dataclass
from pathlib import Path

from research_assistant.pipeline import ResearchPipeline


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    relevant_sources: list[str]
    reference_answer: str


@dataclass(frozen=True)
class EvaluationMetrics:
	recall_at_k: float
	mean_reciprocal_rank: float
	ndcg_at_k: float


@dataclass(frozen=True)
class FaithfulnessMetrics:
	supported_claim_ratio: float


@dataclass(frozen=True)
class AnswerQualityMetrics:
    answer_overlap_score: float


@dataclass(frozen=True)
class EvaluationError:
    question: str
    expected_sources: list[str]
    retrieved_sources: list[str]


def analyze_retrieval_errors(
    cases: list[EvaluationCase],
    retrieved_sources: list[list[str]],
) -> list[EvaluationError]:
    if len(cases) != len(retrieved_sources):
        raise ValueError(
            "cases and retrieved_sources must have the same length"
        )

    errors = []

    for case, retrieved in zip(
        cases,
        retrieved_sources,
    ):
        if not any(
            source in case.relevant_sources
            for source in retrieved
        ):
            errors.append(
                EvaluationError(
                    question=case.question,
                    expected_sources=case.relevant_sources,
                    retrieved_sources=retrieved,
                )
            )

    return errors


def benchmark_retrieval_errors(
    pipeline,
    cases: list[EvaluationCase],
    top_k: int = 5,
) -> list[EvaluationError]:
    retrieved_sources = []

    for case in cases:
        _, sources = pipeline._build_research_context(
            case.question,
            top_k=top_k,
        )

        retrieved_sources.append(
            [source.source for source in sources]
        )

    return analyze_retrieval_errors(
        cases,
        retrieved_sources,
    )


def load_evaluation_dataset(
	path: str | Path,
) -> list[EvaluationCase]:
	cases = []

	with open(path, "r", encoding="utf-8") as file:
		for line in file:
			if not line.strip():
				continue

			data = json.loads(line)
			cases.append(
				EvaluationCase(
					question=data["question"],
					relevant_sources=data["relevant_sources"],
					reference_answer="",
				)
			)

	return cases


def recall_at_k(
	retrieved_sources: list[str],
	relevant_sources: list[str],
	k: int,
) -> float:
	if k < 1:
		raise ValueError("k must be at least 1")

	if not relevant_sources:
		return 0.0

	retrieved = set(retrieved_sources[:k])
	relevant = set(relevant_sources)

	return len(retrieved & relevant) / len(relevant)


def mean_reciprocal_rank(
	retrieved_sources: list[list[str]],
	relevant_sources: list[list[str]],
) -> float:
	if len(retrieved_sources) != len(relevant_sources):
		raise ValueError(
			"retrieved_sources and relevant_sources must have the same length"
		)

	if not retrieved_sources:
		return 0.0

	reciprocal_ranks = []

	for retrieved, relevant in zip(
		retrieved_sources,
		relevant_sources,
	):
		relevant_set = set(relevant)

		rank = next(
			(
				index
				for index, source in enumerate(
					retrieved,
					start=1,
				)
				if source in relevant_set
			),
			None,
		)

		reciprocal_ranks.append(
			1.0 / rank if rank is not None else 0.0
		)

	return sum(reciprocal_ranks) / len(reciprocal_ranks)


def _content_words(text: str) -> set[str]:
	stop_words = {
		"a",
		"an",
		"and",
		"are",
		"as",
		"at",
		"be",
		"by",
		"for",
		"from",
		"in",
		"is",
		"it",
		"of",
		"on",
		"or",
		"that",
		"the",
		"this",
		"to",
		"was",
		"with",
	}

	words = {
		word.strip(".,!?;:()[]{}\"'").lower()
		for word in text.split()
	}

	return {
		word
		for word in words
		if word and word not in stop_words
	}


def supported_claim_ratio(
	answer: str,
	evidence: str,
) -> float:
	claims = [
		claim.strip()
		for claim in answer.split(".")
		if claim.strip()
	]

	if not claims:
		return 0.0

	evidence_words = _content_words(evidence)

	supported = 0

	for claim in claims:
		claim_words = _content_words(claim)

		if not claim_words:
			continue

		overlap = claim_words & evidence_words
		ratio = len(overlap) / len(claim_words)

		if ratio >= 0.5:
			supported += 1

	return supported / len(claims)


def answer_overlap_score(
    answer: str,
    reference_answer: str,
) -> float:
    answer_words = _content_words(answer)
    reference_words = _content_words(reference_answer)

    if not reference_words:
        return 0.0

    overlap = answer_words & reference_words

    return len(overlap) / len(reference_words)


def evaluate_answer_quality(
    answers: list[str],
    reference_answers: list[str],
) -> AnswerQualityMetrics:
    if not answers:
        return AnswerQualityMetrics(
            answer_overlap_score=0.0,
        )

    if len(answers) != len(reference_answers):
        raise ValueError(
            "answers and reference_answers must have the same length"
        )

    scores = [
        answer_overlap_score(answer, reference)
        for answer, reference in zip(
            answers,
            reference_answers,
        )
    ]

    return AnswerQualityMetrics(
        answer_overlap_score=sum(scores) / len(scores),
    )


def evaluate_faithfulness(
	answers: list[str],
	evidence: list[str],
) -> FaithfulnessMetrics:
	if len(answers) != len(evidence):
		raise ValueError(
			"answers and evidence must have the same length"
		)

	if not answers:
		return FaithfulnessMetrics(
			supported_claim_ratio=0.0,
		)

	scores = [
		supported_claim_ratio(answer, context)
		for answer, context in zip(
			answers,
			evidence,
		)
	]

	return FaithfulnessMetrics(
		supported_claim_ratio=sum(scores) / len(scores),
	)


def ndcg_at_k(
	retrieved_sources: list[str],
	relevant_sources: list[str],
	k: int,
) -> float:
	if k < 1:
		raise ValueError("k must be at least 1")

	if not relevant_sources:
		return 0.0

	retrieved = retrieved_sources[:k]
	relevant = set(relevant_sources)

	dcg = sum(
		1.0 / math.log2(index + 2)
		for index, source in enumerate(retrieved)
		if source in relevant
	)

	ideal_count = min(len(relevant), k)
	idcg = sum(
		1.0 / math.log2(index + 2)
		for index in range(ideal_count)
	)

	return dcg / idcg if idcg else 0.0


def evaluate_retrieval(
	retrieved_sources: list[list[str]],
	relevant_sources: list[list[str]],
	k: int,
) -> EvaluationMetrics:
	if len(retrieved_sources) != len(relevant_sources):
		raise ValueError(
			"retrieved_sources and relevant_sources must have the same length"
		)

	recall_scores = [
		recall_at_k(retrieved, relevant, k)
		for retrieved, relevant in zip(
			retrieved_sources,
			relevant_sources,
		)
	]

	recall = (
		sum(recall_scores) / len(recall_scores)
		if recall_scores
		else 0.0
	)

	return EvaluationMetrics(
		recall_at_k=recall,
		mean_reciprocal_rank=mean_reciprocal_rank(
			retrieved_sources,
			relevant_sources,
		),
		ndcg_at_k=sum(
			ndcg_at_k(retrieved, relevant, k)
			for retrieved, relevant in zip(
				retrieved_sources,
				relevant_sources,
			)
		) / len(retrieved_sources)
		if retrieved_sources
		else 0.0,
	)


def benchmark_pipeline(
	pipeline: ResearchPipeline,
	cases: list[EvaluationCase],
	k: int,
) -> EvaluationMetrics:
	retrieved_sources = []

	for case in cases:
		_, sources = pipeline._build_research_context(
			case.question,
			top_k=k,
		)

		retrieved_sources.append(
			[source.source for source in sources]
		)

	relevant_sources = [
		case.relevant_sources
		for case in cases
	]

	return evaluate_retrieval(
		retrieved_sources,
		relevant_sources,
		k,
	)


def benchmark_dataset(
	pipeline: ResearchPipeline,
	dataset_path: str | Path,
	k: int,
) -> EvaluationMetrics:
	cases = load_evaluation_dataset(dataset_path)

	return benchmark_pipeline(
		pipeline,
		cases,
		k,
	)


def benchmark_faithfulness(
    pipeline: ResearchPipeline,
    cases: list[EvaluationCase],
) -> FaithfulnessMetrics:
	answers = []
	evidence = []

	for case in cases:
		context, _ = pipeline._build_research_context(
			case.question,
		)

		answer = pipeline.llm.answer(
			case.question,
			context,
		)

		answers.append(answer)
		evidence.append(context)

	return evaluate_faithfulness(
		answers,
		evidence,
	)


def benchmark_answer_quality(
    pipeline,
    cases: list[EvaluationCase],
) -> AnswerQualityMetrics:
    answers = []
    reference_answers = []

    for case in cases:
        context, _ = pipeline._build_research_context(case.question)

        answer = pipeline.llm.answer(
            case.question,
            context,
        )

        answers.append(answer)
        reference_answers.append(case.reference_answer)

    return evaluate_answer_quality(
        answers,
        reference_answers,
    )