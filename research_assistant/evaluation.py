import json
import math
from dataclasses import dataclass
from pathlib import Path

from research_assistant.pipeline import ResearchPipeline


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    relevant_sources: list[str]


@dataclass(frozen=True)
class EvaluationMetrics:
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float


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