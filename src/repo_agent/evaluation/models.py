"""Schemas for benchmark cases, evaluation results, and human review."""

from typing import Literal

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    path: str
    symbol: str
    start_line: int = Field(gt=0)
    end_line: int = Field(gt=0)


class BenchmarkCase(BaseModel):
    id: str
    question: str
    modes: list[Literal["rag", "agent"]] = ["rag", "agent"]
    expected_sufficient: bool
    expected_sources: list[SourceReference] = Field(default_factory=list)
    review_rubric: list[str] = Field(min_length=1)


class Benchmark(BaseModel):
    schema_version: int = 1
    name: str
    repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    cases: list[BenchmarkCase] = Field(min_length=1)


class HumanReview(BaseModel):
    rating: Literal["correct", "partially_correct", "incorrect"] | None = None
    notes: str = ""


class EvaluationCitation(BaseModel):
    chunk_id: str | None = None
    path: str
    start_line: int | None = None
    end_line: int | None = None
    commit_sha: str | None = None


class EvaluationRecord(BaseModel):
    case_id: str
    mode: Literal["analysis", "rag", "agent"]
    question: str
    repository_commit: str
    expected_sources: list[SourceReference] = Field(default_factory=list)
    observed_commit: str | None = None
    completed: bool
    expected_sufficient: bool
    context_sufficient: bool | None = None
    sufficiency_match: bool | None = None
    answer: str = ""
    error_category: str | None = None
    termination_reason: str | None = None
    model_calls: int = 0
    tool_calls: int = 0
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    prompt_chunk_ids: list[str] = Field(default_factory=list)
    citation_chunk_ids: list[str] = Field(default_factory=list)
    citations: list[EvaluationCitation] = Field(default_factory=list)
    retrieved_files: list[str] = Field(default_factory=list)
    cited_files: list[str] = Field(default_factory=list)
    expected_files_retrieved: list[str] = Field(default_factory=list)
    expected_symbols_retrieved: list[str] = Field(default_factory=list)
    citation_valid: bool | None = None
    citation_relevant: bool | None = None
    unique_chunks: int = 0
    duplicate_chunks: int = 0
    estimated_tokens: int | None = None
    context_budget_tokens: int | None = None
    context_budget_usage: float | None = None
    human_review: HumanReview = Field(default_factory=HumanReview)


class EvaluationReport(BaseModel):
    schema_version: int = 1
    benchmark: str
    mode: Literal["offline_mock", "live"]
    repository_commit: str
    configuration: dict[str, str | int | bool | None]
    records: list[EvaluationRecord]
