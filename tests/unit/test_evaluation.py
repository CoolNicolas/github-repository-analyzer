import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from repo_agent.evaluation.cli import _merge_reviews
from repo_agent.evaluation.models import Benchmark, HumanReview, SourceReference
from repo_agent.evaluation.runner import (
    _reference_record,
    load_benchmark,
    load_snapshot,
    run_live,
    run_offline,
    summarize,
    write_report,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "evaluation" / "benchmark.json"


def test_benchmark_is_pinned_and_ground_truth_references_are_verified() -> None:
    benchmark = load_benchmark(BENCHMARK_PATH)
    snapshot = load_snapshot(benchmark)

    assert benchmark.repository_commit == "a2da9850fdd1fe8a9bb4ae016a30b3f662f90fa2"
    assert len(snapshot.files) >= 5
    assert {case.expected_sufficient for case in benchmark.cases} == {True, False}
    assert any(case.id == "configuration-bounds" for case in benchmark.cases)


@pytest.mark.asyncio
async def test_offline_evaluation_runs_analysis_rag_agent_and_negative_case(tmp_path) -> None:
    benchmark = load_benchmark(BENCHMARK_PATH)
    report = await run_offline(
        benchmark,
        context_budget=8_000,
        sqlite_path=str(tmp_path / "evaluation.sqlite3"),
    )

    assert report.mode == "offline_mock"
    assert {record.mode for record in report.records} == {"analysis", "rag", "agent"}
    assert len(report.records) == 1 + sum(len(case.modes) for case in benchmark.cases)
    assert all(record.repository_commit == benchmark.repository_commit for record in report.records)
    assert all(
        record.sufficiency_match is True for record in report.records if record.mode != "analysis"
    )
    negative = [
        record for record in report.records if record.case_id == "negative-missing-evidence"
    ]
    assert len(negative) == 2
    assert all(record.context_sufficient is False for record in negative)
    assert all(not record.citation_chunk_ids for record in negative)
    assert all(record.citation_relevant is None for record in negative)
    assert all(record.model_calls <= 1 for record in negative if record.mode == "rag")
    assert any(record.tool_calls > 0 for record in report.records if record.mode == "agent")
    assert all(
        record.citation_valid is True
        for record in report.records
        if record.mode in {"rag", "agent"}
    )
    assert all(record.citations for record in report.records if record.citation_chunk_ids)


@pytest.mark.asyncio
async def test_known_citation_relevance_mismatches_and_missing_case_denominator(tmp_path) -> None:
    benchmark = load_benchmark(BENCHMARK_PATH)
    report = await run_offline(benchmark, sqlite_path=str(tmp_path / "audit.sqlite3"))
    failures = {
        (record.case_id, record.mode)
        for record in report.records
        if record.mode in {"rag", "agent"} and record.citation_relevant is False
    }
    assert failures == {
        ("configuration-bounds", "rag"),
        ("configuration-bounds", "agent"),
        ("agent-rag-relationship", "rag"),
        ("agent-rag-relationship", "agent"),
    }
    config_rag = next(
        row for row in report.records if row.case_id == "configuration-bounds" and row.mode == "rag"
    )
    assert config_rag.expected_sources[0].path == "src/repo_agent/settings.py"
    assert "src/repo_agent/settings.py" not in config_rag.retrieved_files
    assert {citation.path for citation in config_rag.citations} == {
        "src/repo_agent/github/service.py"
    }
    relationship_rag = next(
        row
        for row in report.records
        if row.case_id == "agent-rag-relationship" and row.mode == "rag"
    )
    assert relationship_rag.expected_symbols_retrieved == [
        "AgentOrchestrationService",
        "RepositoryRAGService",
    ]
    assert [
        (citation.path, citation.start_line, citation.end_line)
        for citation in relationship_rag.citations
    ] == [
        ("src/repo_agent/agent/service.py", 1, 80),
        ("src/repo_agent/agent/service.py", 651, 730),
        ("src/repo_agent/agent/service.py", 976, 1036),
    ]
    negative = [row for row in report.records if row.case_id == "negative-missing-evidence"]
    assert all(row.citation_relevant is None for row in negative)
    assert "3/5" in summarize(report)


def test_citation_integrity_checks_resolved_path_and_line_range() -> None:
    benchmark = load_benchmark(BENCHMARK_PATH)
    source = benchmark.cases[0].expected_sources[0]
    prompt = [
        SimpleNamespace(
            chunk_id="chunk_" + "a" * 20,
            path=source.path,
            start_line=206,
            end_line=282,
            commit_sha=benchmark.repository_commit,
        )
    ]
    citation = SimpleNamespace(
        chunk_id=prompt[0].chunk_id,
        path=source.path,
        start_line=207,
        end_line=282,
        commit_sha=benchmark.repository_commit,
    )

    result = _reference_record(
        "audit",
        "rag",
        "question",
        True,
        [source],
        "answer",
        benchmark.repository_commit,
        prompt,
        prompt,
        [citation],
        model_calls=1,
    )

    assert result.citation_valid is False


def test_report_outputs_json_csv_and_markdown_without_claiming_mock_accuracy(tmp_path) -> None:
    benchmark = load_benchmark(BENCHMARK_PATH)
    import asyncio

    report = asyncio.run(run_offline(benchmark, sqlite_path=str(tmp_path / "index.sqlite3")))
    json_path, markdown_path, csv_path = write_report(report, tmp_path / "reports")

    assert (
        json.loads(json_path.read_text(encoding="utf-8"))["repository_commit"]
        == benchmark.repository_commit
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "RAG" in markdown or "rag" in markdown
    assert "offline mocks do not establish model accuracy" in markdown
    assert "case_id" in csv_path.read_text(encoding="utf-8")
    generated_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (json_path, markdown_path, csv_path)
    )
    for sensitive_pattern in (
        r"\bgh[pousr]_[A-Za-z0-9]{20,}\b",
        r"\bsk-[A-Za-z0-9_-]{20,}\b",
        r"(?i)\bBearer\s+\S+",
        r"(?i)[A-Z]:\\Users\\",
        r"/(?:Users|home)/[^/\s]+",
    ):
        assert re.search(sensitive_pattern, generated_text) is None
    assert "Human-reviewed correctness score" not in summarize(report)
    reviewed = report.model_copy(deep=True)
    reviewed.records[0].human_review = HumanReview(rating="correct", notes="Matches rubric.")
    assert "Human-reviewed score: 100.0% across 1 rows." in summarize(reviewed)
    reviews_path = tmp_path / "reviews.json"
    reviews_path.write_text(
        json.dumps(
            {
                f"{report.records[0].case_id}:{report.records[0].mode}": {
                    "rating": "correct",
                    "notes": "Matches rubric.",
                }
            }
        ),
        encoding="utf-8",
    )
    _merge_reviews(json_path, reviews_path)
    assert "Human-reviewed score: 100.0% across 1 rows." in markdown_path.read_text(
        encoding="utf-8"
    )
    assert "correct" in csv_path.read_text(encoding="utf-8")


def test_live_runner_requires_explicit_paid_api_approval() -> None:
    benchmark = load_benchmark(BENCHMARK_PATH)

    with pytest.raises(ValueError, match="requires --approve-paid-api"):
        run_live(
            benchmark,
            api_base_url="http://127.0.0.1:8000",
            repository_url="https://github.com/o/r",
            approve_paid_api=False,
        )


def test_invalid_source_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="Ground-truth symbol"):
        from repo_agent.evaluation.runner import _verify_reference

        _verify_reference(
            SourceReference(
                path="src/example.py", symbol="MissingSymbol", start_line=1, end_line=1
            ),
            "class Present: pass",
        )


def test_benchmark_schema_rejects_missing_review_criteria() -> None:
    with pytest.raises(ValueError):
        Benchmark.model_validate(
            {
                "name": "bad",
                "repository_commit": "a" * 40,
                "cases": [{"id": "x", "question": "q", "expected_sufficient": False}],
            }
        )
