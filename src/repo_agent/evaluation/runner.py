"""Reproducible offline and explicitly gated live evaluation runners."""

import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from repo_agent.agent.service import AgentOrchestrationService
from repo_agent.analysis.service import RepositoryAnalysisService
from repo_agent.github.service import RepositoryRevision
from repo_agent.providers.base import ProviderAgentTurn, ProviderResult, ProviderToolCall
from repo_agent.providers.mock import MockLLMProvider
from repo_agent.rag.service import RepositoryRAGService
from repo_agent.schemas import RepositoryFile, RepositoryInspection, RepositoryOverview
from repo_agent.settings import Settings

from .models import (
    Benchmark,
    EvaluationCitation,
    EvaluationRecord,
    EvaluationReport,
    SourceReference,
)

OFFLINE_REPOSITORY_URL = "https://github.com/evaluation/github-ai-agent-fixture"
CHUNK_ID_PATTERN = re.compile(r"chunk_[0-9a-f]{20}")


class SnapshotRepository:
    """Serve files read from a pinned local Git commit; never accesses GitHub."""

    def __init__(self, inspection: RepositoryInspection) -> None:
        self.snapshot = inspection
        self.revision = RepositoryRevision(
            owner="evaluation",
            name="github-ai-agent-fixture",
            full_name=inspection.repository.full_name,
            default_branch="main",
            commit_sha=inspection.repository.commit_sha,
            description=inspection.repository.description,
            language=inspection.repository.language,
            stars=0,
            forks=0,
            html_url=str(inspection.repository.html_url),
        )

    async def current_revision(self, _repository_url: str) -> RepositoryRevision:
        return self.revision

    async def inspect(self, _repository_url: str, **_kwargs: Any) -> RepositoryInspection:
        return self.snapshot


class EvaluationProvider:
    """Count deterministic provider calls and simulate protocol-level agent turns."""

    provider_name = "offline_evaluation_mock"
    live_inference_enabled = False

    def __init__(self, *, mock_sufficient: bool = True) -> None:
        self.mock = MockLLMProvider()
        self.model_calls = 0
        self.mock_sufficient = mock_sufficient

    async def analyze_repository(self, context: str, *, max_output_tokens: int) -> ProviderResult:
        self.model_calls += 1
        return await self.mock.analyze_repository(context, max_output_tokens=max_output_tokens)

    async def answer_question(
        self, question: str, context: str, *, max_output_tokens: int, citation_mode: str = "paths"
    ) -> ProviderResult:
        self.model_calls += 1
        result = await self.mock.answer_question(
            question,
            context,
            max_output_tokens=max_output_tokens,
            citation_mode=citation_mode,  # type: ignore[arg-type]
        )
        if not self.mock_sufficient:
            return ProviderResult(
                answer="The available repository evidence is insufficient to establish this claim.",
                sufficient_context=False,
            )
        return result

    async def agent_turn(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        *,
        max_output_tokens: int,
        tool_choice: str = "auto",
    ) -> ProviderAgentTurn:
        del max_output_tokens
        self.model_calls += 1
        has_tool_result = any(message.get("role") == "tool" for message in messages)
        if tools and tool_choice == "auto" and not has_tool_result:
            latest_user = next(
                str(item.get("content", ""))
                for item in reversed(messages)
                if item.get("role") == "user"
            )
            question = latest_user.rsplit("Question: ", 1)[-1]
            return ProviderAgentTurn(
                tool_calls=[
                    ProviderToolCall(
                        id=f"eval-call-{self.model_calls}",
                        name="search_repository",
                        arguments={"query": question, "path": None},
                    )
                ]
            )
        ids = list(dict.fromkeys(CHUNK_ID_PATTERN.findall(json.dumps(messages))))
        sufficient = bool(ids) and self.mock_sufficient
        return ProviderAgentTurn(
            final_result=ProviderResult(
                answer=(
                    "Offline mock answer grounded in returned repository evidence."
                    if sufficient
                    else (
                        "The available repository evidence is insufficient to establish this claim."
                    )
                ),
                sufficient_context=sufficient,
                citation_chunk_ids=ids[:3] if sufficient else [],
            )
        )


def load_benchmark(path: Path) -> Benchmark:
    return Benchmark.model_validate_json(path.read_text(encoding="utf-8"))


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def load_snapshot(benchmark: Benchmark) -> RepositoryInspection:
    """Read only benchmark-referenced source files from their immutable local commit."""
    paths = sorted(
        {reference.path for case in benchmark.cases for reference in case.expected_sources}
    )
    files: list[RepositoryFile] = []
    for path in paths:
        content = _git_output("show", f"{benchmark.repository_commit}:{path}")
        blob_sha = _git_output("rev-parse", f"{benchmark.repository_commit}:{path}").strip()
        files.append(
            RepositoryFile(
                path=path,
                blob_sha=blob_sha,
                commit_sha=benchmark.repository_commit,
                size_bytes=len(content.encode("utf-8")),
                content=content,
                citation_url=(
                    f"https://github.com/evaluation/github-ai-agent-fixture/blob/"
                    f"{benchmark.repository_commit}/{path}"
                ),
            )
        )
    for case in benchmark.cases:
        for source in case.expected_sources:
            file_content = next(file.content for file in files if file.path == source.path)
            _verify_reference(source, file_content)
    overview = RepositoryOverview(
        name="github-ai-agent-fixture",
        full_name="evaluation/github-ai-agent-fixture",
        description="Local immutable source fixture for offline evaluation.",
        html_url="https://github.com/evaluation/github-ai-agent-fixture",
        stars=0,
        forks=0,
        language="Python",
        default_branch="main",
        commit_sha=benchmark.repository_commit,
        tree_truncated=False,
        files_in_tree=len(files),
        files_selected=len(files),
        source_bytes=sum(file.size_bytes for file in files),
    )
    return RepositoryInspection(repository=overview, files=files)


def _verify_reference(reference: SourceReference, content: str) -> None:
    lines = content.splitlines()
    if reference.end_line > len(lines):
        raise ValueError(f"Ground-truth range is outside source file: {reference.path}")
    excerpt = "\n".join(lines[reference.start_line - 1 : reference.end_line])
    if reference.symbol not in excerpt:
        raise ValueError(
            f"Ground-truth symbol {reference.symbol!r} is absent from "
            f"{reference.path}:{reference.start_line}-{reference.end_line}"
        )


def _reference_record(
    case_id: str,
    mode: str,
    question: str,
    expected_sufficient: bool,
    expected_sources: list[SourceReference],
    answer: str,
    observed_commit: str,
    retrieved: list[Any],
    prompt: list[Any],
    citations: list[Any],
    *,
    model_calls: int,
    tool_calls: int = 0,
    estimated_tokens: int | None = None,
    context_budget: int | None = None,
    error_category: str | None = None,
    termination_reason: str | None = None,
    context_sufficient: bool | None = None,
) -> EvaluationRecord:
    retrieved_ids = [str(item.chunk_id) for item in retrieved if getattr(item, "chunk_id", None)]
    prompt_ids = [str(item.chunk_id) for item in prompt if getattr(item, "chunk_id", None)]
    citation_ids = [str(item.chunk_id) for item in citations if getattr(item, "chunk_id", None)]
    cited_files = [str(item.path) for item in citations if getattr(item, "path", None)]
    citation_evidence = [
        EvaluationCitation(
            chunk_id=getattr(item, "chunk_id", None),
            path=str(item.path),
            start_line=getattr(item, "start_line", None),
            end_line=getattr(item, "end_line", None),
            commit_sha=getattr(item, "commit_sha", None),
        )
        for item in citations
        if getattr(item, "path", None)
    ]
    retrieved_files = list(dict.fromkeys(str(item.path) for item in retrieved if item.path))
    prompt_by_id = {str(item.chunk_id): item for item in prompt if getattr(item, "chunk_id", None)}
    citation_valid: bool | None = None
    citation_relevant: bool | None = None
    if mode in {"rag", "agent"}:
        allowed_by_id = (
            prompt_by_id
            if mode == "rag"
            else {str(item.chunk_id): item for item in retrieved if getattr(item, "chunk_id", None)}
        )
        citation_valid = (
            (not expected_sufficient)
            if not citations
            else all(
                getattr(citation, "chunk_id", None) in allowed_by_id
                and (source := allowed_by_id.get(str(getattr(citation, "chunk_id", ""))))
                is not None
                and getattr(citation, "path", None) == source.path
                and getattr(citation, "start_line", None) == source.start_line
                and getattr(citation, "end_line", None) == source.end_line
                and getattr(citation, "commit_sha", None) == source.commit_sha
                for citation in citations
            )
        )
        citation_relevant = (
            (
                bool(citation_ids)
                and any(
                    any(
                        item.path == source.path
                        and item.start_line <= source.end_line
                        and item.end_line >= source.start_line
                        for item in (prompt_by_id.get(chunk_id) for chunk_id in citation_ids)
                        if item is not None
                    )
                    for source in expected_sources
                )
            )
            if expected_sources
            else (False if citation_ids else None)
        )
    expected_files = sorted({source.path for source in expected_sources})
    expected_files_retrieved = [path for path in expected_files if path in retrieved_files]
    expected_symbols = [
        source.symbol
        for source in expected_sources
        if any(
            getattr(item, "path", None) == source.path
            and getattr(item, "start_line", 0) <= source.end_line
            and getattr(item, "end_line", 0) >= source.start_line
            for item in retrieved
        )
    ]
    unique_count = len(set(retrieved_ids))
    duplicate_count = max(0, len(retrieved_ids) - unique_count)
    usage = (
        estimated_tokens / context_budget
        if estimated_tokens is not None and context_budget
        else None
    )
    return EvaluationRecord(
        case_id=case_id,
        mode=mode,  # type: ignore[arg-type]
        question=question,
        repository_commit=observed_commit,
        expected_sources=expected_sources,
        observed_commit=observed_commit,
        completed=error_category is None
        or error_category in {"insufficient_repository_evidence", "agent_insufficient_evidence"},
        expected_sufficient=expected_sufficient,
        context_sufficient=context_sufficient,
        sufficiency_match=(
            context_sufficient == expected_sufficient if context_sufficient is not None else None
        ),
        answer=answer,
        error_category=error_category,
        termination_reason=termination_reason,
        model_calls=model_calls,
        tool_calls=tool_calls,
        retrieved_chunk_ids=retrieved_ids,
        prompt_chunk_ids=prompt_ids,
        citation_chunk_ids=citation_ids,
        citations=citation_evidence,
        retrieved_files=retrieved_files,
        cited_files=cited_files,
        expected_files_retrieved=expected_files_retrieved,
        expected_symbols_retrieved=expected_symbols,
        citation_valid=citation_valid,
        citation_relevant=citation_relevant,
        unique_chunks=unique_count,
        duplicate_chunks=duplicate_count,
        estimated_tokens=estimated_tokens,
        context_budget_tokens=context_budget,
        context_budget_usage=usage,
    )


async def run_offline(
    benchmark: Benchmark,
    *,
    context_budget: int = 8_000,
    sqlite_path: str = "data/evaluation.sqlite3",
) -> EvaluationReport:
    snapshot = load_snapshot(benchmark)
    settings = Settings(
        _env_file=None,
        llm_max_context_tokens=context_budget,
        agent_enabled=True,
        agent_max_context_tokens=context_budget,
        rag_sqlite_path=sqlite_path,
    )
    repository = SnapshotRepository(snapshot)
    records: list[EvaluationRecord] = []

    # The architecture task exercises the repository analysis endpoint once.
    analysis_case = benchmark.cases[0]
    analysis_provider = EvaluationProvider()
    analysis_service = RepositoryAnalysisService(repository, analysis_provider, settings)  # type: ignore[arg-type]
    try:
        answer = await analysis_service.analyze(OFFLINE_REPOSITORY_URL)
        record = EvaluationRecord(
            case_id=analysis_case.id,
            mode="analysis",
            question=analysis_case.question,
            repository_commit=benchmark.repository_commit,
            expected_sources=analysis_case.expected_sources,
            observed_commit=answer.commit_sha,
            completed=True,
            expected_sufficient=True,
            context_sufficient=answer.context_sufficient,
            sufficiency_match=answer.context_sufficient == analysis_case.expected_sufficient,
            answer=answer.answer,
            model_calls=analysis_provider.model_calls,
            retrieved_files=[str(citation.path) for citation in answer.citations],
            cited_files=[str(citation.path) for citation in answer.citations],
            citations=[
                EvaluationCitation(
                    path=str(citation.path),
                    start_line=citation.start_line,
                    end_line=citation.end_line,
                    commit_sha=citation.commit_sha,
                )
                for citation in answer.citations
            ],
            expected_files_retrieved=[
                item.path
                for item in analysis_case.expected_sources
                if item.path in {str(citation.path) for citation in answer.citations}
            ],
            citation_valid=all(
                citation.commit_sha == answer.commit_sha for citation in answer.citations
            ),
            citation_relevant=any(
                str(citation.path) == expected.path
                for expected in analysis_case.expected_sources
                for citation in answer.citations
            ),
            estimated_tokens=answer.context_tokens,
            context_budget_tokens=settings.llm_max_context_tokens,
            context_budget_usage=answer.context_tokens / settings.llm_max_context_tokens,
        )
        records.append(record)
    except Exception as exc:
        records.append(
            EvaluationRecord(
                case_id=analysis_case.id,
                mode="analysis",
                question=analysis_case.question,
                repository_commit=benchmark.repository_commit,
                expected_sources=analysis_case.expected_sources,
                completed=False,
                expected_sufficient=True,
                model_calls=analysis_provider.model_calls,
                error_category=type(exc).__name__,
            )
        )

    for case in benchmark.cases:
        for mode in case.modes:
            provider = EvaluationProvider(mock_sufficient=case.expected_sufficient)
            rag = RepositoryRAGService(repository, provider, settings)  # type: ignore[arg-type]
            try:
                if mode == "rag":
                    response = await rag.answer(OFFLINE_REPOSITORY_URL, case.question)
                    retrieved = response.retrieved_chunks
                    prompt = response.prompt_chunks
                    citations = response.citations
                    record = _reference_record(
                        case.id,
                        mode,
                        case.question,
                        case.expected_sufficient,
                        case.expected_sources,
                        response.answer,
                        response.commit_sha,
                        retrieved,
                        prompt,
                        citations,
                        model_calls=provider.model_calls,
                        estimated_tokens=response.context_tokens,
                        context_budget=settings.llm_max_context_tokens,
                        error_category=response.error_code
                        if response.error_code not in {None, "insufficient_repository_evidence"}
                        else None,
                        termination_reason=response.error_code,
                        context_sufficient=response.context_sufficient,
                    )
                else:
                    agent = AgentOrchestrationService(  # type: ignore[arg-type]
                        repository, provider, rag, settings
                    )
                    response = await agent.answer(OFFLINE_REPOSITORY_URL, case.question)
                    retrieved = [
                        reference for entry in response.trace for reference in entry.references
                    ]
                    prompt = retrieved
                    citations = response.citations
                    diagnostics = response.diagnostics
                    record = _reference_record(
                        case.id,
                        mode,
                        case.question,
                        case.expected_sufficient,
                        case.expected_sources,
                        response.answer,
                        response.commit_sha,
                        retrieved,
                        prompt,
                        citations,
                        model_calls=provider.model_calls,
                        tool_calls=response.tool_calls,
                        estimated_tokens=(
                            diagnostics.estimated_context_tokens if diagnostics else None
                        ),
                        context_budget=settings.agent_max_context_tokens,
                        error_category=response.error_code,
                        termination_reason=(
                            diagnostics.termination_reason if diagnostics else None
                        ),
                        context_sufficient=response.context_sufficient,
                    )
                    if diagnostics is not None:
                        record.unique_chunks = diagnostics.unique_evidence_chunks
                        record.duplicate_chunks = diagnostics.duplicate_evidence_chunks
                record.repository_commit = benchmark.repository_commit
                records.append(record)
            except Exception as exc:
                records.append(
                    EvaluationRecord(
                        case_id=case.id,
                        mode=mode,
                        question=case.question,
                        repository_commit=benchmark.repository_commit,
                        completed=False,
                        expected_sufficient=case.expected_sufficient,
                        model_calls=provider.model_calls,
                        error_category=type(exc).__name__,
                    )
                )
    return EvaluationReport(
        benchmark=benchmark.name,
        mode="offline_mock",
        repository_commit=benchmark.repository_commit,
        configuration={
            "provider": "deterministic_mock",
            "live_requests": False,
            "context_budget_tokens": context_budget,
            "agent_max_steps": settings.agent_max_steps,
            "agent_max_tool_calls": settings.agent_max_tool_calls,
            "agent_tool_max_chunks": settings.agent_tool_max_chunks,
            "agent_tool_max_response_chars": settings.agent_tool_max_response_chars,
            "llm_max_context_tokens": settings.llm_max_context_tokens,
            "llm_max_output_tokens": settings.llm_max_output_tokens,
            "rag_retrieved_chunks": settings.rag_retrieved_chunks,
            "rag_chunk_lines": settings.rag_chunk_lines,
            "rag_chunk_overlap_lines": settings.rag_chunk_overlap_lines,
            "fixture_repository": OFFLINE_REPOSITORY_URL,
        },
        records=records,
    )


def run_live(
    benchmark: Benchmark,
    *,
    api_base_url: str,
    repository_url: str,
    approve_paid_api: bool,
) -> EvaluationReport:
    """Call an explicitly selected running API only after both live consent flags."""
    if not approve_paid_api:
        raise ValueError("Live evaluation requires --approve-paid-api in addition to --live.")
    records: list[EvaluationRecord] = []
    with httpx.Client(base_url=api_base_url.rstrip("/"), timeout=120) as client:
        analysis_case = benchmark.cases[0]
        analysis_response = client.post(
            "/api/repos/analyze", json={"repository_url": repository_url}
        )
        if analysis_response.is_error:
            body = (
                analysis_response.json()
                if "application/json" in analysis_response.headers.get("content-type", "")
                else {}
            )
            records.append(
                EvaluationRecord(
                    case_id=analysis_case.id,
                    mode="analysis",
                    question=analysis_case.question,
                    repository_commit=benchmark.repository_commit,
                    expected_sources=analysis_case.expected_sources,
                    completed=False,
                    expected_sufficient=True,
                    model_calls=0,
                    error_category=str(body.get("code", f"http_{analysis_response.status_code}")),
                )
            )
        else:
            body = analysis_response.json()
            citations = body.get("citations", [])
            cited_paths = {str(item.get("path", "")) for item in citations}
            observed_commit = str(body.get("commit_sha", ""))
            record = EvaluationRecord(
                case_id=analysis_case.id,
                mode="analysis",
                question=analysis_case.question,
                repository_commit=benchmark.repository_commit,
                expected_sources=analysis_case.expected_sources,
                observed_commit=observed_commit,
                completed=observed_commit == benchmark.repository_commit,
                expected_sufficient=True,
                context_sufficient=body.get("context_sufficient"),
                sufficiency_match=body.get("context_sufficient") is True,
                answer=str(body.get("answer", "")),
                error_category=body.get("error_code"),
                model_calls=1,
                retrieved_files=sorted(cited_paths),
                cited_files=sorted(cited_paths),
                citations=[
                    EvaluationCitation(
                        path=str(item.get("path", "")),
                        start_line=item.get("start_line"),
                        end_line=item.get("end_line"),
                        commit_sha=item.get("commit_sha"),
                    )
                    for item in citations
                    if item.get("path")
                ],
                expected_files_retrieved=[
                    source.path
                    for source in analysis_case.expected_sources
                    if source.path in cited_paths
                ],
                citation_valid=all(
                    str(item.get("commit_sha", "")) == observed_commit for item in citations
                ),
                citation_relevant=any(
                    source.path in cited_paths for source in analysis_case.expected_sources
                ),
                estimated_tokens=int(body.get("context_tokens", 0)) or None,
            )
            if record.observed_commit != benchmark.repository_commit:
                record.completed = False
                record.error_category = "benchmark_commit_mismatch"
            records.append(record)
        for case in benchmark.cases:
            for mode in case.modes:
                endpoint = "/api/repos/ask" if mode == "rag" else "/api/repos/agent"
                response = client.post(
                    endpoint,
                    json={"repository_url": repository_url, "question": case.question},
                )
                if response.is_error:
                    body = (
                        response.json()
                        if "application/json" in response.headers.get("content-type", "")
                        else {}
                    )
                    records.append(
                        EvaluationRecord(
                            case_id=case.id,
                            mode=mode,
                            question=case.question,
                            repository_commit=benchmark.repository_commit,
                            completed=False,
                            expected_sufficient=case.expected_sufficient,
                            error_category=str(body.get("code", f"http_{response.status_code}")),
                        )
                    )
                    continue
                body = response.json()
                retrieved = body.get("retrieved_chunks", [])
                prompt = body.get("prompt_chunks", [])
                if mode == "agent":
                    retrieved = [
                        ref
                        for trace in body.get("trace", [])
                        for ref in trace.get("references", [])
                    ]
                    prompt = retrieved
                citations = body.get("citations", [])
                record = _reference_record(
                    case.id,
                    mode,
                    case.question,
                    case.expected_sufficient,
                    case.expected_sources,
                    str(body.get("answer", "")),
                    str(body.get("commit_sha", "")),
                    [SimpleReference.model_validate(item) for item in retrieved],
                    [SimpleReference.model_validate(item) for item in prompt],
                    [SimpleReference.model_validate(item) for item in citations],
                    model_calls=(
                        int(body.get("steps", 0))
                        if mode == "agent"
                        else int(bool(body.get("prompt_chunks")))
                    ),
                    tool_calls=int(body.get("tool_calls", 0)),
                    estimated_tokens=int(body.get("context_tokens", 0)) or None,
                    context_budget=None,
                    error_category=body.get("error_code"),
                    termination_reason=(body.get("diagnostics") or {}).get("termination_reason"),
                    context_sufficient=body.get("context_sufficient"),
                )
                record.repository_commit = benchmark.repository_commit
                record.observed_commit = str(body.get("commit_sha", "")) or None
                if record.observed_commit != benchmark.repository_commit:
                    record.completed = False
                    record.error_category = "benchmark_commit_mismatch"
                records.append(record)
    return EvaluationReport(
        benchmark=benchmark.name,
        mode="live",
        repository_commit=benchmark.repository_commit,
        configuration={
            "api_base_url": api_base_url,
            "repository_url": repository_url,
            "live_requests": True,
            "paid_api_approved": approve_paid_api,
        },
        records=records,
    )


class SimpleReference:
    """Small adapter for the corresponding safe JSON source-reference fields."""

    def __init__(self, value: dict[str, Any]) -> None:
        self.chunk_id = value.get("chunk_id")
        self.path = value.get("path")
        self.start_line = value.get("start_line", 0)
        self.end_line = value.get("end_line", 0)
        self.commit_sha = value.get("commit_sha")

    @classmethod
    def model_validate(cls, value: dict[str, Any]) -> "SimpleReference":
        return cls(value)


def summarize(report: EvaluationReport) -> str:
    reviewed = [row for row in report.records if row.human_review.rating is not None]
    human_scores = {
        "correct": 1.0,
        "partially_correct": 0.5,
        "incorrect": 0.0,
    }
    reviewed_score = (
        sum(human_scores[row.human_review.rating] for row in reviewed) / len(reviewed)
        if reviewed
        else None
    )
    lines = [
        f"# Evaluation summary: {report.benchmark}",
        "",
        f"- Mode: `{report.mode}`",
        f"- Ground-truth commit: `{report.repository_commit}`",
        f"- Cases: {len(report.records)}",
        "",
        "## Offline pipeline reliability",
        "",
        "| "
        + " | ".join(
            [
                "Mode",
                "Runs",
                "Completed",
                "Sufficiency match",
                "Model calls",
                "Tool calls",
                "Unique chunks",
                "Duplicate chunks",
                "Est. tokens",
            ]
        )
        + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("analysis", "rag", "agent"):
        rows = [record for record in report.records if record.mode == mode]
        if not rows:
            continue
        values = (
            mode,
            len(rows),
            sum(row.completed for row in rows),
            sum(row.sufficiency_match is True for row in rows),
            sum(row.model_calls for row in rows),
            sum(row.tool_calls for row in rows),
            sum(row.unique_chunks for row in rows),
            sum(row.duplicate_chunks for row in rows),
            sum(row.estimated_tokens or 0 for row in rows),
        )
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "## Citation integrity and mechanical relevance",
            "",
            "Relevance excludes cases with no expected sources (shown as N/A in the denominator). "
            "Integrity checks that citations resolve only to request-supplied evidence.",
            "",
            "| Mode | Runs | Integrity passed | Relevant / applicable |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode in ("analysis", "rag", "agent"):
        rows = [record for record in report.records if record.mode == mode]
        if not rows:
            continue
        applicable = [row for row in rows if row.citation_relevant is not None]
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    str(len(rows)),
                    f"{sum(row.citation_valid is True for row in rows)}/{len(rows)}",
                    f"{sum(row.citation_relevant is True for row in applicable)}/{len(applicable)}",
                ]
            )
            + " |"
        )
    failures = Counter(
        row.error_category for row in report.records if row.error_category is not None
    )
    terminations = Counter(
        row.termination_reason for row in report.records if row.termination_reason is not None
    )
    lines.extend(
        [
            "",
            "## Actual LLM accuracy",
            "",
            (
                f"Human-reviewed score: {reviewed_score:.1%} across {len(reviewed)} rows."
                if reviewed_score is not None
                else (
                    "Not measured. Human review is pending; offline mocks do not establish "
                    "model accuracy."
                )
            ),
            "",
            "## Failure and termination categories",
            "",
            "- Error categories: "
            + (", ".join(f"{key} ({value})" for key, value in sorted(failures.items())) or "none"),
            "- Termination reasons: "
            + (
                ", ".join(f"{key} ({value})" for key, value in sorted(terminations.items()))
                or "none"
            ),
            "Correctness requires human review against each benchmark rubric. Citation and "
            "retrieval metrics are mechanical checks against pinned references; they do not "
            "establish semantic correctness.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(report: EvaluationReport, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.mode}-{report.repository_commit[:12]}"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(summarize(report), encoding="utf-8")
    rows = [record.model_dump(mode="json") for record in report.records]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EvaluationRecord.model_fields))
        writer.writeheader()
        for row in rows:
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, separators=(",", ":"))
            writer.writerow(row)
    return json_path, markdown_path, csv_path
