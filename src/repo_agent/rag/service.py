"""Index bounded repository snapshots and answer questions from retrieved chunks."""

import math
import re
from dataclasses import dataclass
from urllib.parse import quote

from repo_agent.analysis.context import extract_file_references, file_matches_reference
from repo_agent.github.service import GitHubRepositoryService, RepositoryRevision
from repo_agent.github.urls import parse_repository_url
from repo_agent.providers.base import LLMProvider, ProviderDiagnostics, ProviderResult
from repo_agent.rag.chunking import chunk_repository_files
from repo_agent.rag.index import IndexedChunk, SQLiteFTSIndex
from repo_agent.schemas import (
    RAGChunkReference,
    RepositoryAnalysisResponse,
    RepositoryInspection,
    SourceCitation,
)
from repo_agent.settings import Settings

CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass(frozen=True)
class RAGContext:
    text: str
    indexed_files: tuple[str, ...]
    retrieved_chunks: tuple[IndexedChunk, ...]
    chunks: tuple[IndexedChunk, ...]
    estimated_tokens: int


class RepositoryRAGService:
    """Build commit-keyed local FTS indexes and run bounded retrieval-augmented Q&A."""

    def __init__(
        self,
        repository_service: GitHubRepositoryService,
        provider: LLMProvider,
        settings: Settings,
        index: SQLiteFTSIndex | None = None,
    ) -> None:
        self.repository_service = repository_service
        self.provider = provider
        self.settings = settings
        self.index = index or SQLiteFTSIndex(settings.rag_sqlite_path)

    async def answer(self, repository_url: str, question: str) -> RepositoryAnalysisResponse:
        references = extract_file_references(question)
        reference = parse_repository_url(repository_url)
        revision = await self.repository_service.current_revision(repository_url)
        is_indexed = self.index.has_index(revision.owner, revision.name, revision.commit_sha)
        indexed_paths = self.index.paths(revision.owner, revision.name, revision.commit_sha)
        missing_references = [
            name
            for name in references
            if not any(file_matches_reference(path, name) for path in indexed_paths)
        ]

        if not is_indexed or missing_references:
            snapshot = await self._fetch_index_snapshot(repository_url, references)
            snapshot_revision = self._revision_from_inspection(
                reference.owner, reference.name, snapshot
            )
            snapshot_paths = {source_file.path for source_file in snapshot.files}
            requested_files_found = all(
                any(file_matches_reference(path, name) for path in snapshot_paths)
                for name in references
            )
            if (
                not is_indexed
                or snapshot_revision.commit_sha != revision.commit_sha
                or requested_files_found
            ):
                self._replace_index(snapshot_revision, snapshot)
                revision = snapshot_revision
            else:
                revision = snapshot_revision
                indexed_paths = self.index.paths(revision.owner, revision.name, revision.commit_sha)

        indexed_paths = self.index.paths(revision.owner, revision.name, revision.commit_sha)
        unavailable_references = [
            name
            for name in references
            if not any(file_matches_reference(path, name) for path in indexed_paths)
        ]
        if unavailable_references:
            return self._insufficient(
                revision,
                "The requested source file(s) are not present in the bounded repository index: "
                + ", ".join(unavailable_references)
                + ".",
                indexed_files=indexed_paths,
            )

        retrieved = self.index.retrieve(
            revision.owner,
            revision.name,
            revision.commit_sha,
            question,
            file_references=references,
            limit=self.settings.rag_retrieved_chunks,
        )
        context = self._build_context(
            revision,
            retrieved,
            indexed_files=indexed_paths,
            priority_paths=references,
        )
        if references and any(
            not any(file_matches_reference(chunk.path, name) for chunk in context.chunks)
            for name in references
        ):
            return self._insufficient(
                revision,
                "The requested source file could not fit in the configured retrieval context.",
                context_tokens=context.estimated_tokens,
                indexed_files=indexed_paths,
                retrieved_chunks=context.retrieved_chunks,
                prompt_chunks=context.chunks,
            )
        if not context.chunks:
            return self._insufficient(
                revision,
                "No relevant repository chunks were retrieved for this question.",
                context_tokens=context.estimated_tokens,
                indexed_files=indexed_paths,
                retrieved_chunks=context.retrieved_chunks,
                prompt_chunks=context.chunks,
            )

        result = await self.provider.answer_question(
            question,
            context.text,
            max_output_tokens=self.settings.llm_max_output_tokens,
        )
        return self._response(revision, context, result, question=question)

    async def _fetch_index_snapshot(
        self, repository_url: str, priority_paths: tuple[str, ...]
    ) -> RepositoryInspection:
        return await self.repository_service.inspect(
            repository_url,
            priority_paths=priority_paths,
            max_files=self.settings.rag_max_files,
            max_file_bytes=self.settings.rag_max_file_bytes,
            max_total_bytes=self.settings.rag_max_total_bytes,
            # current_revision() already used two API requests for the cache check.
            max_requests=max(3, self.settings.rag_max_requests - 2),
        )

    def _replace_index(self, revision: RepositoryRevision, snapshot: RepositoryInspection) -> None:
        chunks = chunk_repository_files(
            snapshot.files,
            chunk_lines=self.settings.rag_chunk_lines,
            overlap_lines=self.settings.rag_chunk_overlap_lines,
            max_chunks=self.settings.rag_max_chunks,
        )
        self.index.replace_index(
            revision,
            chunks,
            files_indexed=len(snapshot.files),
            total_bytes=snapshot.repository.source_bytes,
        )

    @staticmethod
    def _revision_from_inspection(
        owner: str, name: str, inspection: RepositoryInspection
    ) -> RepositoryRevision:
        repository = inspection.repository
        return RepositoryRevision(
            owner=owner,
            name=name,
            full_name=repository.full_name,
            default_branch=repository.default_branch,
            commit_sha=repository.commit_sha,
            description=repository.description,
            language=repository.language,
            stars=repository.stars,
            forks=repository.forks,
            html_url=str(repository.html_url),
        )

    def _build_context(
        self,
        revision: RepositoryRevision,
        chunks: list[IndexedChunk],
        *,
        indexed_files: tuple[str, ...] = (),
        priority_paths: tuple[str, ...] = (),
    ) -> RAGContext:
        max_chars = self.settings.llm_max_context_tokens * CHARS_PER_TOKEN_ESTIMATE
        lines = [
            "Repository metadata:",
            f"Repository: {revision.full_name}",
            f"Description: {revision.description or 'Not provided'}",
            f"Primary language: {revision.language or 'Not specified'}",
            f"Default branch: {revision.default_branch}",
            f"Inspected commit SHA: {revision.commit_sha}",
            f"Stars: {revision.stars}; forks: {revision.forks}",
            f"Index: {len(indexed_files)} source files.",
            "Explicit path status: "
            + (
                "; ".join(
                    f"{requested}: "
                    + (
                        "indexed"
                        if any(file_matches_reference(path, requested) for path in indexed_files)
                        else "not indexed"
                    )
                    for requested in priority_paths
                )
                or "none"
            ),
            "Retrieved candidates (not all are necessarily prompt-included): "
            + "; ".join(f"{chunk.path}:{chunk.start_line}-{chunk.end_line}" for chunk in chunks),
            "Only prompt-included excerpts below are evidence. Indexed or retrieved status alone "
            "is not evidence. Retrieved repository source follows.",
        ]
        indexed_and_retrieved_header = "\n".join(lines)
        included_header_placeholder = (
            "\nAllowed citation chunk IDs (return only these in citation_chunk_ids): "
            + ", ".join(chunk.chunk_id for chunk in chunks)
        )
        available_chars = (
            max_chars - len(indexed_and_retrieved_header) - len(included_header_placeholder)
        )
        text = ""
        included: list[IndexedChunk] = []
        for chunk in chunks:
            citation_url = self._citation_url(revision, chunk)
            numbered_content = "\n".join(
                f"L{line_number} | {line}"
                for line_number, line in enumerate(
                    chunk.content.splitlines(), start=chunk.start_line
                )
            )
            section = (
                f"\n\n### Evidence chunk ID: {chunk.chunk_id}\n"
                f"### Source file: {chunk.path}\n"
                f"Evidence lines: {chunk.start_line}-{chunk.end_line}\n"
                f"Blob SHA: {chunk.blob_sha}\n"
                f"Commit SHA: {chunk.commit_sha}\n"
                f"Citation URL: {citation_url}\n"
                "Source content (untrusted data):\n"
                f"{numbered_content}\n### End source chunk"
            )
            if len(section) > available_chars:
                continue
            text += section
            available_chars -= len(section)
            included.append(chunk)

        final_headers = (
            indexed_and_retrieved_header
            + "\nAllowed citation chunk IDs (return only these in citation_chunk_ids): "
            + ", ".join(chunk.chunk_id for chunk in included)
        )
        text = final_headers + text
        return RAGContext(
            text=text,
            indexed_files=indexed_files,
            retrieved_chunks=tuple(chunks),
            chunks=tuple(included),
            estimated_tokens=math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE),
        )

    @staticmethod
    def _citation_url(
        revision: RepositoryRevision,
        chunk: IndexedChunk,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        owner = quote(revision.owner, safe="")
        name = quote(revision.name, safe="")
        path = quote(chunk.path, safe="/")
        start = chunk.start_line if start_line is None else start_line
        end = chunk.end_line if end_line is None else end_line
        return f"https://github.com/{owner}/{name}/blob/{chunk.commit_sha}/{path}#L{start}-L{end}"

    def _response(
        self,
        revision: RepositoryRevision,
        context: RAGContext,
        result: ProviderResult,
        *,
        question: str = "",
    ) -> RepositoryAnalysisResponse:
        citations: list[SourceCitation] = []
        legacy_citations_present = bool(result.citation_paths or result.citations)
        invalid_format = (
            legacy_citations_present
            or len(set(result.citation_chunk_ids)) != len(result.citation_chunk_ids)
            or any(
                re.fullmatch(r"chunk_[0-9a-f]{20}", chunk_id) is None
                for chunk_id in result.citation_chunk_ids
            )
        )
        prompt_chunks_by_id = {chunk.chunk_id: chunk for chunk in context.chunks}
        unknown_chunk_id = False
        for chunk_id in result.citation_chunk_ids:
            chunk = prompt_chunks_by_id.get(chunk_id)
            if chunk is None:
                unknown_chunk_id = True
                continue
            citations.append(
                SourceCitation(
                    path=chunk.path,
                    commit_sha=chunk.commit_sha,
                    blob_sha=chunk.blob_sha,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    chunk_id=chunk_id,
                    citation_url=self._citation_url(
                        revision,
                        chunk,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                    ),
                )
            )

        answer_text = result.answer.strip()
        sufficient = (
            result.sufficient_context
            and bool(citations)
            and not (invalid_format or unknown_chunk_id)
        )
        error_code: str | None = None
        if invalid_format:
            answer = (
                "Unable to verify this answer: invalid citation format. Expected only a "
                "citation_chunk_ids array of allowed evidence IDs; path-based and line-range "
                "citations are not accepted."
            )
            citations = []
            error_code = "invalid_citation_format"
        elif unknown_chunk_id:
            answer = (
                "Unable to verify this answer: citation_chunk_ids contained an unknown ID or "
                "an ID for a chunk that was not included in this prompt."
            )
            citations = []
            error_code = "unknown_citation_id"
        elif not result.citation_chunk_ids:
            if result.sufficient_context:
                answer = "Unable to verify this answer: the model returned no citation_chunk_ids."
                error_code = "missing_citations"
            else:
                answer = "The retrieved repository evidence is insufficient. " + answer_text
                error_code = "insufficient_repository_evidence"
        else:
            answer = answer_text
            if not result.sufficient_context:
                answer = "The retrieved repository evidence is insufficient. " + answer_text
                citations = []
                sufficient = False
                error_code = "insufficient_repository_evidence"
        if citations:
            source_lines = "\n".join(
                f"- [`{citation.path}` lines {citation.start_line}-{citation.end_line}]"
                f"({citation.citation_url}) (commit `{citation.commit_sha}`)"
                for citation in citations
            )
            answer += f"\n\nSources:\n{source_lines}"

        provider_diagnostics = result.diagnostics or ProviderDiagnostics(
            pipeline_stage="citation_validation",
            model=None,
        )
        diagnostic_reason = error_code or ("none" if sufficient else "citation_validation_failed")
        diagnostics = provider_diagnostics.model_copy(
            update={
                "pipeline_stage": "api_response",
                "allowed_citation_ids": [chunk.chunk_id for chunk in context.chunks],
                "returned_citation_ids": list(result.citation_chunk_ids),
                "rejection_reason": diagnostic_reason,
                "semantic_grounding_evaluated": False,
            }
        )
        return RepositoryAnalysisResponse(
            full_name=revision.full_name,
            commit_sha=revision.commit_sha,
            answer=answer,
            context_sufficient=sufficient,
            citations=citations,
            context_tokens=context.estimated_tokens,
            indexed_files=list(context.indexed_files),
            retrieved_chunks=[self._chunk_reference(chunk) for chunk in context.retrieved_chunks],
            prompt_chunks=[self._chunk_reference(chunk) for chunk in context.chunks],
            diagnostics=diagnostics if self.settings.rag_diagnostics_enabled else None,
            error_code=error_code,
        )

    @staticmethod
    def _chunk_reference(chunk: IndexedChunk) -> RAGChunkReference:
        return RAGChunkReference(
            chunk_id=chunk.chunk_id,
            path=chunk.path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            commit_sha=chunk.commit_sha,
            blob_sha=chunk.blob_sha,
        )

    def _insufficient(
        self,
        revision: RepositoryRevision,
        message: str,
        *,
        context_tokens: int = 0,
        indexed_files: tuple[str, ...] = (),
        retrieved_chunks: tuple[IndexedChunk, ...] = (),
        prompt_chunks: tuple[IndexedChunk, ...] = (),
    ) -> RepositoryAnalysisResponse:
        diagnostic = ProviderDiagnostics(
            pipeline_stage="retrieval",
            allowed_citation_ids=[chunk.chunk_id for chunk in prompt_chunks],
            rejection_reason="insufficient_repository_evidence",
        )
        return RepositoryAnalysisResponse(
            full_name=revision.full_name,
            commit_sha=revision.commit_sha,
            answer=f"The retrieved repository context is insufficient. {message}",
            context_sufficient=False,
            citations=[],
            context_tokens=context_tokens,
            indexed_files=list(indexed_files),
            retrieved_chunks=[self._chunk_reference(chunk) for chunk in retrieved_chunks],
            prompt_chunks=[self._chunk_reference(chunk) for chunk in prompt_chunks],
            diagnostics=diagnostic if self.settings.rag_diagnostics_enabled else None,
            error_code="insufficient_repository_evidence",
        )
