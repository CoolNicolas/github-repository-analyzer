"""Repository analysis and question answering with bounded context and citations."""

import time
from collections import OrderedDict

from repo_agent.analysis.context import (
    BoundedRepositoryContext,
    build_repository_context,
    extract_file_references,
    file_matches_reference,
)
from repo_agent.github.service import GitHubRepositoryService
from repo_agent.github.urls import parse_repository_url
from repo_agent.providers.base import LLMProvider, ProviderResult
from repo_agent.schemas import (
    RepositoryAnalysisResponse,
    RepositoryInspection,
    SourceCitation,
)
from repo_agent.settings import Settings

MAX_CACHED_REPOSITORIES = 8


class RepositoryAnalysisService:
    """Cache inspected repositories briefly and run mock or configured LLM tasks."""

    def __init__(
        self,
        repository_service: GitHubRepositoryService,
        provider: LLMProvider,
        settings: Settings,
    ) -> None:
        self.repository_service = repository_service
        self.provider = provider
        self.settings = settings
        self._inspection_cache: OrderedDict[str, tuple[float, RepositoryInspection]] = OrderedDict()

    async def inspect(self, repository_url: str) -> RepositoryInspection:
        reference = parse_repository_url(repository_url)
        cache_key = f"{reference.owner.casefold()}/{reference.name.casefold()}"
        now = time.monotonic()
        cached = self._inspection_cache.get(cache_key)
        if cached is not None:
            cached_at, inspection = cached
            if now - cached_at <= self.settings.repository_cache_ttl_seconds:
                self._inspection_cache.move_to_end(cache_key)
                return inspection
            del self._inspection_cache[cache_key]

        inspection = await self.repository_service.inspect(repository_url)
        if self.settings.repository_cache_ttl_seconds > 0:
            self._remember_inspection(cache_key, inspection, now)
        return inspection

    def _remember_inspection(
        self, cache_key: str, inspection: RepositoryInspection, stored_at: float | None = None
    ) -> None:
        if self.settings.repository_cache_ttl_seconds <= 0:
            return
        self._inspection_cache[cache_key] = (
            stored_at if stored_at is not None else time.monotonic(),
            inspection,
        )
        self._inspection_cache.move_to_end(cache_key)
        while len(self._inspection_cache) > MAX_CACHED_REPOSITORIES:
            self._inspection_cache.popitem(last=False)

    async def analyze(self, repository_url: str) -> RepositoryAnalysisResponse:
        inspection = await self.inspect(repository_url)
        context = self._context(inspection)
        result = await self.provider.analyze_repository(
            context.text,
            max_output_tokens=self.settings.llm_max_output_tokens,
        )
        return self._response(inspection, context, result)

    async def answer(self, repository_url: str, question: str) -> RepositoryAnalysisResponse:
        references = extract_file_references(question)
        inspection = await self.inspect(repository_url)
        missing_from_inspection = [
            reference
            for reference in references
            if not any(
                file_matches_reference(source_file.path, reference)
                for source_file in inspection.files
            )
        ]
        if missing_from_inspection:
            inspection = await self.repository_service.inspect(
                repository_url,
                priority_paths=references,
            )
            reference = parse_repository_url(repository_url)
            cache_key = f"{reference.owner.casefold()}/{reference.name.casefold()}"
            self._remember_inspection(cache_key, inspection)

        context = self._context(inspection, question=question)
        if context.missing_references:
            missing_names = ", ".join(context.missing_references)
            return self._response(
                inspection,
                context,
                ProviderResult(
                    answer=(
                        "The available repository context is insufficient because the requested "
                        f"file(s) could not be retrieved within the source limits: {missing_names}."
                    ),
                    sufficient_context=False,
                ),
            )
        result = await self.provider.answer_question(
            question,
            context.text,
            max_output_tokens=self.settings.llm_max_output_tokens,
        )
        return self._response(inspection, context, result)

    def _context(
        self, inspection: RepositoryInspection, *, question: str | None = None
    ) -> BoundedRepositoryContext:
        return build_repository_context(
            inspection,
            max_context_tokens=self.settings.llm_max_context_tokens,
            question=question,
        )

    @staticmethod
    def _response(
        inspection: RepositoryInspection,
        context: BoundedRepositoryContext,
        result: ProviderResult,
    ) -> RepositoryAnalysisResponse:
        citations: list[SourceCitation] = []
        cited_paths: set[str] = set()
        for path in result.citation_paths:
            matching_files = [
                source_file
                for source_file in context.files
                if file_matches_reference(source_file.path, path)
            ]
            if len(matching_files) != 1:
                continue
            source_file = matching_files[0]
            if source_file.path in cited_paths:
                continue
            cited_paths.add(source_file.path)
            citations.append(
                SourceCitation(
                    path=source_file.path,
                    commit_sha=source_file.commit_sha,
                    citation_url=source_file.citation_url,
                )
            )

        context_sufficient = result.context_sufficient and (bool(citations) or not context.files)
        answer = result.answer.strip()
        if not context_sufficient:
            if "insufficient" not in answer.casefold():
                if result.context_sufficient and context.files and not citations:
                    reason = (
                        "The available repository context is insufficient to verify this response "
                        "because the provider did not cite a file in the supplied context."
                    )
                else:
                    reason = (
                        "The available repository context is insufficient to answer this reliably."
                    )
                answer = f"{reason}\n\n{answer}"
        if citations:
            source_list = "\n".join(
                f"- [`{citation.path}`]({citation.citation_url}) (commit `{citation.commit_sha}`)"
                for citation in citations
            )
            answer += f"\n\nSources:\n{source_list}"

        return RepositoryAnalysisResponse(
            full_name=inspection.repository.full_name,
            commit_sha=inspection.repository.commit_sha,
            answer=answer,
            context_sufficient=context_sufficient,
            citations=citations,
            context_tokens=context.estimated_tokens,
        )
