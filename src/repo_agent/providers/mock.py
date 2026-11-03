"""Deterministic mock provider used by default and in tests."""

import re
from typing import Literal

from repo_agent.providers.base import ProviderResult

SOURCE_PATH_PATTERN = re.compile(r"^### Source file: (.+)$", re.MULTILINE)
ALLOWED_CHUNK_IDS_PATTERN = re.compile(
    r"^Allowed citation chunk IDs \(return only these in citation_chunk_ids\): (.*)$",
    re.MULTILINE,
)
CHUNK_ID_PATTERN = re.compile(r"chunk_[0-9a-f]{20}")


class MockLLMProvider:
    """Return stable, local-only results with citations to supplied context."""

    provider_name = "mock"
    live_inference_enabled = False

    @staticmethod
    def _source_paths(context: str) -> list[str]:
        return SOURCE_PATH_PATTERN.findall(context)

    async def analyze_repository(self, context: str, *, max_output_tokens: int) -> ProviderResult:
        del max_output_tokens
        paths = self._source_paths(context)
        if not paths:
            return ProviderResult(
                answer="The available repository context is insufficient for a source analysis.",
                sufficient_context=False,
            )
        return ProviderResult(
            answer=(
                "Mock analysis based on the supplied repository context "
                f"({len(paths)} source files)."
            ),
            sufficient_context=True,
            citation_paths=paths[:3],
        )

    async def answer_question(
        self,
        question: str,
        context: str,
        *,
        max_output_tokens: int,
        citation_mode: Literal["paths", "chunk_ids"] = "paths",
    ) -> ProviderResult:
        del max_output_tokens
        if citation_mode == "chunk_ids" or (
            "Allowed citation chunk IDs (return only these in citation_chunk_ids):" in context
        ):
            allowed_line = ALLOWED_CHUNK_IDS_PATTERN.search(context)
            chunk_ids = CHUNK_ID_PATTERN.findall(allowed_line.group(1)) if allowed_line else []
            if not chunk_ids:
                return ProviderResult(
                    answer=(
                        "The available repository context is insufficient to answer this question."
                    ),
                    sufficient_context=False,
                )
            return ProviderResult(
                answer=f"Mock answer to: {question}",
                sufficient_context=True,
                citation_chunk_ids=chunk_ids[:3],
            )
        paths = self._source_paths(context)
        if not paths:
            return ProviderResult(
                answer="The available repository context is insufficient to answer this question.",
                sufficient_context=False,
            )
        return ProviderResult(
            answer=f"Mock answer to: {question}",
            sufficient_context=True,
            citation_paths=paths[:3],
        )
