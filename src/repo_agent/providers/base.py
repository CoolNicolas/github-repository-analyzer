"""Provider contract and normalized model response."""

from typing import Literal, Protocol

from pydantic import AliasChoices, BaseModel, Field


class ProviderDiagnostics(BaseModel):
    pipeline_stage: str
    provider_status: int | None = None
    finish_reason: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    returned_citation_ids: list[str] = Field(default_factory=list)
    allowed_citation_ids: list[str] = Field(default_factory=list)
    rejection_reason: str | None = None
    semantic_grounding_evaluated: bool = False


class AgentTurnDiagnostics(BaseModel):
    response_type: str
    finish_reason: str | None = None
    tool_call_count: int = 0
    parsing_status: str
    error_category: str | None = None
    estimated_context_tokens: int | None = None
    unique_evidence_chunks: int = 0
    duplicate_evidence_chunks: int = 0
    remaining_context_tokens: int | None = None
    termination_reason: str | None = None
    context_token_history: list[int] = Field(default_factory=list)
    http_status: int | None = None
    api_error_code: str | None = None
    api_error_type: str | None = None
    api_error_param: str | None = None
    request_id: str | None = None


class ProviderCitation(BaseModel):
    """Path citation accepted only by the legacy, non-RAG analysis flow."""

    path: str | None = None
    chunk_id: str | None = None
    start_line: int | None = None
    end_line: int | None = None


class ProviderResult(BaseModel):
    answer: str = Field(min_length=1)
    sufficient_context: bool = Field(
        validation_alias=AliasChoices("sufficient_context", "context_sufficient")
    )
    citation_paths: list[str] = Field(default_factory=list)
    citations: list[ProviderCitation] = Field(default_factory=list)
    citation_chunk_ids: list[str] = Field(default_factory=list)
    diagnostics: ProviderDiagnostics | None = None

    @property
    def context_sufficient(self) -> bool:
        """Compatibility accessor for callers that expose the existing API field."""
        return self.sufficient_context


class ProviderToolCall(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=100)
    arguments: dict[str, object]


class ProviderAgentTurn(BaseModel):
    tool_calls: list[ProviderToolCall] = Field(default_factory=list)
    final_result: ProviderResult | None = None
    diagnostics: AgentTurnDiagnostics | None = None


class LLMProviderError(Exception):
    """Safe, user-facing error from a configured LLM provider."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        code: str = "llm_error",
        diagnostics: ProviderDiagnostics | None = None,
        agent_diagnostics: AgentTurnDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.diagnostics = diagnostics
        self.agent_diagnostics = agent_diagnostics


class LLMProvider(Protocol):
    async def analyze_repository(self, context: str, *, max_output_tokens: int) -> ProviderResult:
        """Summarize architecture and behavior using only the bounded context."""

    async def answer_question(
        self,
        question: str,
        context: str,
        *,
        max_output_tokens: int,
        citation_mode: Literal["paths", "chunk_ids"] = "paths",
    ) -> ProviderResult:
        """Answer a repository question using only the bounded context."""

    async def agent_turn(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        *,
        max_output_tokens: int,
        tool_choice: Literal["auto", "none"] = "auto",
    ) -> ProviderAgentTurn:
        """Take one protocol-level tool-calling turn or return the final structured answer."""
