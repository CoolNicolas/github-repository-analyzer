"""Public API response schemas."""

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

from repo_agent.providers.base import AgentTurnDiagnostics, ProviderDiagnostics


class InspectRequest(BaseModel):
    repository_url: str = Field(min_length=1, max_length=2_048)


class RepositoryFile(BaseModel):
    path: str
    blob_sha: str
    commit_sha: str
    size_bytes: int
    content: str
    citation_url: HttpUrl


class RepositoryOverview(BaseModel):
    name: str
    full_name: str
    description: str | None = None
    html_url: HttpUrl
    stars: int
    forks: int
    language: str | None = None
    default_branch: str
    commit_sha: str
    tree_truncated: bool
    files_in_tree: int
    files_selected: int
    source_bytes: int


class RepositoryInspection(BaseModel):
    repository: RepositoryOverview
    files: list[RepositoryFile]


class RepositoryQuestionRequest(InspectRequest):
    question: str = Field(min_length=1, max_length=4_000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question must not be blank.")
        return question


class SourceCitation(BaseModel):
    path: str
    commit_sha: str
    citation_url: HttpUrl
    blob_sha: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    chunk_id: str | None = None


class RAGChunkReference(BaseModel):
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    commit_sha: str
    blob_sha: str


class RepositoryAnalysisResponse(BaseModel):
    full_name: str
    commit_sha: str
    answer: str
    context_sufficient: bool
    citations: list[SourceCitation]
    context_tokens: int
    indexed_files: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RAGChunkReference] = Field(default_factory=list)
    prompt_chunks: list[RAGChunkReference] = Field(default_factory=list)
    diagnostics: ProviderDiagnostics | None = None
    error_code: str | None = None


class AgentFileReference(BaseModel):
    path: str
    commit_sha: str
    chunk_id: str | None = None
    start_line: int | None = None
    end_line: int | None = None


class AgentTraceEntry(BaseModel):
    step: int
    tool_name: str
    arguments: dict[str, str | list[str]] = Field(default_factory=dict)
    status: Literal["success", "error", "repeated"]
    references: list[AgentFileReference] = Field(default_factory=list)
    message: str | None = None


class AgentResponse(BaseModel):
    full_name: str
    commit_sha: str
    answer: str
    context_sufficient: bool
    citations: list[SourceCitation] = Field(default_factory=list)
    trace: list[AgentTraceEntry] = Field(default_factory=list)
    steps: int
    tool_calls: int
    error_code: str | None = None
    diagnostics: AgentTurnDiagnostics | None = None
