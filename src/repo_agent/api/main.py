"""FastAPI endpoints for read-only repository inspection, analysis, and Q&A."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from repo_agent.agent.service import AgentError, AgentOrchestrationService
from repo_agent.analysis.service import RepositoryAnalysisService
from repo_agent.github.service import GitHubError, GitHubRepositoryService
from repo_agent.github.urls import InvalidRepositoryURL
from repo_agent.providers.base import LLMProvider, LLMProviderError
from repo_agent.providers.openai_compatible import create_provider
from repo_agent.rag.service import RepositoryRAGService
from repo_agent.schemas import (
    AgentResponse,
    InspectRequest,
    RepositoryAnalysisResponse,
    RepositoryInspection,
    RepositoryQuestionRequest,
)
from repo_agent.settings import Settings


def create_app(
    repository_service: GitHubRepositoryService | None = None,
    provider: LLMProvider | None = None,
    settings: Settings | None = None,
    rag_service: RepositoryRAGService | None = None,
) -> FastAPI:
    app = FastAPI(
        title="GitHub AI Agent",
        version="0.1.0",
        description="Read-only public repository inspection and analysis API",
    )
    app_settings = settings or (
        repository_service.settings if repository_service is not None else Settings()
    )
    github_service = repository_service or GitHubRepositoryService(app_settings)
    llm_provider = provider or create_provider(app_settings)
    analysis_service = RepositoryAnalysisService(github_service, llm_provider, app_settings)
    repository_rag_service = rag_service or RepositoryRAGService(
        github_service, llm_provider, app_settings
    )
    agent_service = AgentOrchestrationService(
        github_service, llm_provider, repository_rag_service, app_settings
    )
    app.state.repository_analysis_service = analysis_service
    app.state.repository_rag_service = repository_rag_service
    app.state.agent_service = agent_service

    @app.exception_handler(InvalidRepositoryURL)
    async def invalid_url_handler(_request: Request, exc: InvalidRepositoryURL) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"detail": str(exc), "code": "invalid_repository_url"}
        )

    @app.exception_handler(GitHubError)
    async def github_error_handler(_request: Request, exc: GitHubError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"detail": str(exc), "code": exc.code}
        )

    @app.exception_handler(LLMProviderError)
    async def llm_error_handler(_request: Request, exc: LLMProviderError) -> JSONResponse:
        content: dict[str, object] = {"detail": str(exc), "code": exc.code}
        if app_settings.rag_diagnostics_enabled and exc.diagnostics is not None:
            content["diagnostics"] = exc.diagnostics.model_dump()
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(AgentError)
    async def agent_error_handler(_request: Request, exc: AgentError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": str(exc),
                "code": exc.code,
                "steps": exc.steps,
                "tool_calls": exc.tool_calls,
                "trace": [entry.model_dump() for entry in exc.trace],
                "diagnostics": exc.diagnostics.model_dump() if exc.diagnostics else None,
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status")
    async def status() -> dict[str, str | bool]:
        live_enabled = bool(getattr(llm_provider, "live_inference_enabled", False))
        agent_available = app_settings.agent_enabled and callable(
            getattr(llm_provider, "agent_turn", None)
        )
        provider_name = str(getattr(llm_provider, "provider_name", "custom"))
        if live_enabled:
            message = "OpenAI-compatible LLM provider is active; live inference is enabled."
        elif provider_name == "mock":
            message = "Mock LLM provider is active; no external LLM requests are made."
        else:
            message = "A custom LLM provider is active."
        return {
            "provider": provider_name,
            "live_inference_enabled": live_enabled,
            "message": message,
            "agent_enabled": agent_available,
        }

    @app.post("/api/repos/inspect", response_model=RepositoryInspection)
    async def inspect_repository(payload: InspectRequest) -> RepositoryInspection:
        return await analysis_service.inspect(payload.repository_url)

    @app.post("/api/repos/analyze", response_model=RepositoryAnalysisResponse)
    async def analyze_repository(payload: InspectRequest) -> RepositoryAnalysisResponse:
        return await analysis_service.analyze(payload.repository_url)

    @app.post("/api/repos/ask", response_model=RepositoryAnalysisResponse)
    async def answer_repository_question(
        payload: RepositoryQuestionRequest,
    ) -> RepositoryAnalysisResponse:
        return await repository_rag_service.answer(payload.repository_url, payload.question)

    @app.post("/api/repos/agent", response_model=AgentResponse)
    async def answer_with_agent(payload: RepositoryQuestionRequest) -> AgentResponse:
        return await agent_service.answer(payload.repository_url, payload.question)

    return app


app = create_app()
