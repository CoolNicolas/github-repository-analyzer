import base64

import httpx
import pytest

from repo_agent.api.main import create_app
from repo_agent.github.service import GitHubRepositoryService
from repo_agent.providers.mock import MockLLMProvider
from repo_agent.providers.openai_compatible import OpenAICompatibleProvider
from repo_agent.settings import Settings


@pytest.mark.asyncio
async def test_health_and_inspection_endpoints() -> None:
    commit = "a" * 40

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/o/r":
            return httpx.Response(
                200,
                json={
                    "name": "r",
                    "full_name": "o/r",
                    "default_branch": "main",
                    "html_url": "https://github.com/o/r",
                },
                request=request,
            )
        if path.endswith("/branches/main"):
            return httpx.Response(200, json={"commit": {"sha": commit}}, request=request)
        if "/git/trees/" in path:
            return httpx.Response(
                200,
                json={
                    "tree": [{"path": "README.md", "type": "blob", "sha": "b" * 40, "size": 4}],
                    "truncated": False,
                },
                request=request,
            )
        content = base64.b64encode(b"Hi!\n").decode("ascii")
        return httpx.Response(
            200, json={"type": "file", "encoding": "base64", "content": content}, request=request
        )

    service = GitHubRepositoryService(transport=httpx.MockTransport(handler))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(service, provider=MockLLMProvider())),
        base_url="http://testserver",
    ) as client:
        health = await client.get("/health")
        response = await client.post(
            "/api/repos/inspect",
            json={"repository_url": "https://github.com/o/r"},
        )

    assert health.json() == {"status": "ok"}
    assert response.status_code == 200
    body = response.json()
    assert body["repository"]["commit_sha"] == commit
    assert body["files"][0]["path"] == "README.md"


@pytest.mark.asyncio
async def test_inspection_endpoint_returns_readable_errors() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=create_app(provider=MockLLMProvider(), settings=Settings(_env_file=None))
        ),
        base_url="http://testserver",
    ) as client:
        invalid = await client.post(
            "/api/repos/inspect",
            json={"repository_url": "https://example.com/o/r"},
        )
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "invalid_repository_url"


@pytest.mark.asyncio
async def test_analysis_and_question_endpoints_use_mock_provider_and_pinned_citations(
    tmp_path,
) -> None:
    commit = "a" * 40
    github_requests: list[str] = []

    def github_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        github_requests.append(path)
        if path == "/repos/o/r":
            return httpx.Response(
                200,
                json={
                    "name": "r",
                    "full_name": "o/r",
                    "description": "A sample repository",
                    "default_branch": "main",
                    "language": "Python",
                    "html_url": "https://github.com/o/r",
                },
                request=request,
            )
        if path.endswith("/branches/main"):
            return httpx.Response(200, json={"commit": {"sha": commit}}, request=request)
        if "/git/trees/" in path:
            return httpx.Response(
                200,
                json={
                    "tree": [
                        {
                            "path": "README.md",
                            "type": "blob",
                            "sha": "b" * 40,
                            "size": 8,
                        },
                        {
                            "path": "src/app.py",
                            "type": "blob",
                            "sha": "c" * 40,
                            "size": 11,
                        },
                    ],
                    "truncated": False,
                },
                request=request,
            )
        if path.endswith("/contents/README.md"):
            content = base64.b64encode(b"# Sample\n").decode("ascii")
        else:
            content = base64.b64encode(b"def run():\n").decode("ascii")
        return httpx.Response(
            200,
            json={"type": "file", "encoding": "base64", "content": content},
            request=request,
        )

    service = GitHubRepositoryService(transport=httpx.MockTransport(github_handler))
    app = create_app(
        repository_service=service,
        provider=MockLLMProvider(),
        settings=Settings(_env_file=None, rag_sqlite_path=str(tmp_path / "api-rag.sqlite3")),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        inspected = await client.post(
            "/api/repos/inspect", json={"repository_url": "https://github.com/o/r"}
        )
        analysis = await client.post(
            "/api/repos/analyze", json={"repository_url": "https://github.com/o/r"}
        )
        answer = await client.post(
            "/api/repos/ask",
            json={"repository_url": "https://github.com/o/r", "question": "What does run do?"},
        )

    assert inspected.status_code == 200
    assert analysis.status_code == 200
    assert answer.status_code == 200
    for result in (analysis.json(), answer.json()):
        assert result["context_sufficient"] is True
        assert result["commit_sha"] == commit
        assert result["citations"]
        assert all(item["commit_sha"] == commit for item in result["citations"])
        assert all(commit in item["citation_url"] for item in result["citations"])
        assert "Sources:" in result["answer"]
    assert "src/app.py" in answer.json()["answer"]
    # The first RAG question builds a separate, wider local index from GitHub.
    assert len(github_requests) == 12


@pytest.mark.asyncio
async def test_qa_endpoint_reports_insufficient_repository_context(tmp_path) -> None:
    commit = "a" * 40

    def github_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/o/r":
            return httpx.Response(
                200,
                json={"name": "r", "full_name": "o/r", "default_branch": "main"},
                request=request,
            )
        if path.endswith("/branches/main"):
            return httpx.Response(200, json={"commit": {"sha": commit}}, request=request)
        return httpx.Response(200, json={"tree": [], "truncated": False}, request=request)

    service = GitHubRepositoryService(transport=httpx.MockTransport(github_handler))
    app = create_app(
        repository_service=service,
        provider=MockLLMProvider(),
        settings=Settings(_env_file=None, rag_sqlite_path=str(tmp_path / "qa-rag.sqlite3")),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/repos/ask",
            json={"repository_url": "https://github.com/o/r", "question": "Explain the internals"},
        )

    assert response.status_code == 200
    assert response.json()["context_sufficient"] is False
    assert "insufficient" in response.json()["answer"].lower()


@pytest.mark.asyncio
async def test_qa_endpoint_rejects_blank_questions() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=create_app(provider=MockLLMProvider(), settings=Settings(_env_file=None))
        ),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/repos/ask",
            json={"repository_url": "https://github.com/o/r", "question": "  "},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_agent_endpoint_is_explicitly_disabled_by_default() -> None:
    app = create_app(provider=MockLLMProvider(), settings=Settings(_env_file=None))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        status = await client.get("/api/status")
        response = await client.post(
            "/api/repos/agent",
            json={
                "repository_url": "https://github.com/o/r",
                "question": "Inspect source",
            },
        )

    assert status.json()["agent_enabled"] is False
    assert response.status_code == 503
    assert response.json()["code"] == "agent_disabled"


@pytest.mark.asyncio
async def test_provider_status_is_accurate_and_never_exposes_credentials() -> None:
    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="openai_compatible",
        llm_api_key="status-test-secret",
        llm_base_url="https://llm.example/v1",
        llm_model="private-model-name",
    )
    provider = OpenAICompatibleProvider(
        settings,
        transport=httpx.MockTransport(lambda request: httpx.Response(500, request=request)),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(provider=provider, settings=settings)),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/status")

    assert response.status_code == 200
    status = response.json()
    assert status["provider"] == "openai_compatible"
    assert status["live_inference_enabled"] is True
    assert "live inference is enabled" in status["message"]
    assert "status-test-secret" not in response.text
    assert "private-model-name" not in response.text


@pytest.mark.asyncio
async def test_default_provider_status_reports_mock_without_live_inference() -> None:
    app = create_app(settings=Settings(_env_file=None))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["provider"] == "mock"
    assert response.json()["live_inference_enabled"] is False
    assert "no external LLM requests" in response.json()["message"]
