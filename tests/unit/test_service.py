import base64
from datetime import UTC, datetime

import httpx
import pytest

from repo_agent.github.service import GitHubError, GitHubRepositoryService
from repo_agent.settings import Settings


def json_response(request: httpx.Request, payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload, request=request)


def encoded(text: bytes) -> str:
    return base64.b64encode(text).decode("ascii")


def make_service(handler, **limits) -> GitHubRepositoryService:
    settings = Settings(_env_file=None, **limits)
    return GitHubRepositoryService(settings=settings, transport=httpx.MockTransport(handler))


def test_python_selection_prioritizes_source_and_keeps_root_context_files() -> None:
    entries = [
        {"path": ".github/workflows/tests.yml", "type": "blob", "sha": "a", "size": 20},
        {"path": "docs/architecture.md", "type": "blob", "sha": "b", "size": 20},
        {"path": "setup.cfg", "type": "blob", "sha": "c", "size": 20},
        {"path": "src/sample_pkg/core.py", "type": "blob", "sha": "d", "size": 20},
        {"path": "src/sample_pkg/__init__.py", "type": "blob", "sha": "e", "size": 20},
        {"path": "src/sample_pkg/main.py", "type": "blob", "sha": "f", "size": 20},
        {"path": "sample_pkg/app.py", "type": "blob", "sha": "g", "size": 20},
        {"path": "README.md", "type": "blob", "sha": "h", "size": 20},
        {"path": "pyproject.toml", "type": "blob", "sha": "i", "size": 20},
    ]

    service = GitHubRepositoryService(settings=Settings(_env_file=None))
    selected = service._select_text_blobs(entries, language="Python")

    assert [entry["path"] for entry in selected] == [
        "README.md",
        "pyproject.toml",
        "src/sample_pkg/main.py",
        "src/sample_pkg/__init__.py",
        "src/sample_pkg/core.py",
        "sample_pkg/app.py",
        "setup.cfg",
        ".github/workflows/tests.yml",
        "docs/architecture.md",
    ]


def test_non_python_source_precedes_configuration_ci_and_documentation() -> None:
    entries = [
        {"path": "docs/guide.md", "type": "blob", "sha": "a", "size": 10},
        {"path": ".github/workflows/build.yml", "type": "blob", "sha": "b", "size": 10},
        {"path": "package.json", "type": "blob", "sha": "c", "size": 10},
        {"path": "src/app.ts", "type": "blob", "sha": "d", "size": 10},
        {"path": "src/index.ts", "type": "blob", "sha": "e", "size": 10},
    ]

    service = GitHubRepositoryService(settings=Settings(_env_file=None))
    selected = service._select_text_blobs(entries, language="TypeScript")

    assert [entry["path"] for entry in selected] == [
        "src/index.ts",
        "src/app.ts",
        "package.json",
        ".github/workflows/build.yml",
        "docs/guide.md",
    ]


@pytest.mark.asyncio
async def test_inspect_fetches_metadata_commit_tree_and_bounded_text_files() -> None:
    commit = "a" * 40
    tree = [
        {"path": "src/main.py", "type": "blob", "sha": "b" * 40, "size": 18},
        {"path": "README.md", "type": "blob", "sha": "c" * 40, "size": 12},
        {"path": "assets/logo.png", "type": "blob", "sha": "d" * 40, "size": 100},
    ]
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        path = request.url.path
        if path == "/repos/acme/demo":
            return json_response(
                request,
                {
                    "name": "demo",
                    "full_name": "acme/demo",
                    "description": "A demo",
                    "html_url": "https://github.com/acme/demo",
                    "stargazers_count": 9,
                    "forks_count": 2,
                    "language": "Python",
                    "default_branch": "main",
                },
            )
        if path == "/repos/acme/demo/branches/main":
            return json_response(request, {"commit": {"sha": commit}})
        if path == f"/repos/acme/demo/git/trees/{commit}":
            return json_response(request, {"tree": tree, "truncated": False})
        if path == "/repos/acme/demo/contents/README.md":
            return json_response(
                request,
                {"type": "file", "encoding": "base64", "content": encoded(b"# Demo\nHello\n")},
            )
        if path == "/repos/acme/demo/contents/src/main.py":
            return json_response(
                request,
                {"type": "file", "encoding": "base64", "content": encoded(b"print('hello')\n")},
            )
        return httpx.Response(404, json={"message": "missing"}, request=request)

    result = await make_service(handler).inspect("https://github.com/acme/demo")

    assert result.repository.commit_sha == commit
    assert result.repository.files_in_tree == 3
    assert result.repository.files_selected == 2
    assert [file.path for file in result.files] == ["README.md", "src/main.py"]
    assert all(file.commit_sha == commit for file in result.files)
    assert all(commit in str(file.citation_url) for file in result.files)
    assert all("logo.png" not in request for request in requests)
    assert all(request.startswith("https://api.github.com/") for request in requests)


@pytest.mark.asyncio
async def test_github_requests_use_bearer_token_when_configured() -> None:
    token = "test-token-that-must-not-be-logged"
    authorization_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorization_headers.append(request.headers.get("authorization"))
        if request.url.path == "/repos/o/r":
            return json_response(request, {"default_branch": "main"})
        if request.url.path.endswith("/branches/main"):
            return json_response(request, {"commit": {"sha": "a" * 40}})
        return httpx.Response(200, json={"tree": [], "truncated": False}, request=request)

    service = make_service(handler, github_token=token)
    await service.inspect("https://github.com/o/r")

    assert authorization_headers == [f"Bearer {token}"] * 3
    assert token not in repr(service.settings)


@pytest.mark.asyncio
async def test_github_requests_remain_unauthenticated_without_token() -> None:
    authorization_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorization_headers.append(request.headers.get("authorization"))
        if request.url.path == "/repos/o/r":
            return json_response(request, {"default_branch": "main"})
        if request.url.path.endswith("/branches/main"):
            return json_response(request, {"commit": {"sha": "a" * 40}})
        return httpx.Response(200, json={"tree": [], "truncated": False}, request=request)

    await make_service(handler).inspect("https://github.com/o/r")

    assert authorization_headers == [None, None, None]


@pytest.mark.asyncio
async def test_enforces_file_and_total_byte_limits() -> None:
    commit = "a" * 40
    tree = [
        {"path": "README.md", "type": "blob", "sha": "b" * 40, "size": 8},
        {"path": "src/a.py", "type": "blob", "sha": "c" * 40, "size": 8},
    ]
    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/o/r":
            return json_response(
                request, {"name": "r", "full_name": "o/r", "default_branch": "main"}
            )
        if path.endswith("/branches/main"):
            return json_response(request, {"commit": {"sha": commit}})
        if "/git/trees/" in path:
            return json_response(request, {"tree": tree, "truncated": False})
        fetched.append(path)
        return json_response(
            request, {"type": "file", "encoding": "base64", "content": encoded(b"12345678")}
        )

    result = await make_service(
        handler, github_max_file_bytes=10, github_max_total_bytes=8
    ).inspect("https://github.com/o/r")
    assert len(result.files) == 1
    assert result.repository.source_bytes == 8
    assert len(fetched) == 1


@pytest.mark.asyncio
async def test_caps_file_content_requests_at_max_files() -> None:
    commit = "a" * 40
    tree = [
        {"path": f"src/file_{index}.py", "type": "blob", "sha": str(index) * 40, "size": 2}
        for index in range(6)
    ]
    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/o/r":
            return json_response(
                request, {"name": "r", "full_name": "o/r", "default_branch": "main"}
            )
        if path.endswith("/branches/main"):
            return json_response(request, {"commit": {"sha": commit}})
        if "/git/trees/" in path:
            return json_response(request, {"tree": tree, "truncated": False})
        fetched.append(path)
        return json_response(
            request,
            {"type": "file", "encoding": "base64", "content": encoded(b"ok")},
        )

    result = await make_service(handler, github_max_files=2).inspect("https://github.com/o/r")
    assert len(result.files) == 2
    assert len(fetched) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 429])
async def test_rate_limit_is_reported_with_retryable_status(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            headers={"x-ratelimit-remaining": "0"},
            json={"message": "rate limit"},
            request=request,
        )

    with pytest.raises(GitHubError) as error:
        await make_service(handler).inspect("https://github.com/o/r")
    assert error.value.status_code == 429
    assert error.value.code == "rate_limited"


@pytest.mark.asyncio
async def test_rate_limit_uses_retry_after_without_automatic_retries() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            headers={"retry-after": "7", "x-ratelimit-reset": "9999999999"},
            json={"message": "rate limit"},
            request=request,
        )

    with pytest.raises(GitHubError) as error:
        await make_service(handler).inspect("https://github.com/o/r")

    assert calls == 1
    assert error.value.retry_after_seconds == 7
    assert "7 seconds" in str(error.value)


@pytest.mark.asyncio
async def test_rate_limit_uses_reset_header_when_retry_after_is_absent() -> None:
    reset_at = int(datetime.now(UTC).timestamp()) + 45

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(reset_at)},
            json={"message": "rate limit"},
            request=request,
        )

    with pytest.raises(GitHubError) as error:
        await make_service(handler).inspect("https://github.com/o/r")

    assert error.value.retry_after_seconds is not None
    assert 0 < error.value.retry_after_seconds <= 45


@pytest.mark.asyncio
async def test_authentication_failure_is_safe_and_does_not_retry() -> None:
    token = "sensitive-test-token"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers.get("authorization") == f"Bearer {token}"
        return httpx.Response(401, json={"message": "Bad credentials"}, request=request)

    with pytest.raises(GitHubError) as error:
        await make_service(handler, github_token=token).inspect("https://github.com/o/r")

    assert calls == 1
    assert error.value.code == "authentication_failed"
    assert token not in str(error.value)


@pytest.mark.asyncio
async def test_missing_repository_is_reported_as_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"}, request=request)

    with pytest.raises(GitHubError) as error:
        await make_service(handler).inspect("https://github.com/o/r")
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_timeout_and_network_failures_are_friendly() -> None:
    def timeout_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(GitHubError, match="timed out"):
        await make_service(timeout_handler).inspect("https://github.com/o/r")

    def network_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    with pytest.raises(GitHubError, match="Could not connect"):
        await make_service(network_handler).inspect("https://github.com/o/r")


@pytest.mark.asyncio
async def test_invalid_urls_are_rejected_before_network_requests() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    with pytest.raises(ValueError, match="github.com"):
        await make_service(handler).inspect("https://evil.example/o/r")
    assert called is False
