import base64

import httpx
import pytest

from repo_agent.analysis.context import BoundedRepositoryContext
from repo_agent.analysis.service import RepositoryAnalysisService
from repo_agent.github.service import GitHubRepositoryService
from repo_agent.providers.base import ProviderResult
from repo_agent.schemas import RepositoryFile, RepositoryInspection, RepositoryOverview
from repo_agent.settings import Settings


def test_analysis_response_rejects_unverified_citation_paths() -> None:
    commit = "a" * 40
    source_file = RepositoryFile(
        path="src/app.py",
        blob_sha="b" * 40,
        commit_sha=commit,
        size_bytes=15,
        content="def run(): pass",
        citation_url=f"https://github.com/o/r/blob/{commit}/src/app.py",
    )
    inspection = RepositoryInspection(
        repository=RepositoryOverview(
            name="r",
            full_name="o/r",
            html_url="https://github.com/o/r",
            stars=0,
            forks=0,
            default_branch="main",
            commit_sha=commit,
            tree_truncated=False,
            files_in_tree=1,
            files_selected=1,
            source_bytes=15,
        ),
        files=[source_file],
    )
    context = BoundedRepositoryContext(
        text="### Source file: src/app.py",
        files=(source_file,),
        estimated_tokens=20,
    )

    response = RepositoryAnalysisService._response(
        inspection,
        context,
        ProviderResult(
            answer="The application does something.",
            context_sufficient=True,
            citation_paths=["src/not-in-context.py"],
        ),
    )

    assert response.context_sufficient is False
    assert response.citations == []
    assert "insufficient" in response.answer.lower()


def test_analysis_response_resolves_unique_basename_to_verified_context_path() -> None:
    commit = "a" * 40
    source_file = RepositoryFile(
        path="src/flask/config.py",
        blob_sha="b" * 40,
        commit_sha=commit,
        size_bytes=15,
        content="class Config: pass",
        citation_url=f"https://github.com/o/r/blob/{commit}/src/flask/config.py",
    )
    inspection = RepositoryInspection(
        repository=RepositoryOverview(
            name="r",
            full_name="o/r",
            html_url="https://github.com/o/r",
            stars=0,
            forks=0,
            default_branch="main",
            commit_sha=commit,
            tree_truncated=False,
            files_in_tree=1,
            files_selected=1,
            source_bytes=15,
        ),
        files=[source_file],
    )
    context = BoundedRepositoryContext(
        text="### Source file: src/flask/config.py",
        files=(source_file,),
        estimated_tokens=20,
    )

    response = RepositoryAnalysisService._response(
        inspection,
        context,
        ProviderResult(
            answer="Configuration details.",
            context_sufficient=True,
            citation_paths=["config.py"],
        ),
    )

    assert response.context_sufficient is True
    assert response.citations[0].path == "src/flask/config.py"
    assert commit in str(response.citations[0].citation_url)


class CapturingProvider:
    def __init__(self) -> None:
        self.answer_calls = 0
        self.context = ""

    async def analyze_repository(self, context: str, *, max_output_tokens: int) -> ProviderResult:
        del context, max_output_tokens
        return ProviderResult(answer="analysis", context_sufficient=False)

    async def answer_question(
        self, question: str, context: str, *, max_output_tokens: int
    ) -> ProviderResult:
        del question, max_output_tokens
        self.answer_calls += 1
        self.context = context
        return ProviderResult(
            answer="The configuration class defines application settings.",
            context_sufficient=True,
            citation_paths=["src/flask/config.py"],
        )


def repository_service_for_tree(
    tree: list[dict], contents: dict[str, bytes]
) -> GitHubRepositoryService:
    commit = "d" * 40

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/o/r":
            return httpx.Response(
                200,
                json={
                    "name": "r",
                    "full_name": "o/r",
                    "default_branch": "main",
                    "language": "Python",
                    "html_url": "https://github.com/o/r",
                },
                request=request,
            )
        if path.endswith("/branches/main"):
            return httpx.Response(200, json={"commit": {"sha": commit}}, request=request)
        if "/git/trees/" in path:
            return httpx.Response(200, json={"tree": tree, "truncated": False}, request=request)
        file_path = path.split("/contents/", 1)[1]
        content = base64.b64encode(contents[file_path]).decode("ascii")
        return httpx.Response(
            200,
            json={"type": "file", "encoding": "base64", "content": content},
            request=request,
        )

    settings = Settings(_env_file=None, github_max_files=2)
    return GitHubRepositoryService(settings=settings, transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_explicit_filename_is_refetched_and_prioritized_in_provider_context() -> None:
    config_content = b"class Config:\n    setting = True\n"
    tree = [
        {"path": "README.md", "type": "blob", "sha": "a" * 40, "size": 1},
        {"path": "src/a.py", "type": "blob", "sha": "b" * 40, "size": 1},
        {
            "path": "src/flask/config.py",
            "type": "blob",
            "sha": "c" * 40,
            "size": len(config_content),
        },
    ]
    provider = CapturingProvider()
    analysis = RepositoryAnalysisService(
        repository_service_for_tree(
            tree,
            {
                "README.md": b"R",
                "src/a.py": b"x",
                "src/flask/config.py": config_content,
            },
        ),
        provider,
        Settings(_env_file=None, github_max_files=2),
    )

    result = await analysis.answer("https://github.com/o/r", "What does Flask's config.py do?")

    assert provider.answer_calls == 1
    assert "### Source file: src/flask/config.py" in provider.context
    assert config_content.decode() in provider.context
    assert result.context_sufficient is True
    assert result.citations[0].path == "src/flask/config.py"
    assert "src/flask/config.py" in result.answer


@pytest.mark.asyncio
async def test_missing_explicit_file_is_reported_without_calling_provider() -> None:
    tree = [{"path": "README.md", "type": "blob", "sha": "a" * 40, "size": 1}]
    provider = CapturingProvider()
    analysis = RepositoryAnalysisService(
        repository_service_for_tree(tree, {"README.md": b"R"}),
        provider,
        Settings(_env_file=None, github_max_files=2),
    )

    result = await analysis.answer("https://github.com/o/r", "What does config.py do?")

    assert provider.answer_calls == 0
    assert result.context_sufficient is False
    assert result.citations == []
    assert "config.py" in result.answer
    assert "insufficient" in result.answer.lower()
