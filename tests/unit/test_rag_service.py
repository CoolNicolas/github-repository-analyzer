import base64
import re
from typing import Literal

import httpx
import pytest

from repo_agent.github.service import GitHubRepositoryService, RepositoryRevision
from repo_agent.providers.base import ProviderResult
from repo_agent.providers.openai_compatible import OpenAICompatibleProvider
from repo_agent.rag.chunking import chunk_repository_files
from repo_agent.rag.index import IndexedChunk, SQLiteFTSIndex
from repo_agent.rag.service import RAGContext, RepositoryRAGService
from repo_agent.schemas import RepositoryFile
from repo_agent.settings import Settings


class CapturingProvider:
    provider_name = "mock-test"
    live_inference_enabled = False

    def __init__(self, citation_path: str | None = None) -> None:
        self.citation_path = citation_path
        self.contexts: list[str] = []

    async def analyze_repository(self, context: str, *, max_output_tokens: int) -> ProviderResult:
        raise NotImplementedError

    async def answer_question(
        self,
        question: str,
        context: str,
        *,
        max_output_tokens: int,
        citation_mode: Literal["paths", "chunk_ids"] = "paths",
    ) -> ProviderResult:
        del citation_mode
        self.contexts.append(context)
        allowed_chunks = re.findall(
            r"(?m)^### Evidence chunk ID: (chunk_[0-9a-f]{20})\n### Source file: ([^\n]+)",
            context,
        )
        matching_chunk_ids = [
            chunk_id for chunk_id, path in allowed_chunks if path == self.citation_path
        ]
        if matching_chunk_ids:
            return ProviderResult(
                answer="The matched source describes the requested behavior.",
                context_sufficient=True,
                citation_chunk_ids=matching_chunk_ids,
            )
        return ProviderResult(
            answer="The matched source describes the requested behavior.",
            context_sufficient=True,
            citation_paths=[self.citation_path] if self.citation_path else [],
        )


def _github_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    commit = "c" * 40
    if path == "/repos/o/r":
        return httpx.Response(
            200,
            json={
                "name": "r",
                "full_name": "o/r",
                "description": "Test repository",
                "default_branch": "main",
                "language": "Python",
                "html_url": "https://github.com/o/r",
            },
            request=request,
        )
    if path.endswith("/branches/main"):
        return httpx.Response(200, json={"commit": {"sha": commit}}, request=request)
    if "/git/trees/" in path:
        entries = [
            {
                "path": f"src/module_{number:02}.py",
                "type": "blob",
                "sha": f"{number + 1:040x}",
                "size": 100,
            }
            for number in range(25)
        ]
        return httpx.Response(200, json={"tree": entries, "truncated": False}, request=request)
    path_name = path.rsplit("/contents/", 1)[-1]
    if path_name == "src/module_24.py":
        source = "def unique_identifier_24():\n    return 'target behavior'\n"
    else:
        module_number = int(path_name.removeprefix("src/module_").removesuffix(".py"))
        source = f"def ordinary_function_{module_number}():\n    return {module_number}\n"
    encoded = base64.b64encode(source.encode()).decode()
    return httpx.Response(
        200, json={"type": "file", "encoding": "base64", "content": encoded}, request=request
    )


@pytest.mark.asyncio
async def test_indexes_more_than_twenty_files_retrieves_relevant_chunk_and_reuses_commit(
    tmp_path,
) -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return _github_handler(request)

    settings = Settings(
        _env_file=None,
        rag_sqlite_path=str(tmp_path / "rag.sqlite3"),
        rag_max_files=30,
        rag_max_requests=50,
        rag_chunk_lines=20,
        rag_chunk_overlap_lines=2,
        llm_max_context_tokens=512,
    )
    github = GitHubRepositoryService(settings, transport=httpx.MockTransport(handler))
    provider = CapturingProvider("src/module_24.py")
    rag = RepositoryRAGService(github, provider, settings)

    first = await rag.answer("https://github.com/o/r", "What does unique_identifier_24 do?")
    calls_after_index = len(requests)
    second = await rag.answer("https://github.com/o/r", "Explain unique_identifier_24")

    assert first.context_sufficient is True
    assert first.citations[0].path == "src/module_24.py"
    assert first.citations[0].blob_sha == f"{25:040x}"
    assert first.citations[0].start_line == 1
    assert f"/blob/{'c' * 40}/src/module_24.py#L1-L2" in str(first.citations[0].citation_url)
    assert "Retrieved repository source" in provider.contexts[0]
    assert "src/module_24.py" in provider.contexts[0]
    assert len(requests) - calls_after_index == 2  # only metadata and branch SHA rechecked
    assert second.context_sufficient is True


@pytest.mark.asyncio
async def test_rag_indexing_uses_configured_github_token(tmp_path) -> None:
    token = "rag-test-token"
    authorization_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorization_headers.append(request.headers.get("authorization"))
        return _github_handler(request)

    settings = Settings(
        _env_file=None,
        github_token=token,
        rag_sqlite_path=str(tmp_path / "rag.sqlite3"),
        rag_max_files=30,
        rag_max_requests=50,
        rag_chunk_lines=20,
        rag_chunk_overlap_lines=2,
        llm_max_context_tokens=512,
    )
    github = GitHubRepositoryService(settings, transport=httpx.MockTransport(handler))
    rag = RepositoryRAGService(github, CapturingProvider("src/module_24.py"), settings)

    result = await rag.answer("https://github.com/o/r", "What does unique_identifier_24 do?")

    assert result.context_sufficient is True
    assert authorization_headers
    assert all(value == f"Bearer {token}" for value in authorization_headers)


@pytest.mark.asyncio
async def test_missing_file_and_irrelevant_retrieval_do_not_call_provider(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        rag_sqlite_path=str(tmp_path / "rag.sqlite3"),
        rag_max_files=30,
        rag_max_requests=50,
    )
    provider = CapturingProvider("src/module_00.py")
    github = GitHubRepositoryService(settings, transport=httpx.MockTransport(_github_handler))
    rag = RepositoryRAGService(github, provider, settings)

    missing = await rag.answer("https://github.com/o/r", "What does missing_module.py do?")
    irrelevant = await rag.answer("https://github.com/o/r", "Explain a quasar xenolith")

    assert missing.context_sufficient is False
    assert missing.citations == []
    assert missing.indexed_files
    assert missing.retrieved_chunks == []
    assert missing.prompt_chunks == []
    assert irrelevant.context_sufficient is False
    assert irrelevant.citations == []
    assert provider.contexts == []


@pytest.mark.asyncio
async def test_context_token_cap_and_unverified_provider_citation(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        rag_sqlite_path=str(tmp_path / "rag.sqlite3"),
        rag_max_files=30,
        rag_max_requests=50,
        rag_chunk_lines=20,
        llm_max_context_tokens=256,
    )
    provider = CapturingProvider("src/not-retrieved.py")
    github = GitHubRepositoryService(settings, transport=httpx.MockTransport(_github_handler))
    rag = RepositoryRAGService(github, provider, settings)

    response = await rag.answer("https://github.com/o/r", "What does unique_identifier_24 do?")

    assert response.context_tokens <= settings.llm_max_context_tokens
    assert response.context_sufficient is False
    assert response.citations == []
    assert "invalid citation format" in response.answer.casefold()
    assert len(provider.contexts[0]) <= settings.llm_max_context_tokens * 4
    assert response.retrieved_chunks
    assert response.prompt_chunks


def _rag_response_service(tmp_path) -> RepositoryRAGService:
    settings = Settings(_env_file=None, rag_sqlite_path=str(tmp_path / "response.sqlite3"))
    return RepositoryRAGService(GitHubRepositoryService(settings), CapturingProvider(), settings)


def _response_revision() -> RepositoryRevision:
    return RepositoryRevision(
        owner="o",
        name="r",
        full_name="o/r",
        default_branch="main",
        commit_sha="c" * 40,
        description=None,
        language="Python",
        stars=0,
        forks=0,
        html_url="https://github.com/o/r",
    )


def _line_chunk(start_line: int, end_line: int) -> IndexedChunk:
    return IndexedChunk(
        path="src/flask/helpers.py",
        start_line=start_line,
        end_line=end_line,
        blob_sha="b" * 40,
        commit_sha="c" * 40,
        content="\n".join(
            f"source line {line_number}" for line_number in range(start_line, end_line + 1)
        ),
    )


def test_rag_citation_range_must_match_prompt_included_chunk(tmp_path) -> None:
    prompt_chunk = _line_chunk(391, 470)
    retrieved_only_chunk = _line_chunk(326, 405)
    context = RAGContext(
        text="prompt contains only helpers.py lines 326-405",
        indexed_files=("src/flask/helpers.py",),
        retrieved_chunks=(prompt_chunk, retrieved_only_chunk),
        chunks=(prompt_chunk,),
        estimated_tokens=20,
    )
    result = ProviderResult(
        answer="This claim is based on lines that were not shown.",
        context_sufficient=True,
        citation_chunk_ids=[retrieved_only_chunk.chunk_id],
    )

    response = _rag_response_service(tmp_path)._response(
        _response_revision(), context, result, question="How does open_session work?"
    )

    assert response.context_sufficient is False
    assert response.citations == []
    assert "not included in this prompt" in response.answer
    assert response.retrieved_chunks[1].start_line == 326
    assert response.prompt_chunks == [response.retrieved_chunks[0]]


def test_rag_citation_uses_verified_model_range_in_server_generated_url(tmp_path) -> None:
    chunk = _line_chunk(326, 405)
    context = RAGContext(
        text="prompt includes numbered source lines",
        indexed_files=("src/flask/helpers.py",),
        retrieved_chunks=(chunk,),
        chunks=(chunk,),
        estimated_tokens=20,
    )
    result = ProviderResult(
        answer="The code signs the session cookie.",
        context_sufficient=True,
        citation_chunk_ids=[chunk.chunk_id],
    )

    response = _rag_response_service(tmp_path)._response(_response_revision(), context, result)

    assert response.context_sufficient is True
    assert response.citations[0].start_line == 326
    assert response.citations[0].end_line == 405
    assert response.citations[0].chunk_id == chunk.chunk_id
    assert str(response.citations[0].citation_url).endswith("#L326-L405")


def test_rag_keeps_valid_chunk_citations_separate_from_semantic_grounding(tmp_path) -> None:
    unrelated_chunk = IndexedChunk(
        path="src/flask/sessions.py",
        start_line=260,
        end_line=340,
        blob_sha="b" * 40,
        commit_sha="c" * 40,
        content=(
            "    def save_session(self, app, session, response):\n"
            "        response.set_cookie('session', session)\n"
        ),
    )
    context = RAGContext(
        text="only save_session source was included",
        indexed_files=("src/flask/sessions.py",),
        retrieved_chunks=(unrelated_chunk,),
        chunks=(unrelated_chunk,),
        estimated_tokens=20,
    )
    result = ProviderResult(
        answer="open_session loads the signed cookie.",
        context_sufficient=True,
        citation_chunk_ids=[unrelated_chunk.chunk_id],
    )

    response = _rag_response_service(tmp_path)._response(
        _response_revision(), context, result, question="Explain open_session"
    )

    assert response.context_sufficient is True
    assert response.citations[0].chunk_id == unrelated_chunk.chunk_id
    assert response.diagnostics is None


def test_rag_rejects_model_generated_subrange_without_a_prompt_chunk_id(tmp_path) -> None:
    chunk = _line_chunk(261, 340)
    context = RAGContext(
        text="chunk ID evidence is supplied",
        indexed_files=(chunk.path,),
        retrieved_chunks=(chunk,),
        chunks=(chunk,),
        estimated_tokens=20,
    )
    result = ProviderResult(
        answer="A claim about source code.",
        context_sufficient=True,
        citations=[{"path": chunk.path, "start_line": 290, "end_line": 301}],
    )

    response = _rag_response_service(tmp_path)._response(_response_revision(), context, result)

    assert response.context_sufficient is False
    assert response.citations == []
    assert "invalid citation format" in response.answer


def test_rag_distinguishes_legacy_citation_format_from_missing_evidence(tmp_path) -> None:
    chunk = _line_chunk(261, 340)
    context = RAGContext(
        text="only chunk ids are allowed",
        indexed_files=(chunk.path,),
        retrieved_chunks=(chunk,),
        chunks=(chunk,),
        estimated_tokens=20,
    )
    service = _rag_response_service(tmp_path)

    legacy = service._response(
        _response_revision(),
        context,
        ProviderResult(
            answer="An answer.",
            context_sufficient=True,
            citation_paths=[chunk.path],
        ),
    )
    missing = service._response(
        _response_revision(),
        context,
        ProviderResult(answer="Insufficient evidence.", context_sufficient=False),
    )

    assert "invalid citation format" in legacy.answer
    assert legacy.citations == []
    assert "insufficient" in missing.answer.casefold()
    assert missing.error_code == "insufficient_repository_evidence"
    assert missing.citations == []


def test_rag_rejects_unknown_chunk_ids_and_accepts_multiple_allowed_ids(tmp_path) -> None:
    first = _line_chunk(261, 340)
    second = IndexedChunk(
        path="src/flask/sessions.py",
        start_line=326,
        end_line=385,
        blob_sha="d" * 40,
        commit_sha="c" * 40,
        content="session source excerpt",
    )
    context = RAGContext(
        text="two allowed evidence chunks",
        indexed_files=(first.path, second.path),
        retrieved_chunks=(first, second),
        chunks=(first, second),
        estimated_tokens=30,
    )
    service = _rag_response_service(tmp_path)

    unknown = service._response(
        _response_revision(),
        context,
        ProviderResult(
            answer="An answer.",
            context_sufficient=True,
            citation_chunk_ids=["chunk_ffffffffffffffffffff"],
        ),
    )
    multiple = service._response(
        _response_revision(),
        context,
        ProviderResult(
            answer="An answer supported by both chunks.",
            context_sufficient=True,
            citation_chunk_ids=[first.chunk_id, second.chunk_id],
        ),
    )

    assert "unknown ID" in unknown.answer
    assert unknown.citations == []
    assert multiple.context_sufficient is True
    assert [citation.chunk_id for citation in multiple.citations] == [
        first.chunk_id,
        second.chunk_id,
    ]
    assert str(multiple.citations[1].citation_url).endswith("#L326-L385")


def test_rag_prompt_labels_indexed_retrieved_and_prompt_included_evidence(tmp_path) -> None:
    service = _rag_response_service(tmp_path)
    chunk = _line_chunk(326, 405)

    context = service._build_context(
        _response_revision(),
        [chunk],
        indexed_files=("src/flask/helpers.py", "src/flask/sessions.py"),
        priority_paths=("sessions.py",),
    )

    assert "Index: 2 source files" in context.text
    assert "sessions.py: indexed" in context.text
    assert "Retrieved candidates" in context.text
    assert "Allowed citation chunk IDs (return only these in citation_chunk_ids)" in context.text
    assert chunk.chunk_id in context.text
    allowed_line = re.search(
        r"(?m)^Allowed citation chunk IDs \(return only these in citation_chunk_ids\): (.*)$",
        context.text,
    )
    assert allowed_line is not None
    assert allowed_line.group(1) == chunk.chunk_id
    assert chunk.path not in allowed_line.group(1)
    assert "326-405" not in allowed_line.group(1)
    assert "L326 |" in context.text
    assert "L405 |" in context.text


@pytest.mark.asyncio
async def test_flask_session_question_full_offline_pipeline_with_overlapping_chunks(
    tmp_path,
) -> None:
    commit = "49943b5a94413a530761350f04e896f76e8d620e"
    blob = "a" * 40
    source_lines = [f"# filler line {number}" for number in range(1, 386)]
    source_lines[249] = "class SecureCookieSessionInterface(SessionInterface):"
    source_lines[250] = "    session_class = SecureCookieSession"
    source_lines[289] = "    def open_session(self, app, request):"
    source_lines[290:301] = [
        "        val = request.cookies.get(app.session_cookie_name)",
        "        if not val:",
        "            return self.session_class()",
        "        try:",
        "            data = self.get_signing_serializer(app).loads(val)",
        "            return self.session_class(data)",
        "        except BadSignature:",
        "            return self.session_class()",
        "        # session payload is verified by its signature",
        "        # it is not decrypted by this method",
        "        return self.session_class()",
    ]
    source_lines[301] = "    def save_session(self, app, session, response):"
    source_lines[302] = "        name = self.get_cookie_name(app)"
    source_lines[303] = "        domain = self.get_cookie_domain(app)"
    source_lines[304] = "        path = self.get_cookie_path(app)"
    source_lines[305] = "        if not session:"
    source_lines[306] = "            response.delete_cookie(name, domain=domain, path=path)"
    source_lines[307] = "            return"
    source_lines[308] = "        val = self.get_signing_serializer(app).dumps(dict(session))"
    source_lines[309] = "        response.set_cookie(name, val, domain=domain, path=path)"
    source_lines[310:340] = [f"        # save option {number}" for number in range(30)]
    content = "\n".join(source_lines) + "\n"
    source_file = RepositoryFile(
        path="src/flask/sessions.py",
        blob_sha=blob,
        commit_sha=commit,
        size_bytes=len(content.encode()),
        content=content,
        citation_url=f"https://github.com/pallets/flask/blob/{commit}/src/flask/sessions.py",
    )
    revision = RepositoryRevision(
        owner="pallets",
        name="flask",
        full_name="pallets/flask",
        default_branch="main",
        commit_sha=commit,
        description="Flask",
        language="Python",
        stars=0,
        forks=0,
        html_url="https://github.com/pallets/flask",
    )

    class CachedRepository:
        async def current_revision(self, _url):
            return revision

        async def inspect(self, *_args, **_kwargs):
            raise AssertionError("offline fixture should satisfy the commit cache")

    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="openai_compatible",
        llm_api_key="offline-test-key",
        llm_base_url="https://llm.example/v1",
        rag_sqlite_path=str(tmp_path / "flask.sqlite3"),
        rag_chunk_lines=80,
        rag_chunk_overlap_lines=15,
        rag_retrieved_chunks=4,
        rag_diagnostics_enabled=True,
        llm_max_context_tokens=4_000,
    )
    index = SQLiteFTSIndex(settings.rag_sqlite_path)
    chunks = chunk_repository_files([source_file], chunk_lines=80, overlap_lines=15, max_chunks=20)
    index.replace_index(revision, chunks, files_indexed=1, total_bytes=len(content))
    captured: dict = {}

    def model_handler(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.content)
        prompt = body["messages"][1]["content"]
        allowed_match = re.search(
            r"(?m)^Allowed citation chunk IDs \(return only these in citation_chunk_ids\): (.*)$",
            prompt,
        )
        allowed = allowed_match.group(1).split(", ") if allowed_match else []
        captured["body"] = body
        captured["allowed"] = allowed
        answer = {
            "answer": (
                "open_session verifies the signed cookie; save_session serializes and sets it."
            ),
            "sufficient_context": True,
            "citation_chunk_ids": allowed[:2],
        }
        return httpx.Response(
            200,
            json={
                "model": "mock-offline-model",
                "choices": [
                    {
                        "message": {"content": __import__("json").dumps(answer)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 500, "completion_tokens": 40, "total_tokens": 540},
            },
            request=request,
        )

    provider = OpenAICompatibleProvider(settings, transport=httpx.MockTransport(model_handler))
    service = RepositoryRAGService(CachedRepository(), provider, settings, index=index)
    response = await service.answer(
        "https://github.com/pallets/flask",
        "How do SecureCookieSessionInterface open_session and save_session manage sessions?",
    )

    assert response.context_sufficient is True
    assert response.error_code is None
    assert len(response.prompt_chunks) >= 2
    assert any(item.start_line == 261 and item.end_line == 340 for item in response.prompt_chunks)
    assert any(item.start_line == 326 and item.end_line == 385 for item in response.prompt_chunks)
    assert len(response.citations) == 2
    assert all(item.commit_sha == commit for item in response.citations)
    assert all(
        f"/blob/{commit}/src/flask/sessions.py#L" in str(item.citation_url)
        for item in response.citations
    )
    assert response.diagnostics is not None
    assert response.diagnostics.returned_citation_ids == captured["allowed"][:2]
    assert response.diagnostics.allowed_citation_ids == captured["allowed"]
    assert response.diagnostics.pipeline_stage == "api_response"
    assert response.diagnostics.provider_status == 200
    assert response.diagnostics.finish_reason == "stop"
    assert response.diagnostics.model == "mock-offline-model"
    assert response.diagnostics.prompt_tokens == 500
    assert response.diagnostics.completion_tokens == 40
    assert response.diagnostics.total_tokens == 540
    assert response.diagnostics.semantic_grounding_evaluated is False
    assert "offline-test-key" not in str(response.diagnostics.model_dump())
    assert "open_session verifies" not in str(response.diagnostics.model_dump())
    assert captured["body"]["response_format"]["type"] == "json_schema"
