import json

import httpx
import pytest

from repo_agent.providers.base import LLMProviderError
from repo_agent.providers.mock import MockLLMProvider
from repo_agent.providers.openai_compatible import (
    RAG_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    OpenAICompatibleProvider,
    create_provider,
)
from repo_agent.settings import Settings


@pytest.mark.asyncio
async def test_openai_compatible_provider_posts_structured_chat_completion() -> None:
    request_details: dict = {}
    provider_result = {
        "answer": "The module exposes a run function.",
        "context_sufficient": True,
        "citation_paths": ["src/app.py"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        request_details["url"] = str(request.url)
        request_details["authorization"] = request.headers.get("authorization")
        request_details["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(provider_result)}}]},
            request=request,
        )

    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="openai_compatible",
        llm_api_key="test-secret",
        llm_base_url="https://llm.example/v1",
        llm_model="test-model",
        llm_max_output_tokens=123,
    )
    provider = OpenAICompatibleProvider(settings, transport=httpx.MockTransport(handler))
    answer = await provider.answer_question(
        "What does it expose?", "### Source file: src/app.py", max_output_tokens=123
    )

    assert request_details["url"] == "https://llm.example/v1/chat/completions"
    assert request_details["authorization"] == "Bearer test-secret"
    assert request_details["body"]["model"] == "test-model"
    assert request_details["body"]["max_tokens"] == 123
    assert request_details["body"]["response_format"] == {"type": "json_object"}
    assert answer.answer == provider_result["answer"]
    assert answer.citation_paths == ["src/app.py"]


@pytest.mark.asyncio
async def test_provider_normalizes_object_citations_without_trusting_their_urls() -> None:
    response_data = {
        "answer": "The module has an app entry point.",
        "context_sufficient": True,
        "citations": [
            {
                "path": "src/app.py",
                "url": "https://attacker.example/fabricated",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(response_data)}}]},
            request=request,
        )

    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="openai_compatible",
        llm_api_key="test-secret",
        llm_base_url="https://llm.example/v1",
    )
    provider = OpenAICompatibleProvider(settings, transport=httpx.MockTransport(handler))
    result = await provider.answer_question(
        "What does the app do?", "### Source file: src/app.py", max_output_tokens=100
    )

    assert result.citation_paths == ["src/app.py"]


def _provider_with_response(response_payload: dict) -> OpenAICompatibleProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_payload, request=request)

    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="openai_compatible",
        llm_api_key="test-secret",
        llm_base_url="https://llm.example/v1",
    )
    return OpenAICompatibleProvider(settings, transport=httpx.MockTransport(handler))


async def _answer_from_content(
    content: object,
    *,
    finish_reason: str = "stop",
    citation_mode: str = "paths",
) -> object:
    return await _provider_with_response(
        {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
    ).answer_question("Question?", "context", max_output_tokens=100, citation_mode=citation_mode)


@pytest.mark.asyncio
async def test_provider_parses_valid_json_and_json_fenced_in_markdown() -> None:
    payload = {
        "answer": "The app starts in main.py.",
        "context_sufficient": True,
        "citation_paths": ["src/main.py"],
    }
    direct = await _answer_from_content(json.dumps(payload))
    fenced = await _answer_from_content(f"```json\n{json.dumps(payload)}\n```")

    assert direct.answer == fenced.answer
    assert direct.citation_paths == fenced.citation_paths == ["src/main.py"]


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["{bad json}", '```json\n{"answer":\n```'])
async def test_provider_rejects_malformed_json_with_clear_error(content: str) -> None:
    with pytest.raises(LLMProviderError) as error:
        await _answer_from_content(content)

    assert error.value.code == "llm_invalid_response"
    assert "malformed or incomplete" in str(error.value)


@pytest.mark.asyncio
async def test_provider_reports_output_truncation_separately() -> None:
    with pytest.raises(LLMProviderError) as error:
        await _answer_from_content('{"answer":"partial', finish_reason="length")

    assert error.value.code == "llm_truncated_response"
    assert "LLM_MAX_OUTPUT_TOKENS" in str(error.value)


@pytest.mark.asyncio
async def test_provider_requires_answer_but_safely_defaults_missing_optional_fields() -> None:
    with pytest.raises(LLMProviderError) as error:
        await _answer_from_content('{"context_sufficient":true}')
    assert error.value.code == "llm_invalid_response"

    partial = await _answer_from_content('{"answer":"Not enough evidence."}')
    assert partial.context_sufficient is False
    assert partial.citation_paths == []


@pytest.mark.asyncio
async def test_provider_normalizes_alternate_citation_formats_without_ambiguous_paths() -> None:
    result = await _answer_from_content(
        json.dumps(
            {
                "response": "The behavior is in the app module.",
                "sufficient_context": True,
                "sources": [
                    {"file_path": "src/app.py", "url": "https://invalid.example"},
                    {"path": "src/a.py", "source_path": "src/b.py"},
                    {"url": "https://invalid.example/src/fake.py"},
                ],
            }
        )
    )

    assert result.answer == "The behavior is in the app module."
    assert result.context_sufficient is True
    assert result.citation_paths == ["src/app.py"]


@pytest.mark.asyncio
async def test_provider_preserves_cited_line_ranges() -> None:
    result = await _answer_from_content(
        json.dumps(
            {
                "answer": "A verified claim.",
                "context_sufficient": True,
                "citations": [{"path": "src/app.py", "start_line": 12, "end_line": 18}],
            }
        )
    )

    assert result.citations[0].path == "src/app.py"
    assert result.citations[0].start_line == 12
    assert result.citations[0].end_line == 18


def test_system_prompt_requires_grounding_and_explicit_uncertainty() -> None:
    rag_prompt = " ".join(RAG_SYSTEM_PROMPT.split())
    assert "label inferences explicitly" in SYSTEM_PROMPT
    assert "identify unsupported claims as unsupported" in SYSTEM_PROMPT
    assert "repository evidence included in this prompt" in rag_prompt
    assert "citation_chunk_ids" in rag_prompt
    assert "Do not return file paths, URLs, line numbers" in rag_prompt
    assert "distinguish their definitions and bodies from calls" in rag_prompt
    assert "distinguish integrity/authenticity from confidentiality" in rag_prompt
    assert "does not hide the signed data" in rag_prompt
    assert "Use these distinctions to interpret evidence, never as evidence" in rag_prompt
    assert "Check each factual statement" in rag_prompt
    assert "Resolve apparent conflicts only when the supplied evidence does" in rag_prompt


@pytest.mark.asyncio
async def test_provider_preserves_model_marked_unsupported_or_conflicting_evidence() -> None:
    result = await _answer_from_content(
        json.dumps(
            {
                "answer": "The excerpts do not establish whether clients can view this value.",
                "sufficient_context": False,
                "citation_chunk_ids": [],
            }
        ),
        citation_mode="chunk_ids",
    )

    assert result.sufficient_context is False
    assert result.citation_chunk_ids == []
    assert "do not establish" in result.answer


@pytest.mark.asyncio
async def test_provider_preserves_stable_chunk_id_citations() -> None:
    chunk_ids = ["chunk_312d84ff990bab36e7b9", "chunk_fb3e3e661c73734a5e7c"]
    result = await _answer_from_content(
        json.dumps(
            {
                "answer": "The evidence is in the selected chunk.",
                "sufficient_context": True,
                "citation_chunk_ids": chunk_ids,
            }
        ),
        citation_mode="chunk_ids",
    )

    assert result.citation_chunk_ids == chunk_ids
    assert result.citation_paths == []


@pytest.mark.asyncio
async def test_provider_treats_absent_canonical_citations_as_missing_not_as_path_format() -> None:
    result = await _answer_from_content(
        json.dumps(
            {
                "answer": "The source does not establish this.",
                "sufficient_context": False,
                "citation_chunk_ids": [],
            }
        ),
        citation_mode="chunk_ids",
    )

    assert result.citation_chunk_ids == []
    assert result.citation_paths == []


@pytest.mark.asyncio
async def test_rag_provider_requests_strict_schema_for_canonical_citation_field() -> None:
    request_details: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        request_details["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "Evidence-based answer.",
                                    "sufficient_context": True,
                                    "citation_chunk_ids": ["chunk_312d84ff990bab36e7b9"],
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            request=request,
        )

    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="openai_compatible",
        llm_api_key="test-secret",
        llm_base_url="https://llm.example/v1",
    )
    result = await OpenAICompatibleProvider(
        settings, transport=httpx.MockTransport(handler)
    ).answer_question(
        "Question?", "Allowed IDs in context", max_output_tokens=100, citation_mode="chunk_ids"
    )

    response_format = request_details["body"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert "citation_chunk_ids" in response_format["json_schema"]["schema"]["properties"]
    assert result.citation_chunk_ids == ["chunk_312d84ff990bab36e7b9"]


@pytest.mark.asyncio
async def test_provider_rejects_old_path_based_citations_in_rag_mode() -> None:
    with pytest.raises(LLMProviderError) as error:
        await _answer_from_content(
            json.dumps(
                {
                    "answer": "An answer.",
                    "sufficient_context": True,
                    "citation_paths": ["src/flask/sessions.py"],
                }
            ),
            citation_mode="chunk_ids",
        )

    assert error.value.code == "llm_invalid_citation_format"
    assert "citation_chunk_ids" in str(error.value)


@pytest.mark.asyncio
async def test_provider_rejects_noncanonical_context_sufficient_field_in_rag_mode() -> None:
    with pytest.raises(LLMProviderError) as error:
        await _answer_from_content(
            json.dumps(
                {
                    "answer": "An answer.",
                    "context_sufficient": True,
                    "citation_chunk_ids": ["chunk_312d84ff990bab36e7b9"],
                }
            ),
            citation_mode="chunk_ids",
        )

    assert error.value.code == "llm_missing_fields"
    assert "sufficient_context" in str(error.value)


@pytest.mark.asyncio
async def test_provider_rejects_path_and_line_objects_in_rag_mode() -> None:
    with pytest.raises(LLMProviderError) as error:
        await _answer_from_content(
            json.dumps(
                {
                    "answer": "An answer.",
                    "sufficient_context": True,
                    "citations": [
                        {"path": "src/flask/sessions.py", "start_line": 290, "end_line": 301}
                    ],
                }
            ),
            citation_mode="chunk_ids",
        )

    assert error.value.code == "llm_invalid_citation_format"


@pytest.mark.asyncio
async def test_provider_retries_without_json_mode_when_unsupported() -> None:
    request_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        request_bodies.append(body)
        if "response_format" in body:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format is not supported"}},
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"answer":"Supported fallback.","context_sufficient":false}'
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            request=request,
        )

    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="openai_compatible",
        llm_api_key="test-secret",
        llm_base_url="https://llm.example/v1",
    )
    result = await OpenAICompatibleProvider(
        settings, transport=httpx.MockTransport(handler)
    ).answer_question("Question?", "context", max_output_tokens=100)

    assert len(request_bodies) == 2
    assert "response_format" in request_bodies[0]
    assert "response_format" not in request_bodies[1]
    assert result.answer == "Supported fallback."


@pytest.mark.asyncio
async def test_provider_downgrades_unsupported_json_schema_to_json_mode() -> None:
    response_formats: list[dict | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        response_format = body.get("response_format")
        response_formats.append(response_format)
        if response_format and response_format.get("type") == "json_schema":
            return httpx.Response(
                400,
                json={"error": {"message": "json_schema is not supported"}},
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "A response.",
                                    "sufficient_context": False,
                                    "citation_chunk_ids": [],
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            request=request,
        )

    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="openai_compatible",
        llm_api_key="test-secret",
        llm_base_url="https://llm.example/v1",
    )
    result = await OpenAICompatibleProvider(
        settings, transport=httpx.MockTransport(handler)
    ).answer_question("Question?", "context", max_output_tokens=100, citation_mode="chunk_ids")

    assert [item["type"] if item else None for item in response_formats] == [
        "json_schema",
        "json_object",
    ]
    assert result.citation_chunk_ids == []


def test_provider_factory_uses_mock_unless_live_inference_is_enabled() -> None:
    settings = Settings(
        _env_file=None,
        llm_enabled=False,
        llm_provider="openai_compatible",
        llm_base_url="https://llm.example/v1",
    )
    assert isinstance(create_provider(settings), MockLLMProvider)


def test_remote_live_provider_requires_key_and_https() -> None:
    no_key = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="openai_compatible",
        llm_base_url="https://llm.example/v1",
    )
    with pytest.raises(LLMProviderError, match="LLM_API_KEY"):
        OpenAICompatibleProvider(no_key)

    blank_key = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="openai_compatible",
        llm_api_key="",
        llm_base_url="https://llm.example/v1",
    )
    with pytest.raises(LLMProviderError, match="LLM_API_KEY"):
        OpenAICompatibleProvider(blank_key)

    insecure = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="openai_compatible",
        llm_api_key="test-secret",
        llm_base_url="http://llm.example/v1",
    )
    with pytest.raises(LLMProviderError, match="HTTPS"):
        OpenAICompatibleProvider(insecure)


@pytest.mark.asyncio
async def test_provider_errors_do_not_expose_response_body_or_api_key() -> None:
    secret = "never-print-this-key"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=f"failed with {secret}", request=request)

    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="openai_compatible",
        llm_api_key=secret,
        llm_base_url="https://llm.example/v1",
    )
    provider = OpenAICompatibleProvider(settings, transport=httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError) as error:
        await provider.analyze_repository("context", max_output_tokens=100)
    assert secret not in str(error.value)
    assert "failed with" not in str(error.value)
