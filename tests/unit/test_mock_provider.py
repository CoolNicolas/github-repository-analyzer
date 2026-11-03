import pytest

from repo_agent.providers.mock import MockLLMProvider


@pytest.mark.asyncio
async def test_mock_provider_is_deterministic_and_cites_context_paths() -> None:
    provider = MockLLMProvider()
    context = "Repository: sample\n### Source file: src/app.py\nprint('local')\n"
    answer = await provider.answer_question("What is this?", context, max_output_tokens=100)
    assert answer.answer == "Mock answer to: What is this?"
    assert answer.context_sufficient is True
    assert answer.citation_paths == ["src/app.py"]


@pytest.mark.asyncio
async def test_mock_provider_marks_missing_source_context_insufficient() -> None:
    provider = MockLLMProvider()
    answer = await provider.answer_question(
        "What does the code do?", "Repository metadata only", max_output_tokens=100
    )
    assert answer.context_sufficient is False
    assert "insufficient" in answer.answer
