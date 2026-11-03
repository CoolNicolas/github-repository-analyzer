import pytest

from repo_agent.github.urls import InvalidRepositoryURL, parse_repository_url


@pytest.mark.parametrize(
    ("url", "owner", "name"),
    [
        ("https://github.com/octocat/Hello-World", "octocat", "Hello-World"),
        (" https://www.github.com/pallets/flask.git/ ", "pallets", "flask"),
        ("https://github.com/a/repo.name", "a", "repo.name"),
    ],
)
def test_parse_repository_url(url: str, owner: str, name: str) -> None:
    reference = parse_repository_url(url)
    assert (reference.owner, reference.name) == (owner, name)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "http://github.com/owner/repo",
        "https://github.com.evil.example/owner/repo",
        "https://example.com/owner/repo",
        "https://github.com/owner/repo/issues",
        "https://user:pass@github.com/owner/repo",
        "https://github.com:443/owner/repo",
        "https://github.com/../repo",
        "https://github.com/owner/repo?tab=readme",
    ],
)
def test_rejects_unsafe_or_non_repository_urls(url: str) -> None:
    with pytest.raises(InvalidRepositoryURL):
        parse_repository_url(url)
