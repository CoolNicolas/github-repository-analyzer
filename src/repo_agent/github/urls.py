"""Strict parsing for public GitHub repository URLs."""

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

OWNER_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$|^[A-Za-z0-9]$")
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class InvalidRepositoryURL(ValueError):
    """Raised when a URL is not a public github.com owner/repository URL."""


@dataclass(frozen=True)
class RepositoryReference:
    owner: str
    name: str


def parse_repository_url(value: str) -> RepositoryReference:
    """Parse a public repository URL without accepting arbitrary hosts or paths."""
    candidate = value.strip()
    if not candidate or len(candidate) > 2_048:
        raise InvalidRepositoryURL("Enter a valid public GitHub repository URL.")

    try:
        parsed = urlsplit(candidate)
        # Accessing .port/.hostname can raise for malformed authorities.
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise InvalidRepositoryURL("Enter a valid public GitHub repository URL.") from exc

    if parsed.scheme.lower() != "https" or hostname not in {"github.com", "www.github.com"}:
        raise InvalidRepositoryURL("Use an HTTPS URL hosted on github.com.")
    if parsed.username or parsed.password or port is not None or parsed.query or parsed.fragment:
        raise InvalidRepositoryURL(
            "The repository URL must not contain credentials, a port, or extra parameters."
        )

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        raise InvalidRepositoryURL(
            "Use a repository URL in the form https://github.com/owner/repository."
        )

    owner, name = parts
    if name.lower().endswith(".git"):
        name = name[:-4]
    if (
        not OWNER_PATTERN.fullmatch(owner)
        or not REPOSITORY_PATTERN.fullmatch(name)
        or name in {".", ".."}
    ):
        raise InvalidRepositoryURL("The owner or repository name contains unsupported characters.")

    return RepositoryReference(owner=owner, name=name)
