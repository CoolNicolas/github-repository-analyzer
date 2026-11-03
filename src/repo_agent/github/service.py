"""Fetch bounded repository metadata and source through GitHub's REST API."""

import base64
import binascii
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import httpx

from repo_agent.github.urls import parse_repository_url
from repo_agent.schemas import RepositoryFile, RepositoryInspection, RepositoryOverview
from repo_agent.settings import Settings

TEXT_EXTENSIONS = {
    ".c",
    ".cfg",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".svelte",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
    ".lock",
    ".ini",
    ".gradle",
    ".dockerfile",
    ".properties",
    ".makefile",
}
APPLICATION_SOURCE_EXTENSIONS = {
    ".c",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".mjs",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".svelte",
    ".ts",
    ".tsx",
    ".vue",
}
SPECIAL_TEXT_NAMES = {
    "dockerfile",
    "makefile",
    "license",
    "licence",
    "procfile",
    "readme",
    "requirements.txt",
    "gemfile",
    "rakefile",
    "justfile",
    "cmakelists.txt",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".env.example",
    ".npmrc",
    ".prettierrc",
    ".python-version",
}
GITHUB_API_BASE_URL = "https://api.github.com"


@dataclass(frozen=True)
class RepositoryRevision:
    owner: str
    name: str
    full_name: str
    default_branch: str
    commit_sha: str
    description: str | None
    language: str | None
    stars: int
    forks: int
    html_url: str


class GitHubError(Exception):
    """A user-safe error returned by the GitHub integration."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        code: str = "github_error",
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retry_after_seconds = retry_after_seconds


class GitHubRepositoryService:
    """Read-only GitHub repository inspector with conservative resource limits."""

    def __init__(
        self, settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None
    ):
        self.settings = settings or Settings()
        self.transport = transport

    async def current_revision(self, repository_url: str) -> RepositoryRevision:
        """Read repository metadata and default-branch SHA without fetching files."""
        reference = parse_repository_url(repository_url)
        owner = quote(reference.owner, safe="")
        name = quote(reference.name, safe="")
        timeout = httpx.Timeout(self.settings.github_request_timeout_seconds)
        headers = self._request_headers()
        async with httpx.AsyncClient(
            base_url=GITHUB_API_BASE_URL,
            timeout=timeout,
            transport=self.transport,
            headers=headers,
        ) as client:
            metadata = await self._get_json(client, f"/repos/{owner}/{name}")
            default_branch = metadata.get("default_branch")
            if not isinstance(default_branch, str) or not default_branch:
                raise GitHubError("GitHub did not return a default branch for this repository.")
            branch = await self._get_json(
                client,
                f"/repos/{owner}/{name}/branches/{quote(default_branch, safe='')}",
            )
            commit_sha = branch.get("commit", {}).get("sha")
            if not isinstance(commit_sha, str) or not commit_sha:
                raise GitHubError("GitHub did not return a commit for the default branch.")

        return RepositoryRevision(
            owner=reference.owner,
            name=reference.name,
            full_name=str(metadata.get("full_name") or f"{reference.owner}/{reference.name}"),
            default_branch=default_branch,
            commit_sha=commit_sha,
            description=metadata.get("description"),
            language=metadata.get("language"),
            stars=int(metadata.get("stargazers_count") or 0),
            forks=int(metadata.get("forks_count") or 0),
            html_url=str(
                metadata.get("html_url") or f"https://github.com/{reference.owner}/{reference.name}"
            ),
        )

    async def inspect(
        self,
        repository_url: str,
        *,
        priority_paths: Sequence[str] = (),
        max_files: int | None = None,
        max_file_bytes: int | None = None,
        max_total_bytes: int | None = None,
        max_requests: int | None = None,
    ) -> RepositoryInspection:
        effective_max_files = self.settings.github_max_files if max_files is None else max_files
        effective_max_file_bytes = (
            self.settings.github_max_file_bytes if max_file_bytes is None else max_file_bytes
        )
        effective_max_total_bytes = (
            self.settings.github_max_total_bytes if max_total_bytes is None else max_total_bytes
        )
        if max_requests is not None and max_requests < 3:
            raise ValueError("A repository inspection requires at least three GitHub API requests.")
        max_content_requests = (
            effective_max_files if max_requests is None else max(0, max_requests - 3)
        )
        selection_limit = min(effective_max_files, max_content_requests)
        reference = parse_repository_url(repository_url)
        owner = quote(reference.owner, safe="")
        name = quote(reference.name, safe="")
        timeout = httpx.Timeout(self.settings.github_request_timeout_seconds)

        headers = self._request_headers()
        async with httpx.AsyncClient(
            base_url=GITHUB_API_BASE_URL,
            timeout=timeout,
            transport=self.transport,
            headers=headers,
        ) as client:
            metadata = await self._get_json(client, f"/repos/{owner}/{name}")
            default_branch = metadata.get("default_branch")
            if not isinstance(default_branch, str) or not default_branch:
                raise GitHubError("GitHub did not return a default branch for this repository.")

            branch_path = quote(default_branch, safe="")
            branch = await self._get_json(client, f"/repos/{owner}/{name}/branches/{branch_path}")
            commit_sha = branch.get("commit", {}).get("sha")
            if not isinstance(commit_sha, str) or not commit_sha:
                raise GitHubError("GitHub did not return a commit for the default branch.")

            tree = await self._get_json(
                client,
                f"/repos/{owner}/{name}/git/trees/{quote(commit_sha, safe='')}",
                params={"recursive": "1"},
            )
            tree_entries = tree.get("tree", [])
            if not isinstance(tree_entries, list):
                raise GitHubError("GitHub returned an invalid repository file tree.")

            selected = self._select_text_blobs(
                tree_entries,
                language=metadata.get("language"),
                priority_paths=priority_paths,
                max_files=selection_limit,
                max_file_bytes=effective_max_file_bytes,
                max_total_bytes=effective_max_total_bytes,
            )
            files: list[RepositoryFile] = []
            total_bytes = 0
            for entry in selected:
                estimated_size = entry.get("size", 0)
                if not isinstance(estimated_size, int) or estimated_size < 0:
                    continue
                if estimated_size > effective_max_file_bytes:
                    continue
                if total_bytes + estimated_size > effective_max_total_bytes:
                    continue
                path = entry["path"]
                encoded_path = quote(path, safe="/")
                content_data = await self._get_json(
                    client,
                    f"/repos/{owner}/{name}/contents/{encoded_path}",
                    params={"ref": commit_sha},
                )
                file_content = self._decode_content(content_data)
                if file_content is None:
                    continue
                content_bytes = file_content.encode("utf-8")
                if len(content_bytes) > effective_max_file_bytes:
                    continue
                if total_bytes + len(content_bytes) > effective_max_total_bytes:
                    continue
                if self._looks_binary(file_content):
                    continue
                blob_sha = entry.get("sha")
                if not isinstance(blob_sha, str):
                    continue
                citation_url = f"https://github.com/{reference.owner}/{reference.name}/blob/{commit_sha}/{path}"
                files.append(
                    RepositoryFile(
                        path=path,
                        blob_sha=blob_sha,
                        commit_sha=commit_sha,
                        size_bytes=len(content_bytes),
                        content=file_content,
                        citation_url=citation_url,
                    )
                )
                total_bytes += len(content_bytes)
                if len(files) >= effective_max_files:
                    break

        repository = RepositoryOverview(
            name=str(metadata.get("name") or reference.name),
            full_name=str(metadata.get("full_name") or f"{reference.owner}/{reference.name}"),
            description=metadata.get("description"),
            html_url=metadata.get("html_url")
            or f"https://github.com/{reference.owner}/{reference.name}",
            stars=int(metadata.get("stargazers_count") or 0),
            forks=int(metadata.get("forks_count") or 0),
            language=metadata.get("language"),
            default_branch=default_branch,
            commit_sha=commit_sha,
            tree_truncated=bool(tree.get("truncated", False)),
            files_in_tree=len(tree_entries),
            files_selected=len(files),
            source_bytes=total_bytes,
        )
        return RepositoryInspection(repository=repository, files=files)

    async def _get_json(
        self,
        client: httpx.AsyncClient,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> dict:
        try:
            response = await client.get(path, params=params)
        except httpx.TimeoutException as exc:
            raise GitHubError(
                "GitHub request timed out. Please try again.", code="timeout"
            ) from exc
        except httpx.RequestError as exc:
            raise GitHubError(
                "Could not connect to GitHub. Check your network and try again.",
                code="network_error",
            ) from exc

        if response.status_code == 404:
            raise GitHubError(
                "Repository or file not found. Check the URL and that the repository is public.",
                status_code=404,
                code="not_found",
            )
        if response.status_code == 401:
            raise GitHubError(
                "GitHub rejected the configured credentials. Check the GitHub token and try again.",
                status_code=502,
                code="authentication_failed",
            )
        if response.status_code in {403, 429}:
            remaining = response.headers.get("x-ratelimit-remaining")
            if response.status_code == 429 or remaining == "0":
                retry_after = self._rate_limit_wait(response)
                wait_text = (
                    f" Please wait about {retry_after} seconds before trying again."
                    if retry_after is not None
                    else " Please wait before trying again."
                )
                raise GitHubError(
                    "GitHub API rate limit reached." + wait_text,
                    status_code=429,
                    code="rate_limited",
                    retry_after_seconds=retry_after,
                )
            raise GitHubError(
                "GitHub refused the request. Please try again later.",
                status_code=502,
                code="upstream_forbidden",
            )
        if response.is_error:
            raise GitHubError(
                f"GitHub returned an error (HTTP {response.status_code}).", status_code=502
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise GitHubError("GitHub returned an unreadable response.") from exc
        if not isinstance(data, dict):
            raise GitHubError("GitHub returned an unexpected response format.")
        return data

    def _request_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.settings.github_token is not None:
            token = self.settings.github_token.get_secret_value().strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _rate_limit_wait(response: httpx.Response) -> int | None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return max(0, math.ceil(float(retry_after)))
            except (ValueError, OverflowError):
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=UTC)
                    return max(
                        0,
                        math.ceil((retry_at - datetime.now(UTC)).total_seconds()),
                    )
                except (TypeError, ValueError, OverflowError):
                    pass

        reset_at = response.headers.get("x-ratelimit-reset")
        if reset_at:
            try:
                return max(
                    0,
                    math.ceil(float(reset_at) - datetime.now(UTC).timestamp()),
                )
            except (TypeError, ValueError, OverflowError):
                pass
        return None

    def _select_text_blobs(
        self,
        entries: list[dict],
        *,
        language: str | None = None,
        priority_paths: Sequence[str] = (),
        max_files: int | None = None,
        max_file_bytes: int | None = None,
        max_total_bytes: int | None = None,
    ) -> list[dict]:
        effective_max_files = self.settings.github_max_files if max_files is None else max_files
        effective_max_file_bytes = (
            self.settings.github_max_file_bytes if max_file_bytes is None else max_file_bytes
        )
        effective_max_total_bytes = (
            self.settings.github_max_total_bytes if max_total_bytes is None else max_total_bytes
        )
        candidates = [
            entry
            for entry in entries
            if entry.get("type") == "blob"
            and isinstance(entry.get("path"), str)
            and isinstance(entry.get("size"), int)
            and 0 <= entry["size"] <= effective_max_file_bytes
            and entry["size"] <= effective_max_total_bytes
            and self._is_text_candidate(entry["path"])
        ]

        is_python_repository = (language or "").casefold().startswith("python") or any(
            entry["path"].lower().endswith(".py") for entry in candidates
        )

        def priority(entry: dict) -> tuple:
            path = entry["path"]
            basename = path.rsplit("/", 1)[-1].lower()
            normalized_path = path.casefold()
            is_root = "/" not in path
            for index, requested_path in enumerate(priority_paths):
                requested_path = requested_path.casefold().replace("\\", "/").strip("./")
                if not requested_path:
                    continue
                matches = (
                    normalized_path == requested_path
                    or normalized_path.endswith(f"/{requested_path}")
                    if "/" in requested_path
                    else basename == requested_path
                )
                if matches:
                    return (-1, index, normalized_path)

            # Keep these two root files for repository context, then spend the
            # remaining budget on application code before supporting material.
            if is_root and basename in {"readme.md", "pyproject.toml"}:
                return (0, 0 if basename == "readme.md" else 1, 0, 0, normalized_path)

            suffix = "." + basename.rsplit(".", 1)[-1] if "." in basename else ""
            if suffix in APPLICATION_SOURCE_EXTENSIONS:
                under_src = normalized_path.startswith("src/")
                is_test_path = any(
                    component in {"test", "tests"} for component in normalized_path.split("/")[:-1]
                )

                if is_python_repository and suffix == ".py":
                    entrypoint_order = {
                        "__main__.py": 0,
                        "main.py": 1,
                        "app.py": 2,
                        "application.py": 3,
                        "cli.py": 4,
                        "asgi.py": 5,
                        "wsgi.py": 6,
                        "__init__.py": 7,
                    }
                    return (
                        1,
                        0 if under_src else 1,
                        1 if is_test_path else 0,
                        entrypoint_order.get(basename, 8),
                        normalized_path,
                    )

                general_entrypoints = {
                    "main.go": 0,
                    "main.java": 0,
                    "index.js": 0,
                    "index.ts": 0,
                    "app.js": 1,
                    "app.ts": 1,
                    "server.js": 2,
                    "server.ts": 2,
                }
                return (
                    1,
                    0 if under_src else 1,
                    general_entrypoints.get(basename, 3),
                    0,
                    normalized_path,
                )

            if normalized_path.startswith((".github/workflows/", ".circleci/", ".gitlab/")):
                return (3, 0, 0, 0, normalized_path)
            if suffix in {".md", ".txt"} or basename.startswith("readme"):
                return (4, 0, 0, 0, normalized_path)
            return (2, 0, 0, 0, normalized_path)

        candidates.sort(key=priority)
        # This also caps the number of content API requests for very large trees.
        return candidates[:effective_max_files]

    @staticmethod
    def _is_text_candidate(path: str) -> bool:
        basename = path.rsplit("/", 1)[-1].lower()
        suffix = "." + basename.rsplit(".", 1)[-1] if "." in basename else ""
        return basename in SPECIAL_TEXT_NAMES or suffix in TEXT_EXTENSIONS

    @staticmethod
    def _decode_content(data: dict) -> str | None:
        if data.get("type") != "file" or data.get("encoding") != "base64":
            return None
        content = data.get("content")
        if not isinstance(content, str):
            return None
        try:
            decoded = base64.b64decode(content, validate=False)
            return decoded.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            return None

    @staticmethod
    def _looks_binary(content: str) -> bool:
        if "\x00" in content:
            return True
        unusual = sum(ord(char) < 32 and char not in "\n\r\t\f\b" for char in content)
        return bool(content) and unusual / len(content) > 0.02
