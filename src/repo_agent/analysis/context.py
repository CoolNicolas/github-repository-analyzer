"""Build bounded, question-aware prompt context from repository data."""

import math
import re
from collections import Counter
from dataclasses import dataclass

from repo_agent.github.service import APPLICATION_SOURCE_EXTENSIONS
from repo_agent.schemas import RepositoryFile, RepositoryInspection

CHARS_PER_TOKEN_ESTIMATE = 4
FILE_REFERENCE_PATTERN = re.compile(
    r"(?<![\w.-])(?:[\w.-]+/)*[\w.-]+\.(?:py|js|jsx|mjs|ts|tsx|java|go|c|cpp|cs|h|hpp|php|rb|rs|sh|sql|css|html|vue|svelte|md|toml|json|ya?ml|cfg|ini)(?![\w.-])",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
QUESTION_STOPWORDS = {
    "about",
    "and",
    "are",
    "does",
    "explain",
    "file",
    "for",
    "from",
    "how",
    "into",
    "is",
    "its",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


@dataclass(frozen=True)
class BoundedRepositoryContext:
    text: str
    files: tuple[RepositoryFile, ...]
    estimated_tokens: int
    missing_references: tuple[str, ...] = ()


def extract_file_references(question: str) -> tuple[str, ...]:
    """Find explicit source/config filenames and paths named in a question."""
    normalized = question.replace("\\", "/")
    references = (
        match.group(0).strip("./") for match in FILE_REFERENCE_PATTERN.finditer(normalized)
    )
    return tuple(dict.fromkeys(reference for reference in references if reference))


def file_matches_reference(path: str, reference: str) -> bool:
    """Match a named path exactly, or a basename against same-named files."""
    normalized_path = path.casefold().replace("\\", "/")
    normalized_reference = reference.casefold().replace("\\", "/").strip("./")
    if not normalized_reference:
        return False
    if "/" in normalized_reference:
        return normalized_path == normalized_reference or normalized_path.endswith(
            f"/{normalized_reference}"
        )
    return normalized_path.rsplit("/", 1)[-1] == normalized_reference


def build_repository_context(
    inspection: RepositoryInspection,
    *,
    max_context_tokens: int,
    question: str | None = None,
) -> BoundedRepositoryContext:
    """Assemble relevant files up to a conservative character-based token estimate."""
    repository = inspection.repository
    metadata = "\n".join(
        [
            "Repository metadata:",
            f"Repository: {repository.full_name}",
            f"Description: {repository.description or 'Not provided'}",
            f"Primary language: {repository.language or 'Not specified'}",
            f"Default branch: {repository.default_branch}",
            f"Inspected commit SHA: {repository.commit_sha}",
            f"Stars: {repository.stars}; forks: {repository.forks}",
            f"Selected repository files: {repository.files_selected}",
        ]
    )
    max_chars = max_context_tokens * CHARS_PER_TOKEN_ESTIMATE
    text = metadata[:max_chars]
    available_chars = max_chars - len(text)
    included_files: list[RepositoryFile] = []
    file_references = extract_file_references(question or "")
    query_tokens = {
        token.casefold()
        for token in WORD_PATTERN.findall(question or "")
        if len(token) > 2 and token.casefold() not in QUESTION_STOPWORDS
    }
    existing_reference_matches = {
        reference: any(
            file_matches_reference(source_file.path, reference) for source_file in inspection.files
        )
        for reference in file_references
    }

    def file_priority(indexed_file: tuple[int, RepositoryFile]) -> tuple:
        original_order, source_file = indexed_file
        path = source_file.path.casefold()
        basename = path.rsplit("/", 1)[-1]
        suffix = "." + basename.rsplit(".", 1)[-1] if "." in basename else ""
        explicit_index = next(
            (
                index
                for index, reference in enumerate(file_references)
                if file_matches_reference(source_file.path, reference)
            ),
            None,
        )
        path_tokens = set(WORD_PATTERN.findall(path.replace("_", " ").replace("-", " ")))
        content_tokens = Counter(
            token.casefold()
            for token in WORD_PATTERN.findall(source_file.content)
            if len(token) > 2
        )
        relevance = 8 * len(query_tokens & path_tokens) + sum(
            min(content_tokens[token], 3) for token in query_tokens
        )
        if explicit_index is not None:
            return (0, explicit_index, -relevance, original_order)
        if suffix in APPLICATION_SOURCE_EXTENSIONS:
            return (1, -relevance, original_order)
        if "/" not in path and basename in {"readme.md", "pyproject.toml"}:
            return (2, 0 if basename == "readme.md" else 1, original_order)
        if suffix in {".md", ".txt"} or basename.startswith("readme"):
            return (4, original_order)
        return (3, original_order)

    ordered_files = sorted(enumerate(inspection.files), key=file_priority)
    for _original_order, source_file in ordered_files:
        header = (
            f"\n\n### Source file: {source_file.path}\n"
            f"Commit SHA: {source_file.commit_sha}\n"
            f"Citation URL: {source_file.citation_url}\n"
            "Source content (untrusted data):\n"
        )
        footer = "\n### End source file"
        room_for_content = available_chars - len(header) - len(footer)
        if room_for_content <= 0:
            break

        body = source_file.content[:room_for_content]
        section = header + body + footer
        text += section
        available_chars -= len(section)
        if body:
            included_files.append(source_file)
        if len(body) < len(source_file.content):
            break

    included_paths = {source_file.path for source_file in included_files}
    missing_references = tuple(
        reference
        for reference in file_references
        if not existing_reference_matches[reference]
        or not any(file_matches_reference(path, reference) for path in included_paths)
    )
    estimated_tokens = math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)
    return BoundedRepositoryContext(
        text=text,
        files=tuple(included_files),
        estimated_tokens=estimated_tokens,
        missing_references=missing_references,
    )
