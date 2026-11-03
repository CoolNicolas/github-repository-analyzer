"""Line-preserving overlapping chunks for repository source files."""

import hashlib
import re
from dataclasses import dataclass

from repo_agent.schemas import RepositoryFile

IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_BOUNDARY_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@dataclass(frozen=True)
class SourceChunk:
    path: str
    start_line: int
    end_line: int
    blob_sha: str
    commit_sha: str
    content: str
    identifiers: str
    chunk_id: str


def chunk_repository_files(
    files: list[RepositoryFile],
    *,
    chunk_lines: int,
    overlap_lines: int,
    max_chunks: int,
) -> list[SourceChunk]:
    """Split files into bounded, overlapping chunks with 1-based line ranges."""
    overlap = min(overlap_lines, max(0, chunk_lines - 1))
    step = max(1, chunk_lines - overlap)
    chunks: list[SourceChunk] = []

    for source_file in files:
        lines = source_file.content.splitlines(keepends=True)
        if not lines and source_file.content:
            lines = [source_file.content]
        start = 0
        while start < len(lines) and len(chunks) < max_chunks:
            end = min(start + chunk_lines, len(lines))
            content = "".join(lines[start:end])
            identifiers = _identifiers(content)
            chunks.append(
                SourceChunk(
                    path=source_file.path,
                    start_line=start + 1,
                    end_line=end,
                    blob_sha=source_file.blob_sha,
                    commit_sha=source_file.commit_sha,
                    content=content,
                    identifiers=identifiers,
                    chunk_id=stable_chunk_id(
                        source_file.path,
                        start + 1,
                        end,
                        source_file.commit_sha,
                        source_file.blob_sha,
                    ),
                )
            )
            if end >= len(lines):
                break
            start += step
        if len(chunks) >= max_chunks:
            break
    return chunks


def _identifiers(content: str) -> str:
    terms: set[str] = set()
    for identifier in IDENTIFIER_PATTERN.findall(content):
        terms.add(identifier)
        terms.update(identifier.split("_"))
        terms.update(CAMEL_BOUNDARY_PATTERN.sub(" ", identifier).split())
    return " ".join(sorted(term for term in terms if term))


def stable_chunk_id(
    path: str, start_line: int, end_line: int, commit_sha: str, blob_sha: str
) -> str:
    """Build a stable evidence identifier tied to an immutable source snapshot and range."""
    identity = f"{commit_sha}\0{blob_sha}\0{path}\0{start_line}\0{end_line}".encode()
    digest = hashlib.sha256(identity).hexdigest()[:20]
    return f"chunk_{digest}"
