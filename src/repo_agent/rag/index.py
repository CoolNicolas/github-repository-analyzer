"""Persistent SQLite FTS5 repository index and keyword/code-symbol retrieval."""

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from repo_agent.analysis.context import file_matches_reference
from repo_agent.github.service import RepositoryRevision
from repo_agent.rag.chunking import SourceChunk, stable_chunk_id

QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_SPLIT_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
QUERY_STOPWORDS = {
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
class IndexedChunk:
    path: str
    start_line: int
    end_line: int
    blob_sha: str
    commit_sha: str
    content: str
    rank: float = 0.0

    @property
    def chunk_id(self) -> str:
        return stable_chunk_id(
            self.path,
            self.start_line,
            self.end_line,
            self.commit_sha,
            self.blob_sha,
        )


class SQLiteFTSIndex:
    """Store chunks and FTS5 terms by repository and immutable commit SHA."""

    def __init__(self, database_path: str, *, max_cached_commits: int = 3) -> None:
        self.database_path = database_path
        self.max_cached_commits = max_cached_commits
        if database_path != ":memory:":
            Path(database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS repository_indexes (
                    owner TEXT NOT NULL,
                    name TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    default_branch TEXT NOT NULL,
                    description TEXT,
                    language TEXT,
                    stars INTEGER NOT NULL,
                    forks INTEGER NOT NULL,
                    html_url TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    files_indexed INTEGER NOT NULL,
                    chunks_indexed INTEGER NOT NULL,
                    total_bytes INTEGER NOT NULL,
                    PRIMARY KEY (owner, name, commit_sha)
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS source_chunks (
                    id INTEGER PRIMARY KEY,
                    owner TEXT NOT NULL,
                    name TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    blob_sha TEXT NOT NULL,
                    content TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_source_chunks_repo_commit "
                "ON source_chunks(owner, name, commit_sha)"
            )
            connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS source_chunks_fts USING fts5(
                    path, content, identifiers, tokenize='unicode61'
                )"""
            )

    def has_index(self, owner: str, name: str, commit_sha: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM repository_indexes WHERE owner=? AND name=? AND commit_sha=?",
                (owner.casefold(), name.casefold(), commit_sha),
            ).fetchone()
            return row is not None

    def get_index_metadata(self, owner: str, name: str, commit_sha: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM repository_indexes WHERE owner=? AND name=? AND commit_sha=?",
                (owner.casefold(), name.casefold(), commit_sha),
            ).fetchone()
            return dict(row) if row else None

    def paths(self, owner: str, name: str, commit_sha: str) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT path FROM source_chunks "
                "WHERE owner=? AND name=? AND commit_sha=? ORDER BY path",
                (owner.casefold(), name.casefold(), commit_sha),
            ).fetchall()
            return tuple(row["path"] for row in rows)

    def replace_index(
        self,
        revision: RepositoryRevision,
        chunks: list[SourceChunk],
        *,
        files_indexed: int,
        total_bytes: int,
    ) -> None:
        owner, name = revision.owner.casefold(), revision.name.casefold()
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            self._delete_commit(connection, owner, name, revision.commit_sha)
            connection.execute(
                """INSERT OR REPLACE INTO repository_indexes (
                    owner, name, commit_sha, full_name, default_branch, description,
                    language, stars, forks, html_url, indexed_at, files_indexed,
                    chunks_indexed, total_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    owner,
                    name,
                    revision.commit_sha,
                    revision.full_name,
                    revision.default_branch,
                    revision.description,
                    revision.language,
                    revision.stars,
                    revision.forks,
                    revision.html_url,
                    now,
                    files_indexed,
                    len(chunks),
                    total_bytes,
                ),
            )
            for chunk in chunks:
                cursor = connection.execute(
                    """INSERT INTO source_chunks (
                        owner, name, commit_sha, path, start_line, end_line, blob_sha, content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        owner,
                        name,
                        revision.commit_sha,
                        chunk.path,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.blob_sha,
                        chunk.content,
                    ),
                )
                connection.execute(
                    """INSERT INTO source_chunks_fts (rowid, path, content, identifiers)
                    VALUES (?, ?, ?, ?)""",
                    (cursor.lastrowid, chunk.path, chunk.content, chunk.identifiers),
                )
            self._prune_commits(connection, owner, name)

    def _delete_commit(
        self, connection: sqlite3.Connection, owner: str, name: str, commit_sha: str
    ) -> None:
        rows = connection.execute(
            "SELECT id FROM source_chunks WHERE owner=? AND name=? AND commit_sha=?",
            (owner, name, commit_sha),
        ).fetchall()
        connection.executemany(
            "DELETE FROM source_chunks_fts WHERE rowid=?",
            [(row["id"],) for row in rows],
        )
        connection.execute(
            "DELETE FROM source_chunks WHERE owner=? AND name=? AND commit_sha=?",
            (owner, name, commit_sha),
        )
        connection.execute(
            "DELETE FROM repository_indexes WHERE owner=? AND name=? AND commit_sha=?",
            (owner, name, commit_sha),
        )

    def _prune_commits(self, connection: sqlite3.Connection, owner: str, name: str) -> None:
        rows = connection.execute(
            """SELECT commit_sha FROM repository_indexes
            WHERE owner=? AND name=? ORDER BY indexed_at DESC LIMIT -1 OFFSET ?""",
            (owner, name, self.max_cached_commits),
        ).fetchall()
        for row in rows:
            self._delete_commit(connection, owner, name, row["commit_sha"])

    def retrieve(
        self,
        owner: str,
        name: str,
        commit_sha: str,
        question: str,
        *,
        file_references: tuple[str, ...] = (),
        limit: int = 10,
    ) -> list[IndexedChunk]:
        owner, name = owner.casefold(), name.casefold()
        retrieved: list[IndexedChunk] = []
        seen_ids: set[int] = set()
        deferred_file_rows: list[sqlite3.Row] = []
        tokens, symbol_terms = _query_terms(question)

        with self._connect() as connection:
            all_paths = [
                row["path"]
                for row in connection.execute(
                    "SELECT DISTINCT path FROM source_chunks "
                    "WHERE owner=? AND name=? AND commit_sha=?",
                    (owner, name, commit_sha),
                ).fetchall()
            ]
            for reference in file_references:
                matching_paths = [
                    path for path in all_paths if file_matches_reference(path, reference)
                ]
                for path in matching_paths:
                    rows = connection.execute(
                        """SELECT id, path, start_line, end_line, blob_sha, commit_sha, content
                        FROM source_chunks WHERE owner=? AND name=? AND commit_sha=? AND path=?
                        ORDER BY start_line""",
                        (owner, name, commit_sha, path),
                    ).fetchall()

                    def file_chunk_score(row: sqlite3.Row) -> tuple[int, int, int, int]:
                        content = row["content"]
                        definition_hits = _definition_symbol_hits(content, symbol_terms)
                        exact_symbols = sum(
                            1
                            for symbol in symbol_terms
                            if re.search(
                                rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])",
                                content,
                                re.IGNORECASE,
                            )
                        )
                        term_hits = sum(content.casefold().count(token) for token in tokens)
                        return (
                            -definition_hits,
                            -exact_symbols,
                            -term_hits,
                            row["start_line"],
                        )

                    ranked_file_rows = sorted(rows, key=file_chunk_score)
                    # A filename mention should boost its best evidence without allowing a
                    # long file to consume the whole result limit ahead of matching symbols.
                    preferred_count = max(1, min(2, limit // max(1, len(matching_paths))))
                    for row in ranked_file_rows[:preferred_count]:
                        if len(retrieved) >= limit:
                            break
                        retrieved.append(
                            self._row_to_chunk(
                                row,
                                rank=float(
                                    file_chunk_score(row)[0] * 10_000
                                    + file_chunk_score(row)[1] * 1_000
                                    + file_chunk_score(row)[2]
                                ),
                            )
                        )
                        seen_ids.add(row["id"])
                    deferred_file_rows.extend(ranked_file_rows[preferred_count:])

            if tokens and len(retrieved) < limit:
                match_expression = " OR ".join(f'"{token}"' for token in tokens)
                rows = connection.execute(
                    """SELECT source_chunks.id, source_chunks.path, source_chunks.start_line,
                        source_chunks.end_line, source_chunks.blob_sha, source_chunks.commit_sha,
                        source_chunks.content,
                        bm25(source_chunks_fts, 8.0, 1.0, 5.0) AS rank
                    FROM source_chunks_fts
                    JOIN source_chunks ON source_chunks.id=source_chunks_fts.rowid
                    WHERE source_chunks_fts MATCH ? AND source_chunks.owner=?
                        AND source_chunks.name=? AND source_chunks.commit_sha=?
                    ORDER BY rank LIMIT ?""",
                    (match_expression, owner, name, commit_sha, max(limit * 4, limit)),
                ).fetchall()
                scored_rows = []
                for row in rows:
                    if row["id"] in seen_ids:
                        continue
                    exact_symbols = sum(
                        1
                        for symbol in symbol_terms
                        if re.search(
                            rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])",
                            row["content"],
                            re.IGNORECASE,
                        )
                    )
                    definition_hits = _definition_symbol_hits(row["content"], symbol_terms)
                    scored_rows.append((definition_hits, exact_symbols, float(row["rank"]), row))
                scored_rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
                for definition_hits, exact_symbols, rank, row in scored_rows:
                    if len(retrieved) >= limit:
                        break
                    retrieval_boost = definition_hits * 10_000 + exact_symbols * 1_000
                    retrieved.append(self._row_to_chunk(row, rank=rank - retrieval_boost))
                    seen_ids.add(row["id"])

            # If lexical search found too few matches, fill remaining slots from explicitly
            # requested files in their within-file relevance order.
            for row in deferred_file_rows:
                if len(retrieved) >= limit:
                    break
                if row["id"] in seen_ids:
                    continue
                retrieved.append(self._row_to_chunk(row, rank=0.0))
                seen_ids.add(row["id"])
        return retrieved

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row, *, rank: float) -> IndexedChunk:
        return IndexedChunk(
            path=row["path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            blob_sha=row["blob_sha"],
            commit_sha=row["commit_sha"],
            content=row["content"],
            rank=rank,
        )


def _query_terms(question: str) -> tuple[list[str], set[str]]:
    terms: set[str] = set()
    symbols: set[str] = set()
    for raw in QUERY_TOKEN_PATTERN.findall(question):
        term = raw.casefold()
        if len(term) > 2 and term not in QUERY_STOPWORDS:
            terms.add(term)
        for subterm in raw.split("_"):
            if len(subterm) > 2 and subterm.casefold() not in QUERY_STOPWORDS:
                terms.add(subterm.casefold())
        for subterm in CAMEL_SPLIT_PATTERN.sub(" ", raw).split():
            if len(subterm) > 2 and subterm.casefold() not in QUERY_STOPWORDS:
                terms.add(subterm.casefold())
        if "_" in raw or any(char.isupper() for char in raw[1:]):
            symbols.add(raw)
    return sorted(terms), symbols


def _definition_symbol_hits(content: str, symbols: set[str]) -> int:
    """Count queried Python function/class names defined in this source chunk."""
    hits = 0
    for symbol in symbols:
        if re.search(rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(symbol)}\s*\(", content):
            hits += 1
        elif re.search(rf"(?m)^\s*class\s+{re.escape(symbol)}\b", content):
            hits += 1
    return hits
