from repo_agent.rag.chunking import chunk_repository_files
from repo_agent.schemas import RepositoryFile


def test_chunks_preserve_overlapping_line_ranges_and_revision_metadata() -> None:
    source = RepositoryFile(
        path="src/module.py",
        blob_sha="b" * 40,
        commit_sha="c" * 40,
        size_bytes=20,
        content="".join(f"line_{number}\n" for number in range(1, 11)),
        citation_url="https://github.com/o/r/blob/" + "c" * 40 + "/src/module.py",
    )

    chunks = chunk_repository_files([source], chunk_lines=4, overlap_lines=2, max_chunks=10)

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 4),
        (3, 6),
        (5, 8),
        (7, 10),
    ]
    assert all(chunk.blob_sha == "b" * 40 for chunk in chunks)
    assert all(chunk.commit_sha == "c" * 40 for chunk in chunks)
    assert "line_3" in chunks[1].content
    assert len(chunk_repository_files([source], chunk_lines=4, overlap_lines=2, max_chunks=2)) == 2
