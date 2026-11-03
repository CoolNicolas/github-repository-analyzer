from repo_agent.analysis.context import build_repository_context
from repo_agent.schemas import RepositoryFile, RepositoryInspection, RepositoryOverview


def make_inspection(content: str) -> RepositoryInspection:
    commit = "a" * 40
    return RepositoryInspection(
        repository=RepositoryOverview(
            name="sample",
            full_name="owner/sample",
            description="A sample repository",
            html_url="https://github.com/owner/sample",
            stars=3,
            forks=1,
            language="Python",
            default_branch="main",
            commit_sha=commit,
            tree_truncated=False,
            files_in_tree=1,
            files_selected=1,
            source_bytes=len(content.encode("utf-8")),
        ),
        files=[
            RepositoryFile(
                path="src/sample.py",
                blob_sha="b" * 40,
                commit_sha=commit,
                size_bytes=len(content.encode("utf-8")),
                content=content,
                citation_url=f"https://github.com/owner/sample/blob/{commit}/src/sample.py",
            )
        ],
    )


def test_context_contains_metadata_and_citation_fields_within_estimated_token_budget() -> None:
    inspection = make_inspection("def run():\n    return 'sample'\n" * 500)

    context = build_repository_context(inspection, max_context_tokens=256)

    assert context.estimated_tokens <= 256
    assert "Repository: owner/sample" in context.text
    assert "### Source file: src/sample.py" in context.text
    assert "Citation URL: https://github.com/owner/sample/blob/" in context.text
    assert len(context.files) == 1
    assert len(context.text) <= 256 * 4


def test_context_handles_no_source_files() -> None:
    inspection = make_inspection("")
    inspection.files = []

    context = build_repository_context(inspection, max_context_tokens=256)

    assert context.estimated_tokens <= 256
    assert context.files == ()
    assert "Inspected commit SHA" in context.text


def test_source_files_get_context_before_a_large_root_readme() -> None:
    inspection = make_inspection("code line\n" * 2_000)
    source_file = inspection.files[0]
    readme = source_file.model_copy(
        update={
            "path": "README.md",
            "content": "architecture notes " * 2_000,
            "citation_url": (
                f"https://github.com/owner/sample/blob/{source_file.commit_sha}/README.md"
            ),
        }
    )
    inspection.files = [readme, source_file]

    context = build_repository_context(inspection, max_context_tokens=256)

    assert context.files
    assert context.files[0].path == "src/sample.py"
    assert "### Source file: src/sample.py" in context.text


def test_explicit_filename_is_first_and_included_within_context_budget() -> None:
    inspection = make_inspection("long readme text " * 2_000)
    readme = inspection.files[0].model_copy(update={"path": "README.md"})
    target = inspection.files[0].model_copy(
        update={
            "path": "src/flask/config.py",
            "content": "class Config: important setting behavior",
            "citation_url": (
                "https://github.com/owner/sample/blob/"
                f"{inspection.repository.commit_sha}/src/flask/config.py"
            ),
        }
    )
    inspection.files = [readme, target]

    context = build_repository_context(
        inspection,
        max_context_tokens=256,
        question="What does Flask's config.py do?",
    )

    assert context.files[0].path == "src/flask/config.py"
    assert "important setting behavior" in context.text
    assert context.estimated_tokens <= 256
    assert context.missing_references == ()


def test_question_terms_prioritize_relevant_source_over_unrelated_source() -> None:
    inspection = make_inspection("unrelated asset uploader")
    unrelated = inspection.files[0].model_copy(
        update={"path": "src/assets.py", "content": "uploads image assets to storage"}
    )
    relevant = inspection.files[0].model_copy(
        update={
            "path": "src/handlers.py",
            "content": "sessions are signed with a secret serializer",
        }
    )
    inspection.files = [unrelated, relevant]

    context = build_repository_context(
        inspection,
        max_context_tokens=256,
        question="How are sessions signed?",
    )

    assert context.files[0].path == "src/handlers.py"


def test_missing_explicit_file_is_recorded_without_inventing_a_match() -> None:
    context = build_repository_context(
        make_inspection("current source"),
        max_context_tokens=256,
        question="Explain src/flask/config.py",
    )

    assert context.missing_references == ("src/flask/config.py",)
