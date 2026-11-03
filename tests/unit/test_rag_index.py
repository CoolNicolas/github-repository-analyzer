from repo_agent.github.service import RepositoryRevision
from repo_agent.rag.chunking import chunk_repository_files
from repo_agent.rag.index import SQLiteFTSIndex
from repo_agent.schemas import RepositoryFile


def _source_file(path: str, content: str) -> RepositoryFile:
    return RepositoryFile(
        path=path,
        blob_sha="b" * 40,
        commit_sha="c" * 40,
        size_bytes=len(content.encode()),
        content=content,
        citation_url=f"https://github.com/o/r/blob/{'c' * 40}/{path}",
    )


def _revision() -> RepositoryRevision:
    return RepositoryRevision(
        owner="o",
        name="r",
        full_name="o/r",
        default_branch="main",
        commit_sha="c" * 40,
        description=None,
        language="Python",
        stars=0,
        forks=0,
        html_url="https://github.com/o/r",
    )


def test_fts_retrieval_prioritizes_exact_file_and_code_identifiers(tmp_path) -> None:
    index = SQLiteFTSIndex(str(tmp_path / "index.sqlite3"))
    files = [
        _source_file(
            "src/flask/config.py",
            "class Config:\n    def from_mapping(self, mapping):\n        self.update(mapping)\n",
        ),
        _source_file(
            "src/flask/uploads.py", "def save_upload(stream):\n    return stream.read()\n"
        ),
        _source_file("docs/overview.md", "A general overview of the project.\n"),
    ]
    chunks = chunk_repository_files(files, chunk_lines=20, overlap_lines=0, max_chunks=20)
    index.replace_index(_revision(), chunks, files_indexed=len(files), total_bytes=100)

    by_file = index.retrieve(
        "o",
        "r",
        "c" * 40,
        "What does config.py Config.from_mapping do?",
        file_references=("config.py",),
        limit=5,
    )
    by_identifier = index.retrieve("o", "r", "c" * 40, "Explain Config.from_mapping", limit=5)

    assert by_file[0].path == "src/flask/config.py"
    assert any("from_mapping" in chunk.content for chunk in by_identifier)
    assert index.has_index("O", "R", "c" * 40)
    assert index.get_index_metadata("o", "r", "c" * 40)["files_indexed"] == 3


def test_fts_returns_no_chunks_for_irrelevant_terms(tmp_path) -> None:
    index = SQLiteFTSIndex(str(tmp_path / "index.sqlite3"))
    source = _source_file("src/app.py", "def start_server():\n    return True\n")
    chunks = chunk_repository_files([source], chunk_lines=20, overlap_lines=0, max_chunks=10)
    index.replace_index(_revision(), chunks, files_indexed=1, total_bytes=40)

    assert index.retrieve("o", "r", "c" * 40, "quasar xenolith", limit=5) == []


def test_session_identifiers_retrieve_requested_file_and_helper_evidence(tmp_path) -> None:
    sessions_lines = [f"# session filler {line_number}\n" for line_number in range(1, 471)]
    sessions_lines[325] = "class SecureCookieSessionInterface:\n"
    sessions_lines[326] = "    def open_session(self, app, request):\n"
    sessions_lines[327] = "        serializer = self.get_signing_serializer(app)\n"
    sessions_lines[390] = "    def save_session(self, app, session, response):\n"
    sessions_lines[391] = "        name = self.get_cookie_name(app)\n"
    sessions_lines[410] = "        app.session_interface = self\n"

    helper_lines = [f"# helper filler {line_number}\n" for line_number in range(1, 471)]
    helper_lines[325] = "def get_signing_serializer(app):\n"
    helper_lines[326] = "    return URLSafeTimedSerializer(app.secret_key, salt='cookie-session')\n"
    helper_lines[327] = "    # The payload is signed so changes can be detected.\n"
    helper_lines[328] = "    # Signing does not encrypt the cookie contents.\n"

    files = [
        _source_file("src/flask/sessions.py", "".join(sessions_lines)),
        _source_file("src/flask/helpers.py", "".join(helper_lines)),
    ]
    chunks = chunk_repository_files(files, chunk_lines=80, overlap_lines=15, max_chunks=100)
    index = SQLiteFTSIndex(str(tmp_path / "sessions.sqlite3"))
    index.replace_index(_revision(), chunks, files_indexed=len(files), total_bytes=10_000)

    retrieved = index.retrieve(
        "o",
        "r",
        "c" * 40,
        "How do SecureCookieSessionInterface open_session and save_session use "
        "session_interface and get_signing_serializer? Are cookies encrypted?",
        file_references=("sessions.py",),
        limit=10,
    )
    retrieved_ranges = {(chunk.path, chunk.start_line, chunk.end_line) for chunk in retrieved}

    assert ("src/flask/sessions.py", 326, 405) in retrieved_ranges
    assert ("src/flask/sessions.py", 391, 470) in retrieved_ranges
    assert (
        "src/flask/helpers.py",
        326,
        405,
    ) in retrieved_ranges
    assert any(
        "URLSafeTimedSerializer" in chunk.content
        for chunk in retrieved
        if chunk.path.endswith("helpers.py")
    )


def test_session_method_definitions_win_over_mentions_and_keep_overlapping_context(
    tmp_path,
) -> None:
    lines = ["# unrelated filler\n" for _ in range(470)]
    lines[40] = "# open_session is mentioned in a comment, not defined here\n"
    lines[289] = "    def open_session(self, app, request):\n"
    lines[290] = '        """Load a session from the signed cookie."""\n'
    lines[291] = "        serializer = self.get_signing_serializer(app)\n"
    lines[292] = "        if serializer is None:\n"
    lines[293] = "            return None\n"
    lines[294] = "        return serializer.loads(request.cookies.get(self.get_cookie_name(app)))\n"
    lines[301] = "    def save_session(self, app, session, response):\n"
    lines[302] = "        serializer = self.get_signing_serializer(app)\n"
    lines[303] = "        value = serializer.dumps(dict(session))\n"
    lines[330] = "        response.set_cookie('session', value)\n"

    source = _source_file("src/flask/sessions.py", "".join(lines))
    chunks = chunk_repository_files([source], chunk_lines=80, overlap_lines=15, max_chunks=20)
    index = SQLiteFTSIndex(str(tmp_path / "session-definitions.sqlite3"))
    index.replace_index(_revision(), chunks, files_indexed=1, total_bytes=8_000)

    retrieved = index.retrieve(
        "o",
        "r",
        "c" * 40,
        "Explain sessions.py open_session and save_session implementation",
        file_references=("sessions.py",),
        limit=4,
    )

    first = retrieved[0]
    assert first.start_line == 261
    assert "def open_session(" in first.content
    assert "def save_session(" in first.content
    assert first.chunk_id == chunks[4].chunk_id
    assert any(
        chunk.start_line == 326 and "response.set_cookie('session', value)" in chunk.content
        for chunk in retrieved
    )
