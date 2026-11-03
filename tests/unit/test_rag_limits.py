import base64

import httpx
import pytest

from repo_agent.github.service import GitHubRepositoryService
from repo_agent.settings import Settings


@pytest.mark.asyncio
async def test_rag_fetch_enforces_content_request_file_and_total_byte_limits() -> None:
    requests: list[str] = []
    files = {
        "src/a.py": b"1234",
        "src/b.py": b"12345",
        "src/c.py": b"abcd",
    }
    commit = "c" * 40

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        requests.append(path)
        if path == "/repos/o/r":
            return httpx.Response(
                200,
                json={"full_name": "o/r", "default_branch": "main"},
                request=request,
            )
        if path.endswith("/branches/main"):
            return httpx.Response(200, json={"commit": {"sha": commit}}, request=request)
        if "/git/trees/" in path:
            return httpx.Response(
                200,
                json={
                    "tree": [
                        {
                            "path": file_path,
                            "type": "blob",
                            "sha": f"{index + 1:040x}",
                            "size": len(content),
                        }
                        for index, (file_path, content) in enumerate(files.items())
                    ],
                    "truncated": False,
                },
                request=request,
            )
        file_path = path.split("/contents/", 1)[1]
        return httpx.Response(
            200,
            json={
                "type": "file",
                "encoding": "base64",
                "content": base64.b64encode(files[file_path]).decode(),
            },
            request=request,
        )

    service = GitHubRepositoryService(
        Settings(_env_file=None), transport=httpx.MockTransport(handler)
    )
    snapshot = await service.inspect(
        "https://github.com/o/r",
        max_files=10,
        max_file_bytes=4,
        max_total_bytes=6,
        max_requests=4,
    )

    assert len(requests) == 4  # metadata, branch, tree, and at most one content read
    assert [source_file.path for source_file in snapshot.files] == ["src/a.py"]
    assert snapshot.repository.source_bytes <= 6
    assert all(source_file.size_bytes <= 4 for source_file in snapshot.files)
