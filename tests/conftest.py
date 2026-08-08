"""共享 fixture: §8 契约 server（正式实现 environment/server.py 包 Mock 后端,
随机端口上的 session 级 uvicorn 线程）。

remote_server 与客户端（RemoteEnvironmentBackend）共同锁定请求/响应 JSON
格式 —— 阶段 8 的 Mac Environment Server 就是同一个 make_app 跑在
127.0.0.1:8500（CLI: python -m sweforge.environment.server）。
"""

from __future__ import annotations

import tempfile
import threading
import time

import pytest

from sweforge.environment.mock import MockEnvironmentBackend
from sweforge.environment.server import make_app

SEED = {
    "answer.txt": "the wrong answer\n",
    "src/main.py": "def add(a, b):\n    return a + b  # FIXME\n",
}

GOOD_PATCH = "--- a/answer.txt\n+++ b/answer.txt\n@@ -1 +1 @@\n-the wrong answer\n+the answer\n"


@pytest.fixture(scope="session")
def remote_server():
    """随机端口上的 uvicorn 线程; 返回 (base_url, 收到的 request_id 列表)。"""
    import uvicorn

    request_ids: list[str] = []
    backend = MockEnvironmentBackend(
        workspace_root=tempfile.mkdtemp(prefix="sweforge-remote-seed-"), seed_files=SEED
    )
    config = uvicorn.Config(make_app(backend, request_ids), host="127.0.0.1", port=0, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.05)
    assert srv.started, "server failed to start"
    port = srv.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", request_ids
    srv.should_exit = True
    thread.join(timeout=5)
