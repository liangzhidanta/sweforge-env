"""Mac server optional Bearer auth: token 开/关的客户端行为。

vendored make_app 不关心认证; Mac 侧 env_server.server.bearer_auth_app 在
外层加校验。AutoDL 客户端（vendored, 无 token）在开认证时应收 401; Mac 侧
参考实现 AuthenticatedRemoteEnvironmentBackend 全程可用。
"""

from __future__ import annotations

import threading
import time

import pytest

from sweforge.env_server.client import AuthenticatedRemoteEnvironmentBackend
from sweforge.env_server.docker.backend import LocalDockerBackend
from sweforge.env_server.server import bearer_auth_app
from sweforge.environment.remote import RemoteEnvironmentBackend
from sweforge.environment.server import make_app
from sweforge.protocol.tools import BashAction, StrReplaceAction
from sweforge.schemas.task import TaskEnvironment, TaskSpec

TOKEN = "test-secret-token"

EXAMPLES = __import__("pathlib").Path(__file__).resolve().parents[1] / "examples"


def _task(task_id: str = "auth-demo") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        repo="demo",
        base_commit="0000000",
        problem_statement="answer.txt 里有一个多余的 'wrong ' 前缀, 请修复。",
        environment=TaskEnvironment(
            setup_commands=['echo "the wrong answer" > answer.txt'],
            test_commands={"test_answer": ["grep", "-q", "^the answer$", "answer.txt"]},
        ),
        fail_to_pass=[{"test_id": "test_answer", "kind": "fail_to_pass"}],
        pass_to_pass=[],
    )


@pytest.fixture(scope="module")
def auth_server():
    import uvicorn

    backend = LocalDockerBackend(bundles_dir=EXAMPLES)
    app = bearer_auth_app(make_app(backend), TOKEN)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.05)
    assert srv.started, "server failed to start"
    port = srv.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=5)


def test_health_open_without_token(auth_server):
    client = RemoteEnvironmentBackend(base_url=auth_server, max_retries=1)
    try:
        assert client.health()  # /health 不校验 token
    finally:
        client.close()


def test_401_without_token(auth_server):
    client = RemoteEnvironmentBackend(base_url=auth_server, max_retries=1)
    try:
        with pytest.raises(RuntimeError, match="401"):
            client.create(_task(task_id="no-token"))
    finally:
        client.close()


def test_authenticated_full_flow(auth_server):
    client = AuthenticatedRemoteEnvironmentBackend(base_url=auth_server, token=TOKEN, max_retries=1)
    try:
        env_id = client.create(_task(task_id="with-token"))
        assert env_id == "with-token"
        obs = client.execute(env_id, BashAction(command="cat answer.txt"))
        assert obs.stdout.strip() == "the wrong answer"
        obs = client.execute(env_id, StrReplaceAction(path="answer.txt", old_string="wrong ", new_string=""))
        assert obs.success
        patch = client.export_patch(env_id)
        assert "a/answer.txt" in patch
        client.destroy(env_id)
        v = client.verify(_task(task_id="with-token"), patch)
        assert v.verdict == "resolved" and v.reward == 1.0
    finally:
        client.close()
