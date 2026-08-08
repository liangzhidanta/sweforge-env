"""阶段 6: RemoteEnvironmentBackend 契约测试。

fake server（conftest.remote_server）= thin HTTP 层（fastapi）包
MockEnvironmentBackend，与客户端共同锁定 §8 契约 v1 的请求/响应 JSON 格式；
阶段 8 的 Mac Environment Server 照此实现。真实执行走 Mock 语义（无容器
隔离），本测试只验证传输契约。
"""

from __future__ import annotations

import socket

import pytest

from sweforge.environment.remote import RemoteEnvironmentBackend
from sweforge.protocol.tools import (
    BashAction,
    FinishAction,
    SearchAction,
    StrReplaceAction,
    ViewFileAction,
)
from sweforge.schemas.task import TaskEnvironment, TaskSpec
from tests.conftest import GOOD_PATCH, SEED


def _task(task_id: str = "t1") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        repo="demo",
        base_commit="0000000",
        problem_statement="fix the wrong answer",
        environment=TaskEnvironment(
            setup_commands=["echo setup-ran > setup.txt"],
            test_commands={
                "test_answer": ["grep", "-q", "the answer", "answer.txt"],
                "test_sanity": ["true"],
            },
        ),
        fail_to_pass=[{"test_id": "test_answer", "kind": "fail_to_pass"}],
        pass_to_pass=[{"test_id": "test_sanity", "kind": "pass_to_pass"}],
    )


def _client(remote_server) -> RemoteEnvironmentBackend:
    url, _ = remote_server
    return RemoteEnvironmentBackend(base_url=url, max_retries=1)


# ---------------- 生命周期 ----------------

def test_remote_health(remote_server):
    assert _client(remote_server).health()


def test_remote_create_reset_destroy(remote_server):
    client = _client(remote_server)
    env_id = client.create(_task(task_id="lifecycle"))
    client.reset(env_id)
    client.destroy(env_id)


def test_remote_404_propagates(remote_server):
    client = _client(remote_server)
    with pytest.raises(RuntimeError, match="404"):
        client.reset("no-such-env")


def test_remote_connection_refused():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    client = RemoteEnvironmentBackend(base_url=f"http://127.0.0.1:{free_port}")
    with pytest.raises(RuntimeError, match="unreachable"):
        client.create(_task(task_id="down"))


# ---------------- 五工具往返（observation 序列化 round-trip） ----------------

def test_remote_execute_tools_roundtrip(remote_server):
    client = _client(remote_server)
    env_id = client.create(_task(task_id="tools"))

    obs = client.execute(env_id, BashAction(command="echo hello"))
    assert obs.exit_code == 0 and obs.stdout.strip() == "hello"

    obs = client.execute(env_id, StrReplaceAction(path="answer.txt", old_string="wrong ", new_string=""))
    assert obs.success

    obs = client.execute(env_id, ViewFileAction(path="src/main.py", start_line=1, end_line=2))
    assert obs.total_lines == 2
    assert obs.content.splitlines()[0] == "     1\tdef add(a, b):"

    obs = client.execute(env_id, SearchAction(query="return", path="src/main.py"))
    assert [(m.path, m.line) for m in obs.matches] == [("src/main.py", 2)]

    obs = client.execute(env_id, FinishAction())
    assert obs.patch and "a/answer.txt" in obs.patch
    client.destroy(env_id)


def test_remote_export_patch(remote_server):
    client = _client(remote_server)
    env_id = client.create(_task(task_id="patch"))
    assert client.export_patch(env_id) == ""
    client.execute(env_id, StrReplaceAction(path="answer.txt", old_string="wrong ", new_string=""))
    patch = client.export_patch(env_id)
    assert "a/answer.txt" in patch and "+the answer" in patch
    client.destroy(env_id)


# ---------------- verify ----------------

def test_remote_verify_resolved(remote_server):
    client = _client(remote_server)
    v = client.verify(_task(task_id="v-good"), GOOD_PATCH)
    assert v.verdict == "resolved" and v.resolved
    assert v.fail_to_pass[0].test_id == "test_answer"
    assert all(r.passed for r in v.pass_to_pass)


def test_remote_verify_unresolved(remote_server):
    client = _client(remote_server)
    v = client.verify(_task(task_id="v-bad"), "")
    assert v.verdict == "unresolved" and not v.resolved


# ---------------- 幂等 request_id ----------------

def test_remote_request_ids_on_mutations(remote_server):
    url, request_ids = remote_server
    request_ids.clear()
    client = _client(remote_server)

    env_id = client.create(_task(task_id="rid"))
    client.reset(env_id)
    client.execute(env_id, BashAction(command="echo hi"))
    client.export_patch(env_id)  # GET: 不携带 request_id
    client.destroy(env_id)
    client.verify(_task(task_id="rid-verify"), GOOD_PATCH)

    assert len(request_ids) == 5  # create/reset/actions/destroy/verify（GET 不带）
    assert all(len(rid) >= 16 for rid in request_ids)  # uuid hex 非空
