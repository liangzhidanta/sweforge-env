"""阶段 8: Environment Server 行为测试（§8 契约 v1 正式实现）。

remote_server fixture 跑的就是 environment/server.py 的 make_app —— 与
Mac toy Environment Server 是同一份代码。本文件覆盖 fake server 时代没有
的新行为:
    - /v1/tasks/register 登记 + create(task_id 引用)
    - request_id 幂等表（重放返回首次响应, 不重复执行）
    - 404 / 422 语义

幂等断言用 httpx 直接发原始请求（客户端每次自动生成新 request_id,
无法重放同一 id）。
"""

from __future__ import annotations

import httpx

from sweforge.environment.remote import RemoteEnvironmentBackend
from sweforge.protocol.tools import BashAction, StrReplaceAction
from sweforge.schemas.task import TaskEnvironment, TaskSpec
from tests.conftest import GOOD_PATCH


def _task(task_id: str = "srv") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        repo="demo",
        base_commit="0000000",
        problem_statement="fix the wrong answer",
        environment=TaskEnvironment(
            setup_commands=["true"],
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
    return RemoteEnvironmentBackend(base_url=url)


# ---------------- /v1/tasks/register + create(task_id) ----------------

def test_register_then_create_by_task_id(remote_server):
    """登记后 create 只传 task_id 引用, env_id 与 task_id 合一。"""
    client = _client(remote_server)
    assert client.register_task(_task(task_id="registered")) == "registered"

    env_id = client.create(task_id="registered")
    assert env_id == "registered"
    obs = client.execute(env_id, BashAction(command="echo ok"))
    assert obs.exit_code == 0
    client.destroy(env_id)


def test_create_unregistered_task_id_404(remote_server):
    client = _client(remote_server)
    with __import__("pytest").raises(RuntimeError, match="404"):
        client.create(task_id="never-registered")


def test_create_without_task_or_id_422(remote_server):
    url, _ = remote_server
    with httpx.Client(base_url=url) as c:
        resp = c.post("/v1/envs", json={"request_id": "x" * 16})
    assert resp.status_code == 422


# ---------------- request_id 幂等表 ----------------

def test_idempotent_create_replay(remote_server):
    """同 request_id 重发 create: 返回首次响应, 不覆盖已有 env。"""
    url, _ = remote_server
    rid = "a" * 16
    body = {"task": _task(task_id="idem-create").model_dump(mode="json"), "request_id": rid}
    with httpx.Client(base_url=url) as c:
        r1 = c.post("/v1/envs", json=body)
        r2 = c.post("/v1/envs", json=body)
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json() == {"env_id": "idem-create"}
    # 首次 env 未被覆盖: 仍可正常执行
    with httpx.Client(base_url=url) as c:
        assert c.post(
            "/v1/envs/idem-create/reset", json={"request_id": "a" * 16 + "1"}
        ).status_code == 200


def test_idempotent_execute_replays_once(remote_server):
    """同 request_id 重发 execute: 返回首次结果, 不二次执行。

    证明手段: str_replace 第一次成功; 若重放真的再执行一次, old_string
    已不存在会返回失败 observation。重放必须与首次响应逐字一致。
    """
    url, _ = remote_server
    with httpx.Client(base_url=url) as c:
        c.post("/v1/envs", json={"task": _task(task_id="idem-act").model_dump(mode="json")})
        rid = "b" * 16
        body = {
            "action": StrReplaceAction(path="answer.txt", old_string="wrong ", new_string="").model_dump(mode="json"),
            "request_id": rid,
        }
        r1 = c.post("/v1/envs/idem-act/actions", json=body)
        r2 = c.post("/v1/envs/idem-act/actions", json=body)
        assert r1.json() == r2.json()
        assert r1.json()["observation"]["success"] is True

        # 新 request_id 再执行同一条 action: 应失败（old_string 已不存在）
        body["request_id"] = "b" * 16 + "1"
        r3 = c.post("/v1/envs/idem-act/actions", json=body)
        assert r3.json()["observation"]["success"] is False
        c.delete("/v1/envs/idem-act", params={"request_id": "b" * 16 + "2"})


def test_idempotent_destroy_replay(remote_server):
    """同 request_id 重发 destroy: 返回首次响应; 新 request_id 才 404。"""
    url, _ = remote_server
    with httpx.Client(base_url=url) as c:
        c.post("/v1/envs", json={"task": _task(task_id="idem-del").model_dump(mode="json")})
        rid = "c" * 16
        r1 = c.delete("/v1/envs/idem-del", params={"request_id": rid})
        r2 = c.delete("/v1/envs/idem-del", params={"request_id": rid})
        assert r1.status_code == r2.status_code == 200
        assert r1.json() == r2.json() == {"ok": True}
        # 新 request_id: env 已被真正销毁
        r3 = c.delete("/v1/envs/idem-del", params={"request_id": "c" * 16 + "1"})
        assert r3.status_code == 404


def test_idempotent_verify_replay(remote_server):
    """同 request_id 重发 verify: 返回首次验证结果。"""
    url, _ = remote_server
    rid = "d" * 16
    body = {
        "task": _task(task_id="idem-ver").model_dump(mode="json"),
        "patch": GOOD_PATCH,
        "request_id": rid,
    }
    with httpx.Client(base_url=url) as c:
        r1 = c.post("/v1/verifications", json=body)
        r2 = c.post("/v1/verifications", json=body)
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()
    assert r1.json()["verification"]["verdict"] == "resolved"
