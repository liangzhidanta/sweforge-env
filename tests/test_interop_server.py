"""阶段 8 联调 E2E（Mac 侧）: AutoDL 契约客户端 -> LocalDockerBackend server。

与 AutoDL 侧 scripts/interop_mac.py 同流程（两段: 契约直连 + AgentLoop），
只是把 make_app 的 backend 换成 LocalDockerBackend 跑在随机端口上。证明:
  * 无 bundle 的 setup 驱动任务（interop-demo）经 server 契约可用
  * 五工具 + export_patch + verify + destroy 全程 HTTP 往返
  * vendored AgentLoop 经 RemoteEnvironmentBackend 完整闭环并通过 validate_trajectory
"""

from __future__ import annotations

import threading
import time

import pytest

from sweforge.agent.loop import AgentLoop, AssistantDecision
from sweforge.env_server.docker.backend import LocalDockerBackend
from sweforge.environment.remote import RemoteEnvironmentBackend
from sweforge.environment.server import make_app
from sweforge.protocol.tools import (
    BashAction,
    FinishAction,
    SearchAction,
    StrReplaceAction,
    ViewFileAction,
)
from sweforge.schemas.task import TaskEnvironment, TaskSpec

EXAMPLES = __import__("pathlib").Path(__file__).resolve().parents[1] / "examples"


def _interop_task(task_id: str = "interop-demo") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        repo="demo",
        base_commit="0000000",
        problem_statement="answer.txt 里有一个多余的 'wrong ' 前缀, 请修复。",
        environment=TaskEnvironment(
            setup_commands=['echo "the wrong answer" > answer.txt'],
            test_commands={
                "test_answer": ["grep", "-q", "^the answer$", "answer.txt"],
                "test_sanity": ["true"],
            },
        ),
        fail_to_pass=[{"test_id": "test_answer", "kind": "fail_to_pass"}],
        pass_to_pass=[{"test_id": "test_sanity", "kind": "pass_to_pass"}],
    )


@pytest.fixture(scope="module")
def mac_server():
    """随机端口上的 uvicorn 线程, 包 LocalDockerBackend（无 bundle 任务）。"""
    import uvicorn

    backend = LocalDockerBackend(bundles_dir=EXAMPLES)
    config = uvicorn.Config(make_app(backend), host="127.0.0.1", port=0, log_level="warning")
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


class InteropPolicy:
    def __init__(self, steps: list):
        self.steps = steps

    def decide(self, messages):
        if not self.steps:
            return None
        return AssistantDecision(content=f"step {len(self.steps)}", action=self.steps.pop(0))


def test_segment_a_contract_direct(mac_server):
    task = _interop_task()
    client = RemoteEnvironmentBackend(base_url=mac_server, max_retries=1)
    try:
        assert client.health()
        assert client.register_task(task) == "interop-demo"
        env_id = client.create(task_id="interop-demo")
        assert env_id.startswith("interop-demo-")

        obs = client.execute(env_id, BashAction(command="cat answer.txt"))
        assert obs.stdout.strip() == "the wrong answer"

        obs = client.execute(env_id, ViewFileAction(path="answer.txt", start_line=1, end_line=3))
        assert obs.total_lines == 1

        obs = client.execute(env_id, SearchAction(query="wrong", path="answer.txt"))
        assert any("answer.txt" == m.path for m in obs.matches)

        obs = client.execute(env_id, StrReplaceAction(path="answer.txt", old_string="wrong ", new_string=""))
        assert obs.success

        obs = client.execute(env_id, FinishAction())
        patch = obs.patch
        assert "a/answer.txt" in patch
        client.destroy(env_id)

        v = client.verify(task, patch)
        assert v.verdict == "resolved" and v.resolved
        assert v.f2p_passed == 1 and v.p2p_passed == 1
    finally:
        client.close()


def test_segment_b_agent_loop(mac_server):
    task = _interop_task(task_id="interop-loop")
    client = RemoteEnvironmentBackend(base_url=mac_server, max_retries=1)
    try:
        steps = [
            ViewFileAction(path="answer.txt", start_line=1, end_line=3),
            StrReplaceAction(path="answer.txt", old_string="wrong ", new_string=""),
            FinishAction(),
        ]
        traj = AgentLoop(client, InteropPolicy(steps)).run(task)
        assert traj.termination_reason.value == "agent_finished"
        assert traj.num_turns >= 3
        assert "answer.txt" in traj.patch
    finally:
        client.close()
