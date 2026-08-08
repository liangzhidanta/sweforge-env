"""阶段 8/9 集成: 完整 rollout loop -> patch -> clean verify -> validate_trajectory。

用 vendored AgentLoop（AutoDL 与 Mac 同一份代码）驱动 LocalDockerBackend:
Local 路径（本地临时目录, CI/开发）与 Docker 路径（真实容器, 需 daemon+镜像）。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from sweforge.agent.loop import AgentLoop, AssistantDecision
from sweforge.env_server.docker.backend import LocalDockerBackend, load_task_bundle
from sweforge.protocol.messages import CanonicalMessage
from sweforge.protocol.tools import FinishAction, StrReplaceAction, ViewFileAction
from sweforge.protocol.validate import validate_trajectory
from sweforge.schemas.trajectory import TerminationReason

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "toy_cache_aliasing"

OLD = (
    "    value = compute()\n"
    "    if value not in _CACHE:\n"
    "        _CACHE[value] = value\n"
    "    return _CACHE[value]\n"
)
NEW = (
    "    if key not in _CACHE:\n"
    "        _CACHE[key] = compute()\n"
    "    return _CACHE[key]\n"
)


class ScriptedPolicy:
    """确定性脚本 policy: 依次执行给定 action 序列（冒烟/测试用）。"""

    def __init__(self, steps: list):
        self.steps = steps

    def decide(self, messages: list[CanonicalMessage]):
        if not self.steps:
            return None
        return AssistantDecision(content="step", action=self.steps.pop(0))


def _fix_policy():
    return ScriptedPolicy(
        [
            ViewFileAction(path="toy_cache/cache.py"),
            StrReplaceAction(path="toy_cache/cache.py", old_string=OLD, new_string=NEW),
            FinishAction(),
        ]
    )


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        daemon = subprocess.run(("docker", "info"), capture_output=True).returncode == 0
    except OSError:
        return False
    if not daemon:
        return False
    image = subprocess.run(("docker", "image", "inspect", "sweforge-base"), capture_output=True)
    return image.returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_ready(), reason="docker daemon or sweforge-base image not available"
)


def test_full_rollout_loop_and_clean_verify_local():
    task = load_task_bundle(EXAMPLES).task
    backend = LocalDockerBackend(EXAMPLES.parent)
    trajectory = AgentLoop(backend, _fix_policy()).run(task)

    assert trajectory.termination_reason == TerminationReason.AGENT_FINISHED
    assert trajectory.num_turns == 3
    v = validate_trajectory(trajectory)
    assert v.ok, v.errors

    # 修复 patch 在 clean 环境验证为 resolved
    result = backend.verify(task, trajectory.patch)
    assert result.integrity_ok
    assert result.verdict == "resolved"
    assert result.reward == 1.0
    assert "-    value = compute()" in trajectory.patch


@requires_docker
def test_full_rollout_loop_and_clean_verify_docker():
    task = load_task_bundle(EXAMPLES).task
    backend = LocalDockerBackend(EXAMPLES.parent, use_docker=True)
    trajectory = AgentLoop(backend, _fix_policy()).run(task)

    assert trajectory.termination_reason == TerminationReason.AGENT_FINISHED
    v = validate_trajectory(trajectory)
    assert v.ok, v.errors

    result = backend.verify(task, trajectory.patch)
    assert result.verdict == "resolved"
    assert result.reward == 1.0
    assert result.metadata["docker"] is True


@requires_docker
def test_docker_seed_has_no_appledouble_entries():
    """回归: macOS bsdtar 会合成 AppleDouble ._* 条目, 容器内 GNU tar 会
    物化这些文件污染 workspace。seed 必须走 Python tarfile, 产物零 ._*。"""
    task = load_task_bundle(EXAMPLES).task
    backend = LocalDockerBackend(EXAMPLES.parent, use_docker=True)
    bundle = backend._bundle(task)
    executor = backend._make_executor(bundle, "verify")
    try:
        files = executor.list_files()
        assert files, "seeded workspace should not be empty"
        polluted = [rel for rel in files if Path(rel).name.startswith("._") or "AppleDouble" in rel]
        assert polluted == [], f"AppleDouble pollution in seeded workspace: {polluted}"
        assert "toy_cache/cache.py" in files
    finally:
        executor.close()


def test_unfixed_patch_is_not_resolved():
    task = load_task_bundle(EXAMPLES).task
    backend = LocalDockerBackend(EXAMPLES.parent)
    result = backend.verify(task, "")
    assert result.verdict == "unresolved"
    assert result.f2p_passed == 0
    assert not result.resolved
