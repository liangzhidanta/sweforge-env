"""Docker manager: managed-container listing + stale cleanup.

Pure filter-argv tests run anywhere; the real-container test needs the docker
daemon + sweforge-base image (same gate as test_integration).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from sweforge.env_server.docker.backend import LocalDockerBackend, load_task_bundle
from sweforge.env_server.docker import manager
from sweforge.env_server.docker.manager import (
    LABEL_ENV_ID,
    LABEL_MANAGED,
    LABEL_ROLE,
    LABEL_TASK_ID,
    cleanup_stale_containers,
    list_managed_containers,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "toy_cache_aliasing"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        if subprocess.run(("docker", "info"), capture_output=True).returncode != 0:
            return False
    except OSError:
        return False
    return subprocess.run(("docker", "image", "inspect", "sweforge-base"), capture_output=True).returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_ready(), reason="docker daemon or sweforge-base image not available"
)


# ---------------- 纯函数: 过滤参数构造 ----------------

def test_managed_filter_args_default():
    assert manager._managed_filter_args() == [f"--filter=label={LABEL_MANAGED}=true"]


def test_managed_filter_args_scoped():
    args = manager._managed_filter_args(role="task", task_id="t1", env_id="e1")
    assert f"--filter=label={LABEL_MANAGED}=true" in args
    assert f"--filter=label={LABEL_ROLE}=task" in args
    assert f"--filter=label={LABEL_TASK_ID}=t1" in args
    assert f"--filter=label={LABEL_ENV_ID}=e1" in args


# ---------------- 真实容器: 列出 / dry-run / 清理 ----------------

@requires_docker
def test_list_and_cleanup_managed_container():
    backend = LocalDockerBackend(EXAMPLES.parent, use_docker=True)
    env = backend.create(load_task_bundle(EXAMPLES).task)
    try:
        docker = env.executor.docker_binary

        listed = list_managed_containers(docker, env_id=env.env_id)
        assert any(name == env.executor.container_name for _, name in listed)

        # age_seconds=1: 刚创建的容器不算 stale
        young = list_managed_containers(docker, env_id=env.env_id, age_seconds=1)
        assert all(name != env.executor.container_name for _, name in young)
        # age_seconds=0: cutoff=now, 创建于过去的容器命中
        old = list_managed_containers(docker, env_id=env.env_id, age_seconds=0)
        assert any(name == env.executor.container_name for _, name in old)

        dry = cleanup_stale_containers(docker, env_id=env.env_id, dry_run=True)
        assert dry and all("dry-run" in line for line in dry)
        assert any(env.executor.container_name in line for line in dry)

        still = list_managed_containers(docker, env_id=env.env_id)
        assert any(name == env.executor.container_name for _, name in still)

        removed = cleanup_stale_containers(docker, env_id=env.env_id)
        assert env.executor.container_name in removed
        assert not list_managed_containers(docker, env_id=env.env_id)
    finally:
        env.executor.close()  # 容器可能已被清理; close 幂等忽略失败
