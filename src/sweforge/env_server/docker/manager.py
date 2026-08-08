"""Docker container labels, roles, resource limits, and stale-container cleanup."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from datetime import datetime

__all__ = [
    "ContainerLimits",
    "ROLE_TASK",
    "ROLE_VERIFIER",
    "LABEL_MANAGED",
    "LABEL_ROLE",
    "LABEL_TASK_ID",
    "LABEL_ENV_ID",
    "container_labels",
    "list_managed_containers",
    "cleanup_stale_containers",
]


@dataclass(frozen=True)
class ContainerLimits:
    cpus: float = 2.0
    memory: str = "4g"
    pids: int = 128
    timeout_seconds: float = 120.0


ROLE_TASK = "task"
ROLE_VERIFIER = "verifier"

LABEL_MANAGED = "sweforge.managed"
LABEL_ROLE = "sweforge.role"
LABEL_TASK_ID = "sweforge.task_id"
LABEL_ENV_ID = "sweforge.env_id"


def container_labels(task_id: str, env_id: str, role: str = ROLE_TASK) -> tuple[str, ...]:
    return (
        f"{LABEL_MANAGED}=true",
        f"{LABEL_ROLE}={role}",
        f"{LABEL_TASK_ID}={task_id}",
        f"{LABEL_ENV_ID}={env_id}",
    )


def _managed_filter_args(
    *,
    role: str | None = None,
    task_id: str | None = None,
    env_id: str | None = None,
) -> list[str]:
    """docker ps --filter args selecting sweforge-managed containers."""
    filters = [f"label={LABEL_MANAGED}=true"]
    if role:
        filters.append(f"label={LABEL_ROLE}={role}")
    if task_id:
        filters.append(f"label={LABEL_TASK_ID}={task_id}")
    if env_id:
        filters.append(f"label={LABEL_ENV_ID}={env_id}")
    return [f"--filter={f}" for f in filters]


def _created_epochs(
    docker_binary: str, containers: list[tuple[str, str]]
) -> dict[str, float]:
    """Map container_id -> created epoch (UTC), batch docker inspect.

    daemon 的 `until` 过滤在各版本不可靠（Docker 29 直接拒绝），改为查
    {{.Created}}（RFC3339Nano UTC）在 Python 里按年龄过滤。
    """
    if not containers:
        return {}
    ids = [cid for cid, _ in containers]
    completed = subprocess.run(
        (docker_binary, "inspect", "--format", "{{.Id}} {{.Created}}", *ids),
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"docker inspect failed: {completed.stderr.strip()}")
    result: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        cid, _, created = line.partition(" ")
        try:
            result[cid.strip()] = datetime.fromisoformat(
                created.strip().replace("Z", "+00:00")
            ).timestamp()
        except ValueError:
            continue
    return result


def list_managed_containers(
    docker_binary: str = "docker",
    *,
    role: str | None = None,
    task_id: str | None = None,
    env_id: str | None = None,
    age_seconds: int | None = None,
) -> list[tuple[str, str]]:
    """List sweforge-managed containers as (container_id, name), oldest first.

    role/task_id/env_id 任选其一收窄范围；age_seconds 只保留创建时间早于
    该秒数的容器（"重启时清上一轮泄漏"且不误杀刚创建的活跃容器）。
    """
    completed = subprocess.run(
        (
            docker_binary, "ps", "-a", "--no-trunc",
            *_managed_filter_args(role=role, task_id=task_id, env_id=env_id),
            "--format", "{{.ID}} {{.Names}}",
        ),
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"docker ps failed: {completed.stderr.strip()}")
    containers: list[tuple[str, str]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        cid, _, name = line.partition(" ")
        containers.append((cid.strip(), name.strip()))
    if age_seconds is None:
        return containers
    cutoff = time.time() - age_seconds
    created = _created_epochs(docker_binary, containers)
    return [(cid, name) for cid, name in containers if created.get(cid, float("inf")) < cutoff]


def cleanup_stale_containers(
    docker_binary: str = "docker",
    *,
    role: str | None = None,
    task_id: str | None = None,
    env_id: str | None = None,
    age_seconds: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Remove sweforge-managed containers matching the filters; return removed names.

    默认（无过滤/无 age）会删掉 daemon 上**所有** sweforge-managed 容器，
    调用方应明确收窄（env_id / role / age_seconds）。dry_run=True 只列出不删。
    """
    containers = list_managed_containers(
        docker_binary, role=role, task_id=task_id, env_id=env_id, age_seconds=age_seconds
    )
    removed: list[str] = []
    for cid, name in containers:
        if dry_run:
            removed.append(f"{name} (dry-run)")
            continue
        completed = subprocess.run(
            (docker_binary, "rm", "-f", cid), capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(f"docker rm {cid} failed: {completed.stderr.strip()}")
        removed.append(name)
    return removed
