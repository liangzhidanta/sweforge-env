"""Docker container labels, roles, and resource limits (M1 constants)."""

from __future__ import annotations

from dataclasses import dataclass


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
