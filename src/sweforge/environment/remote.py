"""阶段 6 RemoteEnvironmentBackend —— AutoDL 侧客户端（PROJECT_SPEC §8 契约 v1）。

    GET    /health
    POST   /v1/tasks/register           # Mac 侧登记 TaskSpec（server 持有）
    POST   /v1/envs                     # create: 内联 {"task"} 或引用 {"task_id"}
    POST   /v1/envs/{env_id}/reset
    POST   /v1/envs/{env_id}/actions    # execute(action) -> observation
    GET    /v1/envs/{env_id}/patch      # export_patch
    DELETE /v1/envs/{env_id}
    POST   /v1/verifications            # clean-container verify(task, patch)

请求/响应 JSON 格式（本客户端与 tests/ 的 §8 参考 server 共同锁定，
阶段 8 Mac Environment Server 即该 server 实现）:
    register  -> {"task_id": str}
    create    -> {"env_id": str}
    reset     -> {"ok": true}
    actions   -> {"observation": {...ToolObservation dict...}}
    patch     -> {"patch": str}
    destroy   -> {"ok": true}
    verify    -> {"verification": {...VerificationResult dict...}}
    错误      -> HTTP 4xx/5xx + {"error": str}

幂等: 所有修改状态的请求携带 request_id（uuid4）；传输失败/5xx 重试时复用
同一 request_id（server 侧幂等表据此去重）。4xx 是业务错误，不重试。

拓扑（§8）: AutoDL Agent -> RemoteEnvironmentBackend -> HTTP -> SSH tunnel
-> Mac Environment Server（只监听 127.0.0.1）-> Docker。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx

from sweforge.environment.base import EnvironmentBackend
from sweforge.protocol.tools import ToolAction, ToolObservation, parse_tool_observation
from sweforge.schemas.task import TaskSpec
from sweforge.schemas.verification import VerificationResult

__all__ = ["RemoteEnvironmentBackend"]

#: 默认连接目标（SSH tunnel 的本地端口，Mac 侧 Environment Server 的约定端口）
DEFAULT_BASE_URL = "http://127.0.0.1:8500"


class RemoteEnvironmentBackend(EnvironmentBackend):
    """§8 契约 v1 的 HTTP 客户端。env 句柄 = server 侧 env_id（str）。"""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self.max_retries = max_retries

    def close(self) -> None:
        self._client.close()

    # ------------------------- 接口实现 -------------------------

    def register_task(self, task: TaskSpec) -> str:
        """POST /v1/tasks/register: Mac 侧登记 TaskSpec（server 持有）。

        登记后 create(task_id=...) 可只传引用（rollout 批量复用同一任务时
        省带宽, 且 server 持有同一份 TaskSpec, 避免客户端/服务端漂移）。
        """
        data = self._request(
            "POST", "/v1/tasks/register", body={"task": task.model_dump(mode="json")}
        )
        return data["task_id"]

    def create(self, task: TaskSpec | None = None, *, task_id: str | None = None) -> str:
        """创建环境。二选一: 内联 task 或引用已登记的 task_id（§8）。

        env_id = task_id（server 侧注册表 key 与任务 id 合一）。
        """
        if task is not None:
            body: dict[str, Any] = {"task": task.model_dump(mode="json")}
        elif task_id is not None:
            body = {"task_id": task_id}
        else:
            raise ValueError("create() needs task or task_id")
        data = self._request("POST", "/v1/envs", body=body)
        return data["env_id"]

    def reset(self, env: str) -> None:
        self._request("POST", f"/v1/envs/{env}/reset", body={})

    def execute(self, env: str, action: ToolAction) -> ToolObservation:
        data = self._request(
            "POST", f"/v1/envs/{env}/actions", body={"action": action.model_dump(mode="json")}
        )
        return parse_tool_observation(data["observation"])

    def export_patch(self, env: str) -> str:
        data = self._request("GET", f"/v1/envs/{env}/patch", body=None)
        return data["patch"]

    def verify(self, task: TaskSpec, patch: str) -> VerificationResult:
        data = self._request(
            "POST",
            "/v1/verifications",
            body={"task": task.model_dump(mode="json"), "patch": patch},
        )
        return VerificationResult.model_validate(data["verification"])

    def destroy(self, env: str) -> None:
        self._request("DELETE", f"/v1/envs/{env}")

    # ------------------------- 辅助 -------------------------

    def health(self) -> bool:
        """server 可达性检查（GET /health -> 200）。"""
        try:
            resp = self._client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except httpx.TransportError:
            return False

    # ------------------------- 请求管线 -------------------------

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """发送请求；修改状态请求携带 request_id，重试复用（幂等）。

        DELETE 的 request_id 放 query（DELETE 带 body 不标准）。
        """
        mutating = method in ("POST", "PUT", "PATCH", "DELETE")
        request_id = uuid.uuid4().hex if mutating else None
        payload: dict[str, Any] | None = None
        if body is not None:
            payload = dict(body)
            if request_id is not None:
                payload["request_id"] = request_id
        query = f"?request_id={request_id}" if request_id and method == "DELETE" else ""

        content = json.dumps(payload) if payload is not None else None
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = self._client.request(
                    method,
                    f"{self.base_url}{path}{query}",
                    content=content,
                    headers={"Content-Type": "application/json"},
                )
            except httpx.TransportError as e:  # 连接/读超时等: 可重试
                last_error = e
                if attempt < attempts - 1:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"remote env unreachable at {self.base_url}: {e}"
                ) from e
            if resp.status_code >= 500 and attempt < attempts - 1:
                time.sleep(0.2 * (attempt + 1))
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"remote env {method} {path} failed ({resp.status_code}): "
                    f"{_error_detail(resp)}"
                )
            return resp.json() if resp.content else {}
        raise RuntimeError(f"remote env {method} {path} failed after retries: {last_error}")


def _error_detail(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    except ValueError:
        pass
    return resp.text[:500]
