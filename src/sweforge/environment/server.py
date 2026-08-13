"""阶段 8 Mac Environment Server —— §8 契约 v1 参考实现（只监听 127.0.0.1）。

thin HTTP 层（fastapi）包任意 EnvironmentBackend；toy 阶段用
MockEnvironmentBackend（真实文件语义，无容器），Mac 正式版换
LocalDockerBackend 而不改契约。SSH tunnel 提供 AutoDL 侧访问，
Docker daemon 绝不暴露公网（§8 拓扑）。

端点（版本 v1）:
    GET    /health
    POST   /v1/tasks/register           # Mac 侧登记 TaskSpec（server 持有）
    POST   /v1/envs                     # create: 内联 {"task"} 或引用 {"task_id"}
    POST   /v1/envs/{env_id}/reset
    POST   /v1/envs/{env_id}/actions    # execute(action) -> observation
    GET    /v1/envs/{env_id}/patch      # export_patch（GET: 非 mutating）
    DELETE /v1/envs/{env_id}            # request_id 放 query（DELETE 不带 body）
    POST   /v1/verifications            # verify(task, patch)（server 侧真实执行）

幂等（§8: 所有修改状态请求携带 request_id）: mutating 请求带 request_id 时
结果入幂等表；同 request_id 重放（客户端传输失败/5xx 重试）直接返回首次
响应，不重复执行——对 execute 尤其关键（第二次执行同一 str_replace 会因
"匹配不再唯一/不存在"而误报失败）。

CLI: python -m sweforge.environment.server [--port 8500]
"""

from __future__ import annotations

import argparse
import tempfile
import uuid
from collections import OrderedDict
from typing import Any, Callable

from sweforge.environment.base import EnvironmentBackend
from sweforge.environment.mock import MockEnvironmentBackend
from sweforge.protocol.tools import parse_tool_action
from sweforge.schemas.task import TaskSpec
from sweforge.schemas.verification import VerificationResult

__all__ = ["make_app", "main", "DEFAULT_PORT", "IDEMPOTENCY_CACHE_CAP"]

DEFAULT_PORT = 8500
#: 幂等表上限（toy server 防御性 cap; 超限丢最旧一半）
IDEMPOTENCY_CACHE_CAP = 10_000


def make_app(
    backend: EnvironmentBackend,
    request_ids: list[str] | None = None,
    idempotency_cache: OrderedDict[tuple[str, str, str], Any] | None = None,
):
    """§8 契约 v1 server。backend 可插拔（toy = Mock，Mac 正式 = LocalDocker）。

    request_ids（可选）: 收到每个 mutating 请求的 request_id 记录（测试断言用）。
    idempotency_cache（可选）: 幂等表; 不传则新建（每 app 独立, 不跨进程共享）。
    """
    from fastapi import FastAPI, HTTPException

    app = FastAPI()
    envs: dict[str, object] = {}
    tasks: dict[str, TaskSpec] = {}  # /v1/tasks/register 登记表（server 持有）
    rid_log = request_ids if request_ids is not None else []
    idem = idempotency_cache if idempotency_cache is not None else OrderedDict()

    def _idempotent(method: str, path: str, rid: str | None, fn: Callable[[], Any]) -> Any:
        """mutating 端点统一入口: 记录 rid -> 命中幂等表直接返回 -> 执行并入表。"""
        if rid:
            rid_log.append(rid)
            key = (method, path, rid)
            if key in idem:
                idem.move_to_end(key)
                return idem[key]
        result = fn()
        if rid:
            idem[(method, path, rid)] = result
            if len(idem) > IDEMPOTENCY_CACHE_CAP:
                for _ in range(len(idem) // 2):
                    idem.popitem(last=False)
        return result

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/v1/tasks/register")
    def register_task(body: dict):
        def _do():
            task = TaskSpec.model_validate(body["task"])
            tasks[task.task_id] = task
            return {"task_id": task.task_id}

        return _idempotent("POST", "/v1/tasks/register", body.get("request_id"), _do)

    @app.post("/v1/envs")
    def create_env(body: dict):
        def _do():
            if "task" in body:
                task = TaskSpec.model_validate(body["task"])
            elif "task_id" in body:
                task = tasks.get(body["task_id"])
                if task is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"task not registered: {body['task_id']}",
                    )
            else:
                raise HTTPException(
                    status_code=422, detail="body must carry 'task' or 'task_id'"
                )
            # env_id 每次 create 唯一（并发同 task_id 会互相覆盖）; 只动 env_id,
            # task.task_id 保持原值 —— bundle 查找 / verify 走原 task_id 不受影响
            env_id = f"{task.task_id}-{uuid.uuid4().hex[:8]}"
            envs[env_id] = backend.create(task)
            return {"env_id": env_id}

        return _idempotent("POST", "/v1/envs", body.get("request_id"), _do)

    @app.post("/v1/envs/{env_id}/reset")
    def reset_env(env_id: str, body: dict):
        def _do():
            env = envs.get(env_id)
            if env is None:
                raise HTTPException(status_code=404, detail=f"env not found: {env_id}")
            backend.reset(env)
            return {"ok": True}

        return _idempotent("POST", f"/v1/envs/{env_id}/reset", body.get("request_id"), _do)

    @app.post("/v1/envs/{env_id}/actions")
    def do_action(env_id: str, body: dict):
        def _do():
            env = envs.get(env_id)
            if env is None:
                raise HTTPException(status_code=404, detail=f"env not found: {env_id}")
            action = parse_tool_action(body["action"])
            obs = backend.execute(env, action)
            return {"observation": obs.model_dump(mode="json")}

        return _idempotent("POST", f"/v1/envs/{env_id}/actions", body.get("request_id"), _do)

    @app.get("/v1/envs/{env_id}/patch")
    def get_patch(env_id: str):
        env = envs.get(env_id)
        if env is None:
            raise HTTPException(status_code=404, detail=f"env not found: {env_id}")
        return {"patch": backend.export_patch(env)}

    @app.delete("/v1/envs/{env_id}")
    def destroy_env(env_id: str, request_id: str | None = None):
        def _do():
            env = envs.pop(env_id, None)
            if env is None:
                raise HTTPException(status_code=404, detail=f"env not found: {env_id}")
            backend.destroy(env)
            return {"ok": True}

        return _idempotent("DELETE", f"/v1/envs/{env_id}", request_id, _do)

    @app.post("/v1/verifications")
    def verify(body: dict):
        def _do():
            task = TaskSpec.model_validate(body["task"])
            v: VerificationResult = backend.verify(task, body["patch"])
            return {"verification": v.model_dump(mode="json")}

        return _idempotent("POST", "/v1/verifications", body.get("request_id"), _do)

    return app


def main(argv: list[str] | None = None) -> int:
    """CLI: toy Environment Server（Mock 后端, 只监听 127.0.0.1）。"""
    parser = argparse.ArgumentParser(
        prog="sweforge.environment.server",
        description="Mac Environment Server（§8 契约 v1; toy = Mock 后端, "
        "SSH tunnel 提供 AutoDL 侧访问）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="只监听本机（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workspace-root", default=None, help="Mock 后端工作区根目录")
    args = parser.parse_args(argv)

    import uvicorn

    backend = MockEnvironmentBackend(
        workspace_root=args.workspace_root or tempfile.mkdtemp(prefix="sweforge-server-"),
        seed_files={},
    )
    uvicorn.run(
        make_app(backend), host=args.host, port=args.port, log_level="info"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
