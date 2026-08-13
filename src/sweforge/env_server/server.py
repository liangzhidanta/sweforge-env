"""Mac-side Environment Server CLI: vendored make_app + LocalDockerBackend.

AutoDL 侧的 §8 server 是 `sweforge.environment.server.make_app(backend)`（vendored,
不可编辑）; 它的 CLI 只支持 Mock 后端。Mac 侧正式后端是 LocalDockerBackend
（本地临时目录, 或每 env 一个 Docker 容器）, 本模块是 Mac 侧自己的入口:

    python -m sweforge.env_server.server --bundles-dir examples [--docker]

只监听 127.0.0.1（SSH tunnel 提供 AutoDL 侧访问, Docker daemon 不暴露公网）。
可选用 `--token SECRET` 加 Bearer 认证（隧道暴露时的防未授权访问）; 未设
token 时服务与 AutoDL 契约客户端原样互通。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from sweforge.env_server.docker.backend import LocalDockerBackend
from sweforge.env_server.docker.manager import cleanup_stale_containers
from sweforge.environment.server import make_app

DEFAULT_PORT = 8500

logger = logging.getLogger("sweforge.env_server.server")


def bearer_auth_app(app, token: str):
    """Wrap an ASGI app so every request except /health needs `Authorization: Bearer <token>`.

    AutoDL 的 RemoteEnvironmentBackend 需要加一行请求头（见 USAGE.txt）; 未带
    token 的调用返回 401。lifespan 与 body 原样透传。
    """

    async def authed(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        if scope.get("path") == "/health":
            await app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        if headers.get(b"authorization") != f"Bearer {token}".encode():
            body = json.dumps({"error": "unauthorized"}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await app(scope, receive, send)

    return authed


def connection_banner_app(app):
    """在 AutoDL 侧第一次通过隧道创建环境时, 在 Mac 终端打印连接横幅。

    AutoDL Agent 调用 POST /v1/envs 即代表「两端已打通, LLM 可操作 Mac
    Docker coding 环境」。横幅只打印一次, 后续同环境操作由 uvicorn 访问
    日志逐条展示。
    """

    shown = {"value": False}

    async def wrapped(scope, receive, send):
        if (
            not shown["value"]
            and scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path", "").rstrip("/") == "/v1/envs"
        ):
            shown["value"] = True
            print(
                "\n" + "=" * 64 + "\n"
                "  [连接提示] AutoDL 已通过隧道连接到 Mac Docker!\n"
                "  LLM 现在可以操作 Mac 的 Docker coding 环境。\n"
                "  时间: "
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + "\n"
                + "=" * 64,
                file=sys.stderr,
                flush=True,
            )
        await app(scope, receive, send)

    return wrapped


def _tool_summary(action: dict) -> str:
    """把一条 ToolAction JSON 压成一行人类可读摘要（不含大段内容）。"""
    name = action.get("name")
    if name == "bash":
        cmd = (action.get("command") or "").strip()
        return f"bash cmd={cmd[:80]!r}" if cmd else "bash"
    if name == "search":
        q = (action.get("query") or "").strip()
        return f"search query={q[:60]!r}" if q else "search"
    if name == "view_file":
        path = action.get("path") or ""
        start, end = action.get("start_line"), action.get("end_line")
        rng = f":{start}-{end}" if start is not None and end is not None else ""
        return f"view_file path={path}{rng}"
    if name == "str_replace":
        path = action.get("path") or ""
        old = len(action.get("old_string") or "")
        new = len(action.get("new_string") or "")
        return f"str_replace path={path} old={old}c new={new}c"
    if name == "finish":
        s = (action.get("summary") or "")[:50]
        return f"finish summary={s!r}" if s else "finish"
    return name or "?"


def _summarize(method: str, path: str, req: dict, resp: dict, status: int | None) -> str:
    """把一次请求+回应的关键信息压成一行（env/tool/verdict…）。"""
    if method == "POST" and path == "/v1/envs":
        tid = (req.get("task") or {}).get("task_id") or req.get("task_id")
        eid = resp.get("env_id")
        return f"ENV_CREATE env={eid or tid or '?'} task={tid or '?'}"
    if method == "POST" and path == "/v1/tasks/register":
        return f"REGISTER task={resp.get('task_id') or '?'}"
    if method == "POST" and path.endswith("/reset"):
        return f"ENV_RESET env={path.split('/')[3]}"
    if method == "POST" and path.endswith("/actions"):
        env_id = path.split("/")[3]
        return f"ACTION env={env_id} {_tool_summary(req.get('action') or {})}"
    if method == "GET" and path.endswith("/patch"):
        env_id = path.split("/")[3]
        patch = resp.get("patch") or ""
        return f"PATCH env={env_id} chars={len(patch)} diff_files={patch.count('diff --git')}"
    if method == "DELETE":
        return f"ENV_DESTROY env={path.split('/')[3]}"
    if method == "POST" and path == "/v1/verifications":
        v = resp.get("verification") or {}
        f2p = len(v.get("fail_to_pass") or [])
        p2p = len(v.get("pass_to_pass") or [])
        return f"VERIFY verdict={v.get('verdict') or '?'} f2p={f2p} p2p={p2p}"
    return ""


def human_log_app(app):
    """给每个 HTTP 请求（除 /health）打一行人类可读日志: 请求 + Docker 回应关键信息。

    只读透传（capture request/response body 但原样转发）; 不改变响应。配合
    uvicorn access_log=False 使用, 终端只看到整理过的关键行。
    """

    async def wrapped(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path == "/health":
            await app(scope, receive, send)
            return
        method = scope.get("method", "")

        req_chunks: list[bytes] = []
        resp_chunks: list[bytes] = []
        status = {"value": None}

        async def recv():
            message = await receive()
            if message["type"] == "http.request":
                req_chunks.append(message.get("body", b""))
            return message

        async def snd(message):
            if message["type"] == "http.response.start":
                status["value"] = message["status"]
            elif message["type"] == "http.response.body":
                resp_chunks.append(message.get("body", b""))
            await send(message)

        await app(scope, recv, snd)

        try:
            req = json.loads(b"".join(req_chunks)) if req_chunks else {}
        except json.JSONDecodeError:
            req = {}
        try:
            resp = json.loads(b"".join(resp_chunks)) if resp_chunks else {}
        except json.JSONDecodeError:
            resp = {}
        detail = _summarize(method, path, req, resp, status["value"])
        if not detail:
            detail = f"HTTP {status['value'] or '?'} {method} {path}"
        logger.info(
            "[req] %s %s %s -> %s",
            datetime.now().strftime("%H:%M:%S"), method, path, detail,
        )

    return wrapped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweforge.env_server.server",
        description="Mac Environment Server（§8 契约 v1; LocalDockerBackend 后端）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="只监听本机（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--bundles-dir",
        default=None,
        help="task bundle 注册表根目录（含 <task_id>/task_manifest.json）; "
        "未登记的 task 走 setup 自建（Mock 语义）",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="用 DockerExecutor（需 daemon + 基础镜像, 见 env_server/docker/Dockerfile）; "
        "默认 LocalExecutor（临时目录, 无需 Docker）",
    )
    parser.add_argument("--image", default="sweforge-base")
    parser.add_argument(
        "--cleanup-stale",
        type=int,
        default=None,
        metavar="SECONDS",
        help="启动前清掉创建时间早于 SECONDS 秒的 sweforge-managed 泄漏容器"
        "（上一轮 server 崩溃未 destroy 的; 按 age 过滤, 不误杀活跃容器）",
    )
    parser.add_argument(
        "--token",
        default=None,
        metavar="SECRET",
        help="可选 Bearer 认证 token（防隧道暴露时未授权访问）; 缺省读"
        "SWEFORGE_TOKEN 环境变量; 都未设则不加认证（与 AutoDL 契约客户端原样互通）",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    bundles_dir = Path(args.bundles_dir) if args.bundles_dir else Path(tempfile.mkdtemp(prefix="sweforge-bundles-"))
    if args.cleanup_stale is not None:
        removed = cleanup_stale_containers(age_seconds=args.cleanup_stale)
        logger.info("cleanup_stale: removed %d leaked container(s)", len(removed))
    backend = LocalDockerBackend(bundles_dir=bundles_dir, use_docker=args.docker, image=args.image)
    token = args.token or os.environ.get("SWEFORGE_TOKEN")
    app = make_app(backend)
    if token:
        app = bearer_auth_app(app, token)
    app = connection_banner_app(app)
    app = human_log_app(app)
    logger.info(
        "Mac Environment Server: backend=%s bundles_dir=%s port=%s docker=%s auth=%s",
        "local-docker",
        bundles_dir,
        args.port,
        args.docker,
        "on" if token else "off",
    )

    import uvicorn

    # access_log=False: 默认访问日志太碎, 由 human_log_app 输出整理过的关键行
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
