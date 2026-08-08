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
import tempfile
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
    logger.info(
        "Mac Environment Server: backend=%s bundles_dir=%s port=%s docker=%s auth=%s",
        "local-docker",
        bundles_dir,
        args.port,
        args.docker,
        "on" if token else "off",
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
