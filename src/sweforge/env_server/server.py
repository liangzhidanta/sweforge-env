"""Mac-side Environment Server CLI: vendored make_app + LocalDockerBackend.

AutoDL 侧的 §8 server 是 `sweforge.environment.server.make_app(backend)`（vendored,
不可编辑）; 它的 CLI 只支持 Mock 后端。Mac 侧正式后端是 LocalDockerBackend
（本地临时目录, 或每 env 一个 Docker 容器）, 本模块是 Mac 侧自己的入口:

    python -m sweforge.env_server.server --bundles-dir examples [--docker]

只监听 127.0.0.1（SSH tunnel 提供 AutoDL 侧访问, Docker daemon 不暴露公网）。
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from sweforge.env_server.docker.backend import LocalDockerBackend
from sweforge.environment.server import make_app

DEFAULT_PORT = 8500

logger = logging.getLogger("sweforge.env_server.server")


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
    args = parser.parse_args(argv)

    bundles_dir = Path(args.bundles_dir) if args.bundles_dir else Path(tempfile.mkdtemp(prefix="sweforge-bundles-"))
    backend = LocalDockerBackend(bundles_dir=bundles_dir, use_docker=args.docker, image=args.image)
    logger.info(
        "Mac Environment Server: backend=%s bundles_dir=%s port=%s docker=%s",
        "local-docker",
        bundles_dir,
        args.port,
        args.docker,
    )

    import uvicorn

    uvicorn.run(make_app(backend), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
