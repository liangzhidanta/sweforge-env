"""模拟 AutoDL 任务在 Mac 侧创建 rollout 容器并给出 docker 证据。

流程（与 §8 契约一致）:
    登记/加载任务 -> create 出每 env 一个 rollout 容器 -> 容器内跑五工具
    -> 导出 git patch -> 全新容器干净验证（注入隐藏测试）-> 销毁。

每一步旁打印 docker 输出（docker ps / docker inspect）, 直观展示:
    - 容器镜像 = task.environment.image 解析结果（本任务默认 sweforge-base）
    - --network none / --user 1000:1000 / sweforge.managed 标签
    - verify 用独立临时容器, 生命周期与 rollout 容器隔离
    - destroy 后无泄漏容器

运行:  PYTHONPATH=src python examples/docker_rollout_demo.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sweforge.env_server.docker.backend import LocalDockerBackend, load_task_bundle
from sweforge.protocol.tools import FinishAction, SearchAction, StrReplaceAction, ViewFileAction

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
BUNDLE = EXAMPLES / "toy_cache_aliasing"

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


def _docker(*args: str) -> str:
    return subprocess.run(("docker", *args), capture_output=True, text=True).stdout.strip()


def _step(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    task = load_task_bundle(BUNDLE).task
    backend = LocalDockerBackend(EXAMPLES, use_docker=True)

    print(f"task_id        = {task.task_id}")
    print(f"image (task)   = {task.environment.image or '(未指定, 用默认)'}")
    print(f"image (resolved)= {backend._resolve_image(task)}")

    _step("[1] create: AutoDL 任务 -> rollout 容器")
    env = backend.create(task)
    name = env.executor.container_name
    print(f"env_id         = {env.env_id}")
    print(f"container      = {name}")
    print("$ docker ps --filter label=sweforge.managed=true")
    print(_docker("ps", "--no-trunc", "--filter", "label=sweforge.managed=true",
                  "--format", "table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}"))
    print(f"$ docker inspect {name} (image/network/user/labels)")
    print(_docker("inspect", "--format",
                  "Image={{.Config.Image}}  NetworkMode={{.HostConfig.NetworkMode}}  "
                  "User={{.Config.User}}  Labels={{.Config.Labels}}", name))

    _step("[2] 容器内跑 agent 工具")
    obs = backend.execute(env, ViewFileAction(path="toy_cache/cache.py", start_line=1, end_line=12))
    print(f"view_file: {len(obs.content.splitlines())} 行渲染成功")
    obs = backend.execute(env, SearchAction(query="_CACHE"))
    print(f"search: 命中 {len(obs.matches)} 处")
    obs = backend.execute(env, StrReplaceAction(path="toy_cache/cache.py", old_string=OLD, new_string=NEW))
    print(f"str_replace: success={obs.success}")
    obs = backend.execute(env, FinishAction())
    patch = obs.patch
    n_files = sum(1 for line in patch.splitlines() if line.startswith("diff --git"))
    print(f"finish: 导出 patch（{n_files} 个文件）")

    _step("[3] clean verify（全新临时容器 + 注入隐藏测试）")
    result = backend.verify(task, patch)
    print(json.dumps({
        "verdict": result.verdict,
        "f2p_passed": result.f2p_passed,
        "p2p_passed": result.p2p_passed,
        "integrity_ok": result.integrity_ok,
        "reward": result.reward,
        "metadata": {"backend": result.metadata.get("backend"), "docker": result.metadata.get("docker")},
    }, ensure_ascii=False, indent=2))
    print("$ docker ps -a (verify 用临时容器应已销毁, 只剩 rollout 容器)")
    print(_docker("ps", "-a", "--filter", "label=sweforge.managed=true",
                  "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}") or "(无)")

    _step("[4] 独立验证容器演示（fresh workspace + 隐藏测试注入）")
    vexec = backend._make_executor(backend._bundle(task), "verify")
    vname = vexec.container_name
    print(f"verify container = {vname}")
    print(f"$ docker inspect {vname} (image/network/user)")
    print(_docker("inspect", "--format",
                  "Image={{.Config.Image}}  NetworkMode={{.HostConfig.NetworkMode}}  User={{.Config.User}}",
                  vname))
    print("workspace 里的文件（含注入的隐藏测试 tests/）:")
    for rel in vexec.list_files():
        if "hidden_tests" not in rel:
            print(f"  {rel}")
    vexec.close()

    _step("[5] destroy rollout 容器")
    backend.destroy(env)
    print("$ docker ps -a --filter label=sweforge.managed=true")
    print(_docker("ps", "-a", "--filter", "label=sweforge.managed=true", "--format", "{{.Names}}") or "(无泄漏容器)")


if __name__ == "__main__":
    main()
