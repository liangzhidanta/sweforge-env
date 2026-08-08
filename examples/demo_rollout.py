"""端到端演示：打开一个 agent 环境 -> 查看源码 -> 修复 bug -> 导出补丁 -> 干净验证。

运行:  PYTHONPATH=src python examples/demo_rollout.py [--docker]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sweforge.env_server.docker.backend import LocalDockerBackend, load_task_bundle
from sweforge.protocol.tools import SearchAction, StrReplaceAction, ViewFileAction

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action="store_true", help="用真实 Docker 容器（需 daemon + sweforge-base 镜像）")
    args = parser.parse_args(argv)

    task = load_task_bundle(EXAMPLES).task
    backend = LocalDockerBackend(EXAMPLES.parent, use_docker=args.docker)

    print("=" * 64)
    print(f"任务: {task.task_id}")
    print(f"问题描述: {task.problem_statement}")
    print(f"F2P (隐藏测试): {[t.test_id for t in task.fail_to_pass]}")
    print(f"P2P (公开测试): {[t.test_id for t in task.pass_to_pass]}")
    print(f"backend: {'docker' if args.docker else 'local'}")
    print("=" * 64)

    print("\n[1] 创建一个干净的 agent 工作区 (env)")
    env = backend.create(task)
    try:
        print(f"    env_id   = {env.env_id}")
        print(f"    root     = {env.executor.root}")

        print("\n[2] view_file 查看有 bug 的源码")
        obs = backend.execute(env, ViewFileAction(path="toy_cache/cache.py", start_line=1, end_line=20))
        print(obs.content)

        print("[3] search 定位 _CACHE 的使用位置")
        obs = backend.execute(env, SearchAction(query="_CACHE"))
        print(f"    命中 {len(obs.matches)} 处" + ("（截断）" if obs.truncated else "") + ":")
        for match in obs.matches[:10]:
            print(f"    {match.path}:{match.line}: {match.content}")

        print("\n[4] 先用空补丁验证 buggy 版本（预期: F2P 失败 / P2P 通过）")
        before = backend.verify(task, "")
        print(f"    verdict={before.verdict}  F2P={before.f2p_passed}/{len(before.fail_to_pass)}  "
              f"P2P={before.p2p_passed}/{len(before.pass_to_pass)}")

        print("\n[5] 用 str_replace 修复 bug（把按 value 缓存改成按 key 缓存）")
        obs = backend.execute(env, StrReplaceAction(path="toy_cache/cache.py", old_string=OLD, new_string=NEW))
        print(f"    success={obs.success}")

        print("\n[6] 导出 git 补丁")
        patch = backend.export_patch(env)
        print(patch)

        print("[7] 干净验证（全新 workspace + 注入隐藏测试后跑 F2P/P2P）")
        after = backend.verify(task, patch)
        print(f"    verdict={after.verdict}  F2P={after.f2p_passed}/{len(after.fail_to_pass)}  "
              f"P2P={after.p2p_passed}/{len(after.pass_to_pass)}  "
              f"integrity_ok={after.integrity_ok}  reward={after.reward}")
    finally:
        backend.destroy(env)
        print("\n[done] env 已销毁，临时工作区已清理")


if __name__ == "__main__":
    main()
