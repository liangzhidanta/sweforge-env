"""真实 coding 环境容器 + 并发真实交互（raw docker 输出）。

Part A —— 部署一个真实 coding 环境容器（sweforge-base + toy_cache 仓库,
  并把 Mac 私有的隐藏 F2P 测试注入容器）, 用原始 docker exec 走一遍 agent 的
  完整动作: 读文件(cat) -> 搜索(grep) -> 跑测试(pytest, 复现 bug 失败)
  -> 修复(python 改文件) -> 再跑测试(通过) -> 导 patch(git diff)。
  每一步打印真实的 docker 命令与容器返回的原始输出 —— 这就是"docker 传出的内容"。

Part B —— 并发真实交互: 同时创建 5 个 rollout 容器, 用 Barrier 让 5 个容器
  同一时刻开跑「修复 + 连续 pytest」真实负载, 用 docker ps / docker top /
  docker stats / 墙钟时间证明 5 个容器是真实并行工作, 而不是空跑的 sleep 容器。

运行: NO_PROXY=127.0.0.1,localhost PYTHONPATH=src python examples/docker_coding_demo.py
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from sweforge.env_server.docker.backend import LocalDockerBackend, load_task_bundle

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
BUNDLE = EXAMPLES / "toy_cache_aliasing"
HIDDEN_F2P = BUNDLE / "private" / "hidden_tests" / "tests" / "test_f2p.py"

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

# 在容器内用 python heredoc 做 str_replace 修复（等价于工具里的 str_replace）。
FIX_SCRIPT = (
    "python3 - <<'PYEOF'\n"
    "import pathlib\n"
    "p = pathlib.Path('/workspace/toy_cache/cache.py')\n"
    "s = p.read_text()\n"
    f"old = {OLD!r}\n"
    f"new = {NEW!r}\n"
    "n = s.count(old)\n"
    "if n == 1:\n"
    "    p.write_text(s.replace(old, new))\n"
    "    print('str_replace: 1 occurrence replaced -> cache keyed by key, not value')\n"
    "elif n == 0:\n"
    "    print('str_replace: already applied (0 occurrences)')\n"
    "else:\n"
    "    raise AssertionError(f'expected at most 1 occurrence, got {n}')\n"
    "PYEOF\n"
)


def _docker(*args: str) -> str:
    return subprocess.run(("docker", *args), capture_output=True, text=True).stdout.strip()


def _run(cname: str, script: str):
    """Run `docker exec <cname> /bin/sh -c '<script>'`, echo command + raw output."""
    if "\n" in script:
        print(f"$ docker exec {cname} /bin/sh -c <<'SHELL'")
        for line in script.rstrip("\n").splitlines():
            print(f"    {line}")
        print("    SHELL")
    else:
        print(f"$ docker exec {cname} /bin/sh -c '{script}'")
    cp = subprocess.run(("docker", "exec", cname, "/bin/sh", "-c", script),
                        capture_output=True, text=True)
    if cp.stdout:
        print(cp.stdout.rstrip("\n"))
    if cp.stderr:
        print(cp.stderr.rstrip("\n"))
    print(f"  (exit {cp.returncode})")
    return cp


def _inject_hidden_f2p(cname: str) -> None:
    """与 verify 同款机制（executor.write_text）: 以容器 uid 1000 写入, 属主正确。"""
    content = HIDDEN_F2P.read_text(encoding="utf-8")
    script = f"cat > /workspace/tests/test_f2p.py <<'PYEOF'\n{content}PYEOF\n"
    subprocess.run(("docker", "exec", cname, "/bin/sh", "-c", script),
                   capture_output=True, text=True, check=True)


def _to_mib(s: str) -> float:
    s = s.strip()
    if s.endswith("KiB"):
        return float(s[:-3]) / 1024
    if s.endswith("MiB"):
        return float(s[:-3])
    if s.endswith("GiB"):
        return float(s[:-3]) * 1024
    return 0.0


def part_a_coding(backend: LocalDockerBackend, task) -> None:
    print()
    print("############################################################")
    print("# Part A: 部署真实 coding 环境容器, 看 docker 传出的原始内容   #")
    print("############################################################")
    env = backend.create(task)
    cname = env.executor.container_name
    try:
        print(f"已部署 coding 环境容器: {cname}  (镜像 {backend.image}, 内含 toy_cache 仓库)")
        print("$ docker ps --filter label=sweforge.managed=true")
        print(_docker("ps", "--filter", "label=sweforge.managed=true",
                      "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}"))
        print("$ docker inspect --format 'Image/NetworkMode/User/Labels'")
        print(_docker("inspect", "--format",
                      "Image={{.Config.Image}}  NetworkMode={{.HostConfig.NetworkMode}}  "
                      "User={{.Config.User}}  Labels={{.Config.Labels}}", cname))

        print("\n-- 把 Mac 私有的隐藏 F2P 测试注入容器（与 verify 同款机制: docker exec 写入） --")
        _inject_hidden_f2p(cname)
        print(f"$ docker exec {cname} ls /workspace/tests/")
        print(_docker("exec", cname, "ls", "/workspace/tests/"))

        print("\n-- [1] agent 读文件 -> docker exec cat（原始内容） --")
        _run(cname, "cat /workspace/toy_cache/cache.py")

        print("\n-- [2] agent 搜索 -> docker exec grep --")
        _run(cname, 'grep -rn "_CACHE" /workspace/toy_cache')

        print("\n-- [3] agent 跑测试 -> docker exec pytest（隐藏 F2P 应失败 = 复现 bug） --")
        _run(cname, "cd /workspace && python3 -m pytest -q -p no:cacheprovider")

        print("\n-- [4] agent 修复 -> docker exec python 做 str_replace --")
        _run(cname, FIX_SCRIPT)

        print("\n-- [5] 修复后再读 -> docker exec cat（对比第 1 步） --")
        _run(cname, "cat /workspace/toy_cache/cache.py")

        print("\n-- [6] agent 再跑测试 -> docker exec pytest（应全部通过） --")
        _run(cname, "cd /workspace && python3 -m pytest -q -p no:cacheprovider")

        print("\n-- [7] agent 导出 patch -> docker exec git diff --")
        _run(cname, "cd /workspace && git diff --no-ext-diff -- toy_cache/cache.py")
    finally:
        backend.destroy(env)
    print("$ docker ps -a --filter label=sweforge.managed=true  (已销毁)")
    print(_docker("ps", "-a", "--filter", "label=sweforge.managed=true",
                  "--format", "table {{.Names}}\t{{.Status}}") or "(无泄漏容器)")


def _run_fix_and_pytest(env, iters: int):
    """在一个容器里做真实交互: 修复 bug + 连续跑 iters 次 pytest。"""
    cname = env.executor.container_name
    fix = subprocess.run(("docker", "exec", cname, "/bin/sh", "-c", FIX_SCRIPT),
                         capture_output=True, text=True)
    result = env.executor.run_shell(
        f"i=0; while [ $i -lt {iters} ]; do "
        "python3 -m pytest -q -p no:cacheprovider || exit $?; i=$((i+1)); done",
        timeout=240, cwd=".")
    return cname, fix.returncode, result.exit_code, result


def part_b_concurrency(backend: LocalDockerBackend, task) -> None:
    n = 5
    iters = 12
    print()
    print("############################################################")
    print(f"# Part B: 并发真实交互 —— {n} 个容器同时做真实修复+测试          #")
    print("############################################################")
    print()
    print("-- 创建 5 个 rollout 容器（各带真实仓库 + 注入隐藏 F2P 测试） --")
    envs = [backend.create(task) for _ in range(n)]
    names = [e.executor.container_name for e in envs]
    for cname in names:
        _inject_hidden_f2p(cname)
    print(f"  已创建 {len(envs)} 个: {', '.join(names)}")
    print("$ docker ps --filter label=sweforge.managed=true")
    print(_docker("ps", "--filter", "label=sweforge.managed=true",
                  "--format", "table {{.Names}}\t{{.Status}}"))

    print("\n-- 先测单个容器的耗时（做一遍 fix + pytest×%d 需要多久） --" % iters)
    t0 = time.monotonic()
    c0, fix0, pyt0, res0 = _run_fix_and_pytest(envs[0], iters)
    t_seq = time.monotonic() - t0
    print(f"  容器 {c0}: fix_rc={fix0} pytest_rc={pyt0}  耗时 {t_seq:.1f}s")
    print(f"  5 个如果串行做会花 ≈ {t_seq*5:.1f}s; 如果并行 ≈ {t_seq:.1f}s")

    print("\n-- Barrier 让 5 个容器同一时刻开跑同样的真实负载 --")
    barrier = threading.Barrier(n)
    results: dict[int, tuple] = {}
    stop = threading.Event()
    peak = {"mib": 0.0}

    def _poll() -> None:
        while not stop.is_set():
            raw = _docker("stats", "--no-stream", "--format", "{{.MemUsage}}", *names)
            for line in raw.splitlines():
                try:
                    peak["mib"] = max(peak["mib"], _to_mib(line.split("/")[0]))
                except (IndexError, ValueError):
                    pass
            time.sleep(0.15)

    def _work(i: int, env) -> None:
        try:
            barrier.wait(timeout=15)
            t0 = time.monotonic()
            cname, fr, pr, r = _run_fix_and_pytest(env, iters)
            results[i] = (cname, fr, pr, time.monotonic() - t0)
        except Exception:
            results[i] = (env.executor.container_name, -1, -1, 0.0)

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()
    wall0 = time.monotonic()
    threads = [threading.Thread(target=_work, args=(i, envs[i])) for i in range(n)]
    for t in threads:
        t.start()
    time.sleep(1.2)

    print("\n  运行中证据（5 个容器都在干活）:")
    print("$ docker ps  (5 个同时 Up)")
    print(_docker("ps", "--filter", "label=sweforge.managed=true",
                  "--format", "table {{.Names}}\t{{.Status}}"))
    print(f"$ docker top {names[0]}  (容器 0 里实际进程: 应能看到 python pytest 在跑)")
    top = _docker("top", names[0]) or "(top 无输出)"
    print(top)
    print("$ docker stats --no-stream  (5 个同时的内存, 都应 > 空闲基线 0.6MiB)")
    print(_docker("stats", "--no-stream",
                  "--format", "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}", *names))

    for t in threads:
        t.join()
    stop.set()
    poller.join(timeout=3)
    wall = time.monotonic() - wall0

    print("\n-- 并发结果 --")
    durs = [results[i][3] for i in range(n)]
    for i in range(n):
        cname, fr, pr, dur = results[i]
        print(f"  {cname}: fix_rc={fr}  pytest_rc={pr}  本容器耗时 {dur:.1f}s")
    saved = (1 - wall / (t_seq * n)) * 100 if t_seq > 0 else 0.0
    print(f"  5 个并发总墙钟 = {wall:.1f}s   (若串行会 ≈ {t_seq*n:.1f}s -> 省 {saved:.0f}% 时间)")
    print(f"  每个容器本机耗时 {min(durs):.1f}-{max(durs):.1f}s（含 5 容器抢 4 核的 CPU 争抢）")
    print(f"  并发期间 docker stats 峰值(单容器) ≈ {peak['mib']:.0f} MiB (空闲基线 0.6 MiB)")
    print("  结论: 墙钟 << 串行时间, 且 docker ps/top/stats 同时抓到 5 个容器都在真实工作,")
    print("        证明 5 个 rollout 容器是并行运行的真实交互, 不是空跑。")

    print("\n-- 清理: 销毁 5 个容器 --")
    for env in envs:
        backend.destroy(env)
    print("$ docker ps -a --filter label=sweforge.managed=true")
    print(_docker("ps", "-a", "--filter", "label=sweforge.managed=true",
                  "--format", "table {{.Names}}\t{{.Status}}") or "(无泄漏容器)")


def main() -> None:
    task = load_task_bundle(BUNDLE).task
    backend = LocalDockerBackend(EXAMPLES, use_docker=True)
    print(f"task_id        = {task.task_id}")
    print(f"image (resolved)= {backend._resolve_image(task)}")
    part_a_coding(backend, task)
    part_b_concurrency(backend, task)


if __name__ == "__main__":
    main()
