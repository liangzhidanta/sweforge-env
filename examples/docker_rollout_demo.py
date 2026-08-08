"""AutoDL(模拟客户端) ↔ Mac server ↔ Docker 的交互, + 容器容量实测。

Part A —— 真实 HTTP 层交互: 起 uvicorn(make_app(LocalDockerBackend(--docker))),
  用 vendored RemoteEnvironmentBackend（AutoDL 侧同一份契约客户端）驱动完整流程
  health -> register -> create -> 五工具 -> verify -> destroy, 每步同时打印
  Mac 侧 docker 证据（docker ps / inspect）, 展示"AutoDL 一个请求在 Mac 上
  对应一次 docker 操作"（create -> 起容器, execute -> 容器内 docker exec,
  verify -> 全新临时容器, destroy -> 销毁容器）。

Part B —— 容量推测: 同时开 5 个 rollout 容器, docker stats 测空闲内存,
  在容器内跑 pytest 测 verify 峰值, 结合 colima VM 内存(8GB) 推测并发上限。

运行:
  NO_PROXY=127.0.0.1,localhost PYTHONPATH=src python examples/docker_rollout_demo.py
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

from sweforge.env_server.docker.backend import LocalDockerBackend, load_task_bundle
from sweforge.env_server.docker.executors import DockerExecutor
from sweforge.environment.remote import RemoteEnvironmentBackend
from sweforge.environment.server import make_app
from sweforge.protocol.tools import (
    FinishAction,
    SearchAction,
    StrReplaceAction,
    ViewFileAction,
)

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


def _managed_names() -> list[str]:
    out = _docker("ps", "-a", "--filter", "label=sweforge.managed=true", "--format", "{{.Names}}")
    return [n for n in out.splitlines() if n]


def _step(title: str) -> None:
    print(f"\n--- {title} ---")


def _start_server(backend: LocalDockerBackend):
    import uvicorn

    config = uvicorn.Config(make_app(backend), host="127.0.0.1", port=0, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(400):
        if srv.started:
            break
        time.sleep(0.05)
    if not srv.started:
        raise RuntimeError("server failed to start")
    port = srv.servers[0].sockets[0].getsockname()[1]
    return srv, thread, f"http://127.0.0.1:{port}"


def part_a_interaction(backend: LocalDockerBackend, task) -> None:
    print()
    print("############################################################")
    print("# Part A: AutoDL 客户端 ↔ Mac server ↔ Docker 交互           #")
    print("############################################################")
    print()
    print("拓扑: AutoDL Agent -> RemoteEnvironmentBackend -> HTTP(127.0.0.1)")
    print("      -> Mac server(make_app + LocalDockerBackend) -> docker 容器")
    print()
    srv, thread, base = _start_server(backend)
    try:
        print(f"  Mac server     = {base}")
        print(f"  AutoDL 客户端   = RemoteEnvironmentBackend(base_url={base})")
        print("                   (vendored 契约客户端, 与 AutoDL 侧同一份代码)")
        client = RemoteEnvironmentBackend(base_url=base, max_retries=1)

        print("\n--- [A1] health: 存活检查 ---")
        print("[AutoDL] GET /health")
        print("  <- 200 ok =", client.health())

        print("\n--- [A2] register: 登记任务 ---")
        print("[AutoDL] POST /v1/tasks/register  {task_id=toy_cache_aliasing}")
        print("  <- 200", client.register_task(task))

        print("\n--- [A3] create: AutoDL 发一个 create, Mac 就起一个容器 ---")
        print("[AutoDL] POST /v1/envs  {task_id=toy_cache_aliasing}")
        print("  -> make_app 路由到 LocalDockerBackend.create()")
        print("  -> under the hood, 对每个 env 发的 docker run 命令:")
        docker_cmd = DockerExecutor(
            image=backend.image,
            container_name="sweforge-task-<task_id前12位>-<随机8位>",
            task_id=task.task_id,
            env_id=task.task_id,
        ).create_command()
        print("     " + " \\\n     ".join(docker_cmd))
        print("  -> 再把 repo 快照 tar 进容器 (docker cp + 容器内 tar -xf),")
        print("     然后容器内 git init/add/commit 打 baseline。")
        env_id = client.create(task_id=task.task_id)
        print("  <- 200 env_id =", env_id)

        names = _managed_names()
        print("\n[Mac docker] 这个 create 请求在 Mac 侧留下的容器证据:")
        print("$ docker ps --filter label=sweforge.managed=true")
        print(_docker("ps", "--filter", "label=sweforge.managed=true",
                      "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}") or "(无)")
        if names:
            cname = names[-1]
            print(f"\n$ docker inspect {cname} (image/network/user/labels)")
            print(_docker("inspect", "--format",
                          "Image={{.Config.Image}}  NetworkMode={{.HostConfig.NetworkMode}}  "
                          "User={{.Config.User}}  Labels={{.Config.Labels}}", cname))

        print("\n--- [A4] execute: AutoDL 的每个工具调用, 都在容器内 docker exec 执行 ---")
        obs = client.execute(env_id, ViewFileAction(path="toy_cache/cache.py", start_line=1, end_line=12))
        print("[AutoDL] POST /v1/envs/toy_cache_aliasing/actions  {view_file}")
        print("  <- ", end="")
        if obs.error:
            print(f"error={obs.error}")
        else:
            print(f"total_lines={obs.total_lines}, 前 3 行:\n     " +
                  "\n     ".join(obs.content.splitlines()[:3]))

        sobs = client.execute(env_id, SearchAction(query="_CACHE"))
        print("[AutoDL] POST /v1/envs/toy_cache_aliasing/actions  {search '_CACHE'}")
        print(f"  <- matches={len(sobs.matches)}" +
              (f", 例: {sobs.matches[0].path}:{sobs.matches[0].line}"
               if sobs.matches else ""))

        srobs = client.execute(env_id, StrReplaceAction(
            path="toy_cache/cache.py", old_string=OLD, new_string=NEW))
        print("[AutoDL] POST /v1/envs/toy_cache_aliasing/actions  {str_replace}")
        print(f"  <- success={srobs.success}" + (f", error={srobs.error}" if not srobs.success else ""))

        fobs = client.execute(env_id, FinishAction())
        patch = fobs.patch or ""
        n_files = sum(1 for line in patch.splitlines() if line.startswith("diff --git"))
        print("[AutoDL] POST /v1/envs/toy_cache_aliasing/actions  {finish}")
        print(f"  <- 导出 patch（{n_files} 个文件）")
        print("$ docker ps --filter label=sweforge.managed=true  (全程只有 1 个 rollout 容器)")
        print(_docker("ps", "--filter", "label=sweforge.managed=true",
                      "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}"))

        print("\n--- [A5] verify: 全新临时容器干净验证（注入隐藏测试） ---")
        print("[AutoDL] POST /v1/verifications  {task, patch}")
        print("  -> make_app 调 LocalDockerBackend.verify(): 另起一个临时容器,")
        print("     apply patch + 注入 private/hidden_tests + 跑 F2P/P2P/integrity")
        v = client.verify(task, patch)
        print(json.dumps({
            "verdict": v.verdict,
            "f2p_passed": v.f2p_passed,
            "p2p_passed": v.p2p_passed,
            "integrity_ok": v.integrity_ok,
            "reward": v.reward,
        }, ensure_ascii=False, indent=2))
        print("$ docker ps -a --filter label=sweforge.managed=true  (verify 临时容器已销毁)")
        print(_docker("ps", "-a", "--filter", "label=sweforge.managed=true",
                      "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}") or "(无)")

        print("\n--- [A6] verify 容器长什么样（独立演示 fresh workspace + 隐藏测试） ---")
        vexec = backend._make_executor(backend._bundle(task), "verify")
        try:
            vname = vexec.container_name
            print(f"verify container = {vname}")
            print(_docker("inspect", "--format",
                          "Image={{.Config.Image}}  NetworkMode={{.HostConfig.NetworkMode}}  "
                          "User={{.Config.User}}", vname))
            print("workspace 里的文件（含注入的隐藏测试 tests/）:")
            for rel in vexec.list_files():
                if "hidden_tests" not in rel:
                    print(f"  {rel}")
        finally:
            vexec.close()

        print("\n--- [A7] destroy: AutoDL 发 DELETE, Mac 销毁 rollout 容器 ---")
        print("[AutoDL] DELETE /v1/envs/toy_cache_aliasing")
        client.destroy(env_id)
        print("  <- 200, rollout 容器已销毁")
        print("$ docker ps -a --filter label=sweforge.managed=true")
        print(_docker("ps", "-a", "--filter", "label=sweforge.managed=true",
                      "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}") or "(无泄漏容器)")

        client.close()
    finally:
        srv.should_exit = True
        thread.join(timeout=5)


def _to_mib(s: str) -> float:
    s = s.strip()
    if s.endswith("KiB"):
        return float(s[:-3]) / 1024
    if s.endswith("MiB"):
        return float(s[:-3])
    if s.endswith("GiB"):
        return float(s[:-3]) * 1024
    return 0.0


def _colima_vm_mem_gib() -> float:
    """colima VM 内存(GiB), 读 ~/.colima/default/colima.yaml 的 memory: N。"""
    try:
        yaml_path = Path.home() / ".colima" / "default" / "colima.yaml"
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("memory:"):
                return float(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return 8.0


def _host_ram_gib() -> float:
    try:
        out = subprocess.run(("sysctl", "-n", "hw.memsize"),
                             capture_output=True, text=True).stdout.strip()
        return float(out) / 1024 / 1024 / 1024
    except Exception:
        return 8.0


def _colima_available_mib() -> float:
    """colima VM 当前可用内存(MiB), 解析 `colima ssh -- free -m` 的 available。"""
    try:
        out = subprocess.run(("colima", "ssh", "--", "free", "-m"),
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Mem:"):
                return float(line.split()[6])
    except Exception:
        pass
    return _colima_vm_mem_gib() * 1024 * 0.9


def part_b_capacity(backend: LocalDockerBackend, task) -> None:
    n = 5
    vm_gib = _colima_vm_mem_gib()
    print()
    print("############################################################")
    print(f"# Part B: 容器容量实测 —— 同时开 {n} 个 rollout 容器并推测上限        #")
    print("############################################################")
    print()
    print("本机资源:")
    print(f"  宿主 RAM      = {_host_ram_gib():.1f} GiB")
    print(f"  colima VM     = {vm_gib:.0f} GiB 内存 / 4 CPU  (~/.colima/default/colima.yaml)")
    print("  每容器限制     = --cpus 2 --memory 4g --pids-limit 128 (ContainerLimits)")
    print("  (4g 是单容器硬上限, 不是典型占用)")
    print()
    print(f"-- 阶段 1: 同时创建 {n} 个 rollout 容器（同一任务, 模拟一次问题开多个 rollout）--")
    envs = [backend.create(task) for _ in range(n)]
    names = [e.executor.container_name for e in envs]
    print(f"  已创建 {len(envs)} 个: {', '.join(names)}")
    print("$ docker ps --filter label=sweforge.managed=true")
    print(_docker("ps", "--filter", "label=sweforge.managed=true",
                  "--format", "table {{.Names}}\t{{.Status}}"))

    print("\n$ docker stats --no-stream (每个容器的内存/CPU 占用, 空闲基线)")
    print(_docker("stats", "--no-stream",
                  "--format", "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}",
                  *names))
    raw = _docker("stats", "--no-stream", "--format", "{{.MemUsage}}", *names)
    idle_mib = [_to_mib(line.split("/")[0]) for line in raw.splitlines() if line.strip()]
    idle_avg = sum(idle_mib) / len(idle_mib) if idle_mib else 0.0

    print("\n-- 阶段 2: 在一个容器里跑完整 pytest（真实验证负载）, 量它进程峰值内存 --")
    print("  (在容器内用 resource.getrusage(RUSAGE_CHILDREN).ru_maxrss 取 pytest")
    print("   子进程峰值, 比外部轮询 docker stats 更准: toy 测试 ~0.01s 就跑完,"
          " 轮询根本赶不上)")
    peak_script = (
        "import resource, subprocess, sys\n"
        "rc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider'], cwd='/workspace')\n"
        "print('pytest_rc=', rc.returncode)\n"
        "kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss\n"
        "print('maxrss_mib=%.1f' % (kb / 1024))\n"
    )
    result = envs[0].executor.run_argv(("python", "-c", peak_script), timeout=180, cwd=".")
    peak_mib = 0.0
    pytest_rc = None
    for line in result.stdout.splitlines():
        if line.startswith("maxrss_mib="):
            peak_mib = float(line.split("=", 1)[1])
        elif line.startswith("pytest_rc="):
            pytest_rc = int(line.split("=", 1)[1])
    print(f"  pytest 退出码 = {pytest_rc}")
    print(f"  pytest 进程峰值内存 = {peak_mib:.0f} MiB (ru_maxrss)")

    print("\n$ colima ssh -- free -m   (colima VM 内存现状)")
    print(subprocess.run(("colima", "ssh", "--", "free", "-m"),
                         capture_output=True, text=True).stdout.strip())
    usable_mib = _colima_available_mib()

    print("\n-- 推测: 本机并发 rollout 上限（内存维度） --")
    overhead = 30.0  # 每容器页面缓存 + docker 层开销(MiB), 保守估计
    per_container = peak_mib + overhead
    n_peak = usable_mib / per_container if per_container else 0.0
    print(f"  每容器空闲基线   ≈ {idle_avg:.1f} MiB  (docker stats, sleep 进程)")
    print(f"  每容器 pytest 峰值 ≈ {peak_mib:.0f} MiB  (ru_maxrss)")
    print(f"  每容器估算开销   ≈ +{overhead:.0f} MiB (页面缓存/docker 层)")
    print(f"  可用内存         ≈ {usable_mib:.0f} MiB (colima VM free 的 available)")
    print(f"  并发上限(内存)   ≈ {n_peak:.0f} 个 rollout 容器同时跑")
    print(f"  留 50% 余量      → 建议 ≈ {n_peak*0.5:.0f} 个并发 rollout")
    print("  注: 这是内存维度估计; CPU(VM 4 核, 每容器最多 2 核) 与 pids-limit")
    print("      (每容器 128) 也会约束并发, 实际以压测为准。")

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
    print(f"image (task)   = {task.environment.image or '(未指定, 用默认)'}")
    print(f"image (resolved)= {backend._resolve_image(task)}")
    part_a_interaction(backend, task)
    part_b_capacity(backend, task)


if __name__ == "__main__":
    main()
