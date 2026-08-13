#!/usr/bin/env python3
"""Mac 侧 Docker 层监控（只读观察器; 不修改 server / 不重启 / 不动隧道）。

GRPO / 联调训练时 AutoDL 经隧道把动作打到 Mac Docker。本脚本以 Docker 层
信号给出人类可读视图:
  - /health 轮询        -> 服务健康（--health-port, 默认 8501 = AutoDL 实际连接的服务）
  - docker events       -> 容器生命周期（create/start/die/destroy）+ 每次 exec 的工具命令
  - docker stats        -> 运行中容器 CPU / 内存 / PID 快照
setup 内部 exec（git / tar / .git 元数据 / test / find）默认折叠, 只显示
agent 动作与生命周期; --all 显示全部。

注意: 这是 Docker 层监控, 看不到 HTTP 层 verify 的 verdict（resolved/…）。
agent 动作以「exec 命令」形式呈现。

用法:
  python scripts/monitor_docker.py [--health-port 8501] [--snapshot 10]
                                   [--all] [--no-commands]

输出行示例:
  [19:15:01] [env:create] role=task env=hum-v5-remove_conditional-013-...
  [19:15:02] [action:bash] env=hum-v5-... /bin/sh -c python -m pip install ...
  [19:15:03] [env:die  ] env=hum-v5-... exit=0
  [19:15:04] [env:rm   ] env=hum-v5-...
  [19:15:10] ── 快照: health=OK 运行容器=3 | 累计: create=… agent_exec=… ──
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import threading
import time
import urllib.request
from datetime import datetime


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def fmt_epoch(secs: str) -> str:
    try:
        return datetime.fromtimestamp(int(secs)).strftime("%H:%M:%S")
    except ValueError:
        return str(secs)


def run(args, timeout=8) -> tuple[int, str, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def classify(cmd: str) -> str:
    """把一条 exec 命令分类: agent 动作 vs setup 内部行为。"""
    c = cmd.strip()
    # agent bash 动作永远显示
    if c.startswith("/bin/sh -c ") or c.startswith("sh -c ") or c.startswith("bash -c "):
        return "bash"
    # 以下属 setup 内部行为, 默认折叠（放在 cat/test 之前: .git 下读取
    # 是 git/commit 钩子的内部行为, 不是 agent 的 view/search）
    if c.startswith("git ") or c.startswith("tar -"):
        return "setup"
    if "/.git/" in c:
        return "setup"
    if c.startswith("test ") or c.startswith("find "):
        return "setup"
    if c.startswith("cat >"):
        return "write"
    if c.startswith("cat "):
        return "read"
    if c.startswith("grep"):
        return "search"
    return "cmd"


def event_reader(docker: str, q: "queue.Queue[tuple[str, str]]", stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            proc = subprocess.Popen(
                (
                    docker, "events", "--filter", "type=container",
                    "--format", "{{json .}}",
                ),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except Exception as exc:  # docker 不存在等情况
            q.put(("MSG", f"docker events 启动失败: {exc}"))
            time.sleep(3)
            continue
        for raw in proc.stdout:
            if stop.is_set():
                proc.terminate()
                return
            q.put(("EVT", raw.rstrip("\n")))
        proc.wait()
        if not stop.is_set():
            q.put(("MSG", "docker events 连接断开, 3s 后自动重连..."))
            time.sleep(3)


def fmt_event(raw: str, args: argparse.Namespace, counters: dict) -> str | None:
    try:
        event = json.loads(raw)
    except Exception:
        return None
    attrs = event.get("Actor", {}).get("Attributes", {}) or {}
    env = attrs.get("sweforge.env_id") or attrs.get("name", "")
    role = attrs.get("sweforge.role", "?")
    t = fmt_epoch(str(event.get("time", "")))
    act = str(event.get("Action", "")).strip()

    if act.startswith("exec_create"):
        cmd = act[len("exec_create"):].lstrip(":").strip()
        kind = classify(cmd)
        if kind == "setup":
            counters["setup"] += 1
            if not args.all:
                return None
            tag = "[setup]"
        else:
            counters["agent"] += 1
            tag = f"[action:{kind}]"
        if args.no_commands or not cmd:
            return f"{t} {tag} env={env}"
        cmd = cmd if len(cmd) <= 100 else cmd[:97] + "..."
        return f"{t} {tag} env={env} {cmd}"

    if act == "create":
        counters["create"] += 1
        return f"{t} [env:create] role={role} env={env}"
    if act == "start":
        counters["start"] += 1
        return None
    if act == "die":
        counters["die"] += 1
        return f"{t} [env:die  ] env={env} exit={attrs.get('exitCode', '?')}"
    if act == "destroy":
        counters["destroy"] += 1
        return f"{t} [env:rm   ] env={env}"
    if act == "kill":
        # rm -f 会先 kill 再 die; die 里计数即可, 这里只提示
        return f"{t} [env:kill ] env={env}"
    return None


def snapshot(docker: str, port: int, counters: dict, cap: int = 15) -> str:
    lines = []
    ok = health_ok(port)
    _rc, out, _ = run(
        (docker, "ps", "--filter", "label=sweforge.managed=true",
         "--format", "{{.Names}}\t{{.Status}}"),
    )
    names = [ln for ln in out.splitlines() if ln.strip()]
    hdr = (
        f"[{now()}] ── 快照: health={'OK' if ok else 'DOWN'} | 运行容器={len(names)} "
        f"| 累计: create={counters['create']} start={counters['start']} "
        f"agent_exec={counters['agent']} setup_exec={counters['setup']} "
        f"die={counters['die']} destroy={counters['destroy']} ──"
    )
    lines.append(hdr)
    if not names:
        return "\n".join(lines)
    shown = names[:cap]
    for n in shown:
        lines.append(f"    {n}")
    if len(names) > cap:
        lines.append(f"    ... 共 {len(names)} 个")
    ctr_names = [n.split("\t")[0] for n in shown]
    _rc2, st, _ = run(
        (docker, "stats", "--no-stream", "--format",
         "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.PIDs}}", *ctr_names),
        timeout=15,
    )
    if _rc2 == 0:
        for sline in st.splitlines():
            if sline.strip():
                lines.append(f"      {sline}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Mac 侧 Docker 层监控（只读）")
    ap.add_argument("--health-port", type=int, default=8501,
                    help="Mac Environment Server 端口（AutoDL 实际连接的服务, 默认 8501）")
    ap.add_argument("--snapshot", type=int, default=10, help="资源快照间隔秒（默认 10）")
    ap.add_argument("--all", action="store_true", help="连 setup 内部 exec（git/tar/.git）也显示")
    ap.add_argument("--no-commands", action="store_true", help="不显示 exec 命令原文")
    ap.add_argument("--docker", default="docker")
    args = ap.parse_args()

    counters = {"create": 0, "start": 0, "agent": 0, "setup": 0, "die": 0, "destroy": 0}
    print(f"[{now()}] monitor 启动 | health=127.0.0.1:{args.health_port}/health "
          f"| snapshot={args.snapshot}s | docker={args.docker} | all={args.all}", flush=True)
    print(f"[{now()}] 说明: Docker 层监控, 看不到 HTTP 层 verify 的 verdict; "
          f"agent 动作以 exec 命令呈现。Ctrl+C 退出。", flush=True)

    q: queue.Queue[tuple[str, str]] = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=event_reader, args=(args.docker, q, stop), daemon=True).start()

    last_snap = time.monotonic()
    try:
        while True:
            try:
                item = q.get(timeout=args.snapshot)
            except queue.Empty:
                item = None
            if item is not None:
                kind, raw = item
                if kind == "MSG":
                    print(f"[{now()}] {raw}", flush=True)
                else:
                    line = fmt_event(raw, args, counters)
                    if line:
                        print(line, flush=True)
            if time.monotonic() - last_snap >= args.snapshot:
                last_snap = time.monotonic()
                print(snapshot(args.docker, args.health_port, counters), flush=True)
    except KeyboardInterrupt:
        stop.set()
        print(f"\n[{now()}] monitor 停止", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
