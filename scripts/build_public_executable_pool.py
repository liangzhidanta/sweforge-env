#!/usr/bin/env python3
"""Execute R2E candidates in their native images and keep honest F2P/P2P tasks."""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sections(patch: str) -> list[str]:
    lines = patch.splitlines(keepends=True)
    # Native git diffs carry ``diff --git`` headers; R2E's reconstructed patch
    # starts directly at ``--- a/...``.  Splitting a native diff at ``---``
    # would incorrectly attach the next file's diff/index headers to the
    # previous section and produce a corrupt filtered patch.
    marker = "diff --git " if any(line.startswith("diff --git ") for line in lines) else "--- "
    starts = [index for index, line in enumerate(lines) if line.startswith(marker)]
    return ["".join(lines[start : starts[i + 1] if i + 1 < len(starts) else len(lines)])
            for i, start in enumerate(starts)]


def _new_path(section: str) -> str | None:
    for line in section.splitlines():
        if line.startswith("+++ b/"):
            return line[6:]
    return None


def _is_test_path(path: str | None) -> bool:
    if not path:
        return False
    parts = Path(path).parts
    name = Path(path).name
    return "tests" in parts or "test" in parts or name.startswith("test_") or name.endswith("_test.py")


def test_only_patch(patch: str) -> str:
    return "\n".join(section.rstrip("\n") for section in _sections(patch) if _is_test_path(_new_path(section))) + "\n"


def source_only_patch(patch: str) -> str:
    return "\n".join(section.rstrip("\n") for section in _sections(patch) if not _is_test_path(_new_path(section))) + "\n"


def _changed_new_lines(patch: str) -> dict[str, set[int]]:
    changed: dict[str, set[int]] = collections.defaultdict(set)
    for section in _sections(patch):
        path = _new_path(section)
        if not _is_test_path(path):
            continue
        new_line: int | None = None
        for line in section.splitlines():
            match = _HUNK.match(line)
            if match:
                new_line = int(match.group(1))
                continue
            if new_line is None or line.startswith(("--- ", "+++ ")):
                continue
            if line.startswith("+"):
                changed[path].add(new_line)
                new_line += 1
            elif line.startswith("-"):
                continue
            else:
                new_line += 1
    return changed


def _test_nodes(source: str, path: str) -> list[tuple[str, int, int]]:
    tree = ast.parse(source, filename=path)
    nodes: list[tuple[str, int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            nodes.append((f"{path}::{node.name}", node.lineno, node.end_lineno or node.lineno))
        if not isinstance(node, ast.ClassDef):
            continue
        for method in node.body:
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name.startswith("test_"):
                nodes.append((
                    f"{path}::{node.name}::{method.name}",
                    method.lineno,
                    method.end_lineno or method.lineno,
                ))
    return nodes


def changed_test_selectors(files: dict[str, str], patch: str) -> list[str]:
    changed = _changed_new_lines(patch)
    selectors: list[str] = []
    for path in sorted(changed):
        if path not in files:
            continue
        for selector, start, end in _test_nodes(files[path], path):
            if any(start <= line <= end for line in changed[path]):
                selectors.append(selector)
    return sorted(set(selectors))


def all_test_selectors(files: dict[str, str]) -> list[str]:
    return sorted({selector for path, source in files.items() for selector, _, _ in _test_nodes(source, path)})


def finalize_task(
    task: dict[str, Any], *, f2p: list[str], p2p: list[str], evidence: dict[str, Any]
) -> dict[str, Any]:
    task = json.loads(json.dumps(task))
    environment = task.setdefault("environment", {})
    environment.update({
        "workspace": "/testbed",
        "runtime_user": "0:0",
        "seed_from_snapshot": False,
        "setup_commands": [],
        "build_commands": [],
        "test_commands": {
            selector: ["python", "-m", "pytest", "-q", "--disable-warnings", selector]
            for selector in [*f2p, *p2p]
        },
    })
    task["fail_to_pass"] = [
        {"test_id": selector, "kind": "fail_to_pass", "timeout_seconds": 300, "notes": "test-only patch fails; full repair passes"}
        for selector in f2p
    ]
    task["pass_to_pass"] = [
        {"test_id": selector, "kind": "pass_to_pass", "timeout_seconds": 300, "notes": "passes in both executable states"}
        for selector in p2p
    ]
    metadata = task.setdefault("metadata", {})
    metadata.update({
        "executable_verified": True,
        "verification_protocol": "r2e-test-injection-base-vs-full-fix-v2",
        "container_workspace": "/testbed",
        "container_user": "0:0",
        "verification_evidence": evidence,
    })
    task["gold_patch"] = source_only_patch(str(task.get("gold_patch") or ""))
    return task


def write_bundle(bundle_root: Path, task: dict[str, Any], hidden_files: dict[str, str]) -> None:
    bundle = bundle_root / str(task["task_id"])
    (bundle / "repo").mkdir(parents=True, exist_ok=True)
    hidden_root = bundle / "private" / "hidden_tests"
    for relative, content in hidden_files.items():
        destination = hidden_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    manifest = {
        "task": task,
        "integrity_protected": sorted(hidden_files),
    }
    (bundle / "task_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class DockerRuntime:
    def __init__(self, binary: str = "docker", timeout: int = 180) -> None:
        self.binary = binary
        self.timeout = timeout

    def _run(self, command: list[str], *, input_text: str | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout or self.timeout,
        )

    def pull(self, image: str) -> None:
        errors: list[str] = []
        for attempt in range(1, 5):
            result = self._run([self.binary, "pull", "--platform", "linux/amd64", image], timeout=600)
            if result.returncode == 0:
                return
            errors.append(f"attempt {attempt}: {result.stderr[-300:]}")
            if attempt < 4:
                time.sleep(2 ** attempt)
        raise RuntimeError("docker pull failed after 4 attempts: " + " | ".join(errors))

    def create(self, image: str, task_id: str, role: str) -> str:
        name = f"sweforge-public-{role}-{task_id[:20]}-{uuid.uuid4().hex[:6]}"
        result = self._run([
            self.binary, "run", "--detach", "--rm", "--platform", "linux/amd64",
            "--network", "none", "--pids-limit", "1024", "--cpus", "4", "--memory", "8g",
            "--user", "0:0", "--workdir", "/testbed", "--name", name,
            "--entrypoint", "/bin/sh", image, "-lc", "sleep infinity",
        ])
        if result.returncode:
            raise RuntimeError(f"docker run failed: {result.stderr[-500:]}")
        return name

    def copy_text(self, container: str, text: str, remote_path: str) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(text)
            local = Path(handle.name)
        try:
            os.chmod(local, 0o644)
            result = self._run([self.binary, "cp", str(local), f"{container}:{remote_path}"])
            if result.returncode:
                raise RuntimeError(f"docker cp failed: {result.stderr[-500:]}")
        finally:
            local.unlink(missing_ok=True)

    def exec(self, container: str, command: str, *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        return self._run(
            [self.binary, "exec", "--user", "0:0", "--workdir", "/testbed", container, "/bin/sh", "-lc", command],
            timeout=timeout,
        )

    def read_file(self, container: str, path: str) -> str:
        result = self.exec(container, f"cat -- {json.dumps(path)}")
        if result.returncode:
            raise RuntimeError(f"cannot read {path}: {result.stderr[-500:]}")
        return result.stdout

    def remove_container(self, container: str | None) -> None:
        if container:
            self._run([self.binary, "rm", "-f", container], timeout=30)

    def remove_image(self, image: str) -> None:
        self._run([self.binary, "image", "rm", image], timeout=120)


def _pytest(container: str, selector: str, runtime: DockerRuntime, timeout: int) -> int:
    command = (
        "py=.venv/bin/python; test -x \"$py\" || py=python; "
        f"\"$py\" -m pytest -q --disable-warnings {json.dumps(selector)}"
    )
    return runtime.exec(container, command, timeout=timeout).returncode


def verify_candidate(
    task: dict[str, Any], runtime: DockerRuntime, *, timeout: int = 180
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, str]]:
    started = time.monotonic()
    image = str((task.get("environment") or {}).get("image") or "")
    patch = str(task.get("gold_patch") or "")
    tests_patch = test_only_patch(patch)
    evidence: dict[str, Any] = {"task_id": task.get("task_id"), "image": image}
    if not image or not tests_patch.strip():
        return None, {**evidence, "status": "rejected", "reason": "missing image or test patch"}, {}

    base = fixed = None
    try:
        runtime.pull(image)
        base = runtime.create(image, str(task["task_id"]), "base")
        fixed = runtime.create(image, str(task["task_id"]), "fix")
        runtime.copy_text(base, tests_patch, "/tmp/tests.patch")
        runtime.copy_text(fixed, patch, "/tmp/full.patch")
        apply_base = runtime.exec(base, "git apply /tmp/tests.patch")
        apply_fix = runtime.exec(fixed, "git apply /tmp/full.patch")
        if apply_base.returncode or apply_fix.returncode:
            reason = f"patch apply failed base={apply_base.returncode} fix={apply_fix.returncode}"
            return None, {**evidence, "status": "rejected", "reason": reason}, {}

        test_paths = sorted(_changed_new_lines(tests_patch))
        files = {path: runtime.read_file(base, path) for path in test_paths}
        changed = changed_test_selectors(files, tests_patch)
        if not changed:
            return None, {**evidence, "status": "rejected", "reason": "no changed Python test selector"}, {}

        f2p: list[str] = []
        f2p_attempts: list[dict[str, Any]] = []
        for selector in changed[:6]:
            base_rc = _pytest(base, selector, runtime, timeout)
            fix_rc = _pytest(fixed, selector, runtime, timeout)
            f2p_attempts.append({"selector": selector, "base_rc": base_rc, "fix_rc": fix_rc})
            if base_rc == 1 and fix_rc == 0:
                f2p.append(selector)
                break
        if not f2p:
            return None, {**evidence, "status": "rejected", "reason": "no test-only fail/full-fix pass selector", "f2p_attempts": f2p_attempts}, {}

        p2p: list[str] = []
        p2p_attempts: list[dict[str, Any]] = []
        for selector in all_test_selectors(files):
            if selector in changed:
                continue
            base_rc = _pytest(base, selector, runtime, timeout)
            fix_rc = _pytest(fixed, selector, runtime, timeout)
            p2p_attempts.append({"selector": selector, "base_rc": base_rc, "fix_rc": fix_rc})
            if base_rc == 0 and fix_rc == 0:
                p2p.append(selector)
                break
            if len(p2p_attempts) >= 8:
                break
        if not p2p:
            return None, {**evidence, "status": "rejected", "reason": "no stable P2P selector", "p2p_attempts": p2p_attempts}, {}

        evidence.update({
            "status": "verified",
            "base_f2p_rc": 1,
            "fix_f2p_rc": 0,
            "base_p2p_rc": 0,
            "fix_p2p_rc": 0,
            "f2p_attempts": f2p_attempts,
            "p2p_attempts": p2p_attempts,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        return finalize_task(task, f2p=f2p, p2p=p2p, evidence=evidence), evidence, files
    except (RuntimeError, subprocess.TimeoutExpired, SyntaxError, OSError) as exc:
        return None, {**evidence, "status": "rejected", "reason": f"{type(exc).__name__}: {exc}"}, {}
    finally:
        runtime.remove_container(base)
        runtime.remove_container(fixed)


def balanced_candidates(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_repo: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for task in sorted(tasks, key=lambda item: item["task_id"]):
        by_repo[str(task["repo"])].append(task)
    result: list[dict[str, Any]] = []
    offsets = {repo: 0 for repo in by_repo}
    while True:
        progressed = False
        for repo in sorted(by_repo):
            offset = offsets[repo]
            if offset < len(by_repo[repo]):
                result.append(by_repo[repo][offset])
                offsets[repo] += 1
                progressed = True
        if not progressed:
            return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--bundles-dir", required=True, type=Path)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--remove-images", action="store_true")
    parser.add_argument("--repo", action="append", default=[], help="only process these repo names")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    verified = read_jsonl(args.out) if args.out.exists() else []
    state = read_jsonl(args.state) if args.state.exists() else []
    done = {str(row["task_id"]) for row in state}
    accepted = {str(task["task_id"]) for task in verified}
    runtime = DockerRuntime(timeout=args.timeout)

    with args.out.open("a", encoding="utf-8") as out_handle, args.state.open("a", encoding="utf-8") as state_handle:
        candidates = read_jsonl(args.candidates)
        if args.repo:
            allowed = set(args.repo)
            candidates = [task for task in candidates if task.get("repo") in allowed]
        for task in balanced_candidates(candidates):
            if len(accepted) >= args.target:
                break
            if task["task_id"] in done:
                continue
            result, evidence, hidden_files = verify_candidate(task, runtime, timeout=args.timeout)
            state_handle.write(json.dumps(evidence, ensure_ascii=False) + "\n")
            state_handle.flush()
            done.add(str(task["task_id"]))
            if result is not None:
                write_bundle(args.bundles_dir, result, hidden_files)
                out_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_handle.flush()
                accepted.add(str(task["task_id"]))
            print(
                f"[{len(done):03d}] {task['task_id']} {evidence['status']} "
                f"accepted={len(accepted)}/{args.target} reason={evidence.get('reason', '-')}",
                flush=True,
            )
            if args.remove_images:
                runtime.remove_image(str(task["environment"]["image"]))

    if len(accepted) < args.target:
        raise SystemExit(f"verified only {len(accepted)} tasks; target={args.target}")
    print(f"PUBLIC_EXECUTABLE_READY={len(accepted)} out={args.out}")


if __name__ == "__main__":
    main()
