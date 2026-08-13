#!/usr/bin/env python3
"""Build executable Repair Reversal tasks from real repository history."""

from __future__ import annotations

import argparse
import collections
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_public_executable_pool import (  # noqa: E402
    _changed_new_lines,
    _test_nodes,
    all_test_selectors,
    changed_test_selectors,
    source_only_patch,
    test_only_patch,
)


_FIX_RE = re.compile(r"\b(fix(?:e[ds])?|bug|regress(?:ion)?|incorrect|error|issue)\b", re.I)


def fix_like(subject: str) -> bool:
    return bool(_FIX_RE.search(subject))


def _git(repo: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=text,
        check=False,
        timeout=180,
    )


def _is_test_path(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    return "tests" in parts or "test" in parts or name.startswith("test_") or name.endswith("_test.py")


def candidate_commits(repo: Path) -> list[dict[str, Any]]:
    log = _git(repo, "log", "--all", "--no-merges", "--format=%H%x09%P%x09%ct%x09%s")
    if log.returncode:
        raise RuntimeError(log.stderr)
    candidates: list[dict[str, Any]] = []
    for line in log.stdout.splitlines():
        commit, parents, timestamp, subject = line.split("\t", 3)
        parent_list = parents.split()
        if len(parent_list) != 1:
            continue
        changed = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
        if changed.returncode:
            continue
        paths = [path for path in changed.stdout.splitlines() if path.endswith(".py")]
        has_test = any(_is_test_path(path) for path in paths)
        has_source = any(not _is_test_path(path) for path in paths)
        if has_test and has_source:
            candidates.append({
                "commit": commit,
                "parent": parent_list[0],
                "timestamp": int(timestamp),
                "subject": subject,
                "fix_like": fix_like(subject),
            })
    return sorted(candidates, key=lambda item: (not item["fix_like"], -item["timestamp"], item["commit"]))


def _archive(repo: Path, commit: str, destination: Path) -> None:
    result = _git(repo, "archive", "--format=tar", commit, text=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        archive.extractall(destination, filter="data")


def _apply_patch(root: Path, patch: str) -> None:
    result = subprocess.run(
        ["git", "apply", "--unsafe-paths", "-p1", "-"],
        cwd=root,
        input=patch,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-500:])


def snapshot_test_selectors(root: Path, preferred_files: dict[str, str], limit: int = 300) -> list[str]:
    """Collect stable P2P candidates beyond the file changed by the fix.

    A real fix may add the only test in its file.  Restricting P2P discovery to
    that file rejects an otherwise valid task, so we scan the repository while
    still testing every selected node in both parent and fix states.
    """
    selectors = all_test_selectors(preferred_files)
    seen = set(selectors)
    excluded = {".git", ".tox", ".venv", "venv", "build", "dist", "__pycache__"}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts) or not _is_test_path(relative.as_posix()):
            continue
        if relative.as_posix() in preferred_files:
            continue
        try:
            source = path.read_text(encoding="utf-8")
            nodes = _test_nodes(source, relative.as_posix())
        except (OSError, UnicodeError, SyntaxError):
            continue
        for selector, _, _ in nodes:
            if selector not in seen:
                selectors.append(selector)
                seen.add(selector)
                if len(selectors) >= limit:
                    return selectors
    return selectors


class SnapshotRuntime:
    def __init__(self, image: str, timeout: int = 180, docker: str = "docker") -> None:
        self.image = image
        self.timeout = timeout
        self.docker = docker

    def create(self, snapshot: Path, task_id: str, role: str) -> str:
        name = f"sweforge-reversal-{role}-{task_id[:16]}-{uuid.uuid4().hex[:6]}"
        result = subprocess.run([
            self.docker, "run", "--detach", "--rm", "--network", "none",
            "--pids-limit", "1024", "--cpus", "4", "--memory", "8g",
            "--user", "0:0", "--workdir", "/workspace",
            "--name", name, "--entrypoint", "/bin/sh", self.image, "-lc", "sleep infinity",
        ], capture_output=True, text=True, check=False, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr[-500:])
        with tempfile.NamedTemporaryFile(prefix="sweforge-reversal-seed-", suffix=".tar", delete=False) as handle:
            archive_path = Path(handle.name)
        try:
            with tarfile.open(archive_path, "w") as archive:
                for path in sorted(snapshot.rglob("*")):
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                        archive.add(path, arcname=path.relative_to(snapshot).as_posix())
            copy = subprocess.run(
                [self.docker, "cp", str(archive_path), f"{name}:/tmp/repo.tar"],
                capture_output=True, text=True, check=False, timeout=120,
            )
            if copy.returncode:
                raise RuntimeError(copy.stderr[-500:])
            extract = subprocess.run(
                [self.docker, "exec", "--user", "0:0", name, "tar", "-xf", "/tmp/repo.tar", "-C", "/workspace"],
                capture_output=True, text=True, check=False, timeout=120,
            )
            if extract.returncode:
                raise RuntimeError(extract.stderr[-500:])
        except Exception:
            self.remove(name)
            raise
        finally:
            archive_path.unlink(missing_ok=True)
        return name

    def pytest(self, container: str, selector: str) -> int:
        command = (
            "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src "
            f"python -m pytest -q --disable-warnings {json.dumps(selector)}"
        )
        result = subprocess.run(
            [self.docker, "exec", "--user", "0:0", "--workdir", "/workspace", container, "/bin/sh", "-lc", command],
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout,
        )
        return result.returncode

    def remove(self, container: str | None) -> None:
        if container:
            subprocess.run([self.docker, "rm", "-f", container], capture_output=True, check=False, timeout=30)


def finalize_reversal(
    *, repo: str, parent: str, fix_commit: str, subject: str, patch: str,
    image: str, f2p: list[str], p2p: list[str], evidence: dict[str, Any],
) -> dict[str, Any]:
    task_id = f"reversal-{repo}-{fix_commit[:12]}"
    commands = {
        selector: ["python", "-m", "pytest", "-q", "--disable-warnings", selector]
        for selector in [*f2p, *p2p]
    }
    return {
        "task_id": task_id,
        "repo": repo,
        "base_commit": parent,
        "problem_statement": f"{subject}。请定位并修复该回归，保持既有行为不受影响；不要修改测试文件。",
        "environment": {
            "image": image,
            "workspace": "/workspace",
            "runtime_user": "1000:1000",
            "seed_from_snapshot": True,
            "setup_commands": [],
            "build_commands": [],
            "test_commands": commands,
        },
        "fail_to_pass": [
            {"test_id": selector, "kind": "fail_to_pass", "timeout_seconds": 300, "notes": "fix test injected into parent fails; fix commit passes"}
            for selector in f2p
        ],
        "pass_to_pass": [
            {"test_id": selector, "kind": "pass_to_pass", "timeout_seconds": 300, "notes": "passes in parent and fix states"}
            for selector in p2p
        ],
        "gold_patch": source_only_patch(patch),
        "mutation": {
            "kind": "real_repair_reversal",
            "source_commit": fix_commit,
            "source_pr": None,
            "file": None,
            "location": None,
            "recipe_seed": None,
            "entity": None,
            "reference_state": None,
            "target_index": None,
        },
        "difficulty": None,
        "metadata": {
            "protocol_version": "canonical-v1",
            "source": "sweforge-history-reversal",
            "source_record_id": f"reversal:{repo}:{fix_commit}",
            "source_version": "history-reversal-v2",
            "task_source": "REPAIR_REVERSAL",
            "repo_id": repo,
            "generator_type": "repair:git-history-reversal",
            "generator_version": "git-history-reversal-v2",
            "commit_subject": subject,
            "commit_message_fix_like": fix_like(subject),
            "executable_verified": True,
            "verification_protocol": "test-injection-parent-vs-fix-v2",
            "verification_evidence": evidence,
        },
    }


def _write_bundle(bundle_root: Path, task: dict[str, Any], repo_snapshot: Path, hidden_files: dict[str, str]) -> None:
    bundle = bundle_root / task["task_id"]
    shutil.copytree(repo_snapshot, bundle / "repo", dirs_exist_ok=False)
    hidden_root = bundle / "private" / "hidden_tests"
    for relative, content in hidden_files.items():
        destination = hidden_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    (bundle / "task_manifest.json").write_text(
        json.dumps({"task": task, "integrity_protected": sorted(hidden_files)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_commit(
    repo_path: Path, repo_name: str, candidate: dict[str, Any], runtime: SnapshotRuntime,
    bundle_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    started = time.monotonic()
    commit = candidate["commit"]
    parent = candidate["parent"]
    evidence: dict[str, Any] = {"repo": repo_name, "fix_commit": commit, "parent": parent}
    patch_result = _git(repo_path, "diff", "--binary", parent, commit)
    if patch_result.returncode:
        return None, {**evidence, "status": "rejected", "reason": "git diff failed"}
    patch = patch_result.stdout
    tests_patch = test_only_patch(patch)
    if not source_only_patch(patch).strip() or not tests_patch.strip():
        return None, {**evidence, "status": "rejected", "reason": "missing source or test patch"}

    with tempfile.TemporaryDirectory(prefix="sweforge-reversal-") as temp:
        root = Path(temp)
        parent_clean = root / "parent_clean"
        parent_tests = root / "parent_tests"
        fixed = root / "fixed"
        base_container = fix_container = None
        try:
            _archive(repo_path, parent, parent_clean)
            shutil.copytree(parent_clean, parent_tests)
            _apply_patch(parent_tests, tests_patch)
            _archive(repo_path, commit, fixed)
            paths = sorted(_changed_new_lines(tests_patch))
            hidden_files = {path: (parent_tests / path).read_text(encoding="utf-8") for path in paths}
            changed = changed_test_selectors(hidden_files, tests_patch)
            if not changed:
                return None, {**evidence, "status": "rejected", "reason": "no changed test selector"}

            task_id = f"reversal-{repo_name}-{commit[:12]}"
            base_container = runtime.create(parent_tests, task_id, "base")
            fix_container = runtime.create(fixed, task_id, "fix")
            f2p: list[str] = []
            f2p_attempts: list[dict[str, Any]] = []
            for selector in changed[:6]:
                base_rc = runtime.pytest(base_container, selector)
                fix_rc = runtime.pytest(fix_container, selector)
                f2p_attempts.append({"selector": selector, "base_rc": base_rc, "fix_rc": fix_rc})
                if base_rc == 1 and fix_rc == 0:
                    f2p.append(selector)
                    break
            if not f2p:
                return None, {**evidence, "status": "rejected", "reason": "no F2P", "f2p_attempts": f2p_attempts}

            p2p: list[str] = []
            p2p_attempts: list[dict[str, Any]] = []
            for selector in snapshot_test_selectors(parent_tests, hidden_files):
                if selector in changed:
                    continue
                base_rc = runtime.pytest(base_container, selector)
                fix_rc = runtime.pytest(fix_container, selector)
                p2p_attempts.append({"selector": selector, "base_rc": base_rc, "fix_rc": fix_rc})
                if base_rc == 0 and fix_rc == 0:
                    p2p.append(selector)
                    break
                if len(p2p_attempts) >= 8:
                    break
            if not p2p:
                return None, {**evidence, "status": "rejected", "reason": "no P2P", "p2p_attempts": p2p_attempts}

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
            task = finalize_reversal(
                repo=repo_name, parent=parent, fix_commit=commit,
                subject=candidate["subject"], patch=patch, image=runtime.image,
                f2p=f2p, p2p=p2p, evidence=evidence,
            )
            _write_bundle(bundle_root, task, parent_clean, hidden_files)
            return task, evidence
        except (RuntimeError, subprocess.TimeoutExpired, OSError, UnicodeError, SyntaxError) as exc:
            return None, {**evidence, "status": "rejected", "reason": f"{type(exc).__name__}: {exc}"}
        finally:
            runtime.remove(base_container)
            runtime.remove(fix_container)


def round_robin_candidates(repos: dict[str, Path]) -> list[tuple[str, Path, dict[str, Any]]]:
    groups = {name: candidate_commits(path) for name, path in repos.items()}
    offsets = {name: 0 for name in groups}
    result: list[tuple[str, Path, dict[str, Any]]] = []
    while True:
        progressed = False
        for name in sorted(groups):
            offset = offsets[name]
            if offset < len(groups[name]):
                result.append((name, repos[name], groups[name][offset]))
                offsets[name] += 1
                progressed = True
        if not progressed:
            return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", required=True, help="NAME=/absolute/repo/path")
    parser.add_argument("--image", default="sweforge-repair:py311")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--bundles-dir", required=True, type=Path)
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    repos: dict[str, Path] = {}
    for value in args.repo:
        name, raw_path = value.split("=", 1)
        repos[name] = Path(raw_path).resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    accepted_rows = _read_jsonl(args.out)
    state_rows = _read_jsonl(args.state)
    accepted = {row["task_id"] for row in accepted_rows}
    done = {(row["repo"], row["fix_commit"]) for row in state_rows}
    runtime = SnapshotRuntime(args.image, timeout=args.timeout)

    with args.out.open("a", encoding="utf-8") as out_handle, args.state.open("a", encoding="utf-8") as state_handle:
        for repo_name, repo_path, candidate in round_robin_candidates(repos):
            if len(accepted) >= args.target:
                break
            key = (repo_name, candidate["commit"])
            if key in done:
                continue
            task, evidence = verify_commit(repo_path, repo_name, candidate, runtime, args.bundles_dir)
            state_handle.write(json.dumps(evidence, ensure_ascii=False) + "\n")
            state_handle.flush()
            done.add(key)
            if task is not None:
                out_handle.write(json.dumps(task, ensure_ascii=False) + "\n")
                out_handle.flush()
                accepted.add(task["task_id"])
            print(
                f"[{len(done):03d}] {repo_name}:{candidate['commit'][:10]} "
                f"{evidence['status']} accepted={len(accepted)}/{args.target} "
                f"reason={evidence.get('reason', '-')}",
                flush=True,
            )
    if len(accepted) < args.target:
        raise SystemExit(f"verified only {len(accepted)} tasks; target={args.target}")
    print(f"REPAIR_REVERSAL_READY={len(accepted)} out={args.out}")


if __name__ == "__main__":
    main()
