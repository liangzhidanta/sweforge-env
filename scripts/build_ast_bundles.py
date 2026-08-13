#!/usr/bin/env python3
"""Select verified AST tasks and replay their exact mutation commits into bundles."""

from __future__ import annotations

import argparse
import collections
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from sweforge.data.factory.factory import replay_mutation_commit
from sweforge.data.factory.reversal import git_cmd, rebuild_seed_repo


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_ast_tasks(tasks: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    eligible = [task for task in tasks if (task.get("metadata") or {}).get("task_source") == "AST_MUTATION"]
    if len(eligible) < quota:
        raise ValueError(f"AST_MUTATION: need {quota} tasks, found {len(eligible)}")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for task in sorted(eligible, key=lambda item: item["task_id"]):
        groups[(str(task["repo"]), str((task.get("mutation") or {}).get("kind") or "unknown"))].append(task)
    offsets = {key: 0 for key in groups}
    selected: list[dict[str, Any]] = []
    while len(selected) < quota:
        for key in sorted(groups):
            offset = offsets[key]
            if offset >= len(groups[key]):
                continue
            selected.append(groups[key][offset])
            offsets[key] += 1
            if len(selected) == quota:
                break
    return selected


def _archive(repo: Path, commit: str, destination: Path) -> None:
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", commit],
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        archive.extractall(destination, filter="data")


def _task_for_runtime(task: dict[str, Any], image: str) -> dict[str, Any]:
    task = json.loads(json.dumps(task))
    environment = task.setdefault("environment", {})
    environment.update({
        "image": image,
        "workspace": "/workspace",
        "runtime_user": "1000:1000",
        "seed_from_snapshot": True,
    })
    metadata = task.setdefault("metadata", {})
    metadata["bundle_replayed"] = True
    metadata["bundle_protocol"] = "exact-mutation-replay-v2"
    return task


def build_bundles(
    tasks: list[dict[str, Any]], sources: dict[str, Path], *,
    work_root: Path, bundle_root: Path, image: str, target: int | None = None,
) -> list[dict[str, Any]]:
    work_root.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)
    work_repos: dict[str, tuple[Path, str]] = {}
    for repo_name in sorted({str(task["repo"]) for task in tasks}):
        if repo_name not in sources:
            raise ValueError(f"missing source repo: {repo_name}")
        source = sources[repo_name]
        work = work_root / repo_name
        if work.exists():
            shutil.rmtree(work)
        work, fix_commit = rebuild_seed_repo(source, work)
        work_repos[repo_name] = (work, fix_commit)

    output: list[dict[str, Any]] = []
    for index, raw_task in enumerate(tasks, start=1):
        if target is not None and len(output) >= target:
            break
        task = _task_for_runtime(raw_task, image)
        mutation = task.get("mutation") or {}
        work, fix_commit = work_repos[str(task["repo"])]
        replayed = replay_mutation_commit(
            work,
            fix_commit,
            str(mutation["kind"]),
            int(mutation["recipe_seed"]),
            file=mutation.get("file"),
            location=mutation.get("location"),
        )
        if replayed != task["base_commit"]:
            print(
                f"[skip] replay hash mismatch {task['task_id']}: "
                f"expected={task['base_commit']} got={replayed}",
                flush=True,
            )
            continue
        bundle = bundle_root / str(task["task_id"])
        if bundle.exists():
            shutil.rmtree(bundle)
        _archive(work, replayed, bundle / "repo")
        (bundle / "private" / "hidden_tests").mkdir(parents=True, exist_ok=True)
        protected = sorted({
            spec["test_id"].split("::", 1)[0]
            for spec in [*(task.get("fail_to_pass") or []), *(task.get("pass_to_pass") or [])]
        })
        (bundle / "task_manifest.json").write_text(
            json.dumps({"task": task, "integrity_protected": protected}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output.append(task)
        print(f"[{len(output):03d}/{target or len(tasks):03d}] {task['task_id']} replay={replayed[:12]}", flush=True)
    if target is not None and len(output) < target:
        raise ValueError(f"AST bundle shortage: need {target}, built {len(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True, help="NAME=/absolute/repo/path")
    parser.add_argument("--quota", type=int, default=50)
    parser.add_argument("--image", default="sweforge-repair:py311")
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--bundles-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    sources: dict[str, Path] = {}
    for value in args.source:
        name, raw_path = value.split("=", 1)
        sources[name] = Path(raw_path).resolve()
    all_tasks = read_jsonl(args.full)
    selected = select_ast_tasks(all_tasks, len(all_tasks))
    built = build_bundles(
        selected,
        sources,
        work_root=args.work_root,
        bundle_root=args.bundles_dir,
        image=args.image,
        target=args.quota,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for task in built:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"AST_MUTATION_READY={len(built)} out={args.out}")


if __name__ == "__main__":
    main()
