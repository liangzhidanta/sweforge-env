#!/usr/bin/env python3
"""Build the quota-controlled Mixed RL Task Pool used by formal GRPO."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_QUOTAS = {
    "PUBLIC_EXECUTABLE": 100,
    "AST_MUTATION": 50,
    "REPAIR_REVERSAL": 50,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source(task: dict[str, Any]) -> str:
    return str((task.get("metadata") or {}).get("task_source") or "")


def _round_robin(groups: dict[str, list[dict[str, Any]]]) -> Iterable[dict[str, Any]]:
    offsets = {key: 0 for key in groups}
    while True:
        progressed = False
        for key in sorted(groups):
            offset = offsets[key]
            if offset >= len(groups[key]):
                continue
            yield groups[key][offset]
            offsets[key] += 1
            progressed = True
        if not progressed:
            return


def select_quota(tasks: list[dict[str, Any]], source: str, quota: int) -> list[dict[str, Any]]:
    """Select exactly ``quota`` tasks, deterministically balancing repositories."""
    if quota < 0:
        raise ValueError("quota must be non-negative")
    eligible = [task for task in tasks if _source(task) == source]
    if len(eligible) < quota:
        raise ValueError(f"{source}: need {quota} tasks, found {len(eligible)}")
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for task in sorted(eligible, key=lambda item: item["task_id"]):
        groups[str(task["repo"])].append(task)
    return list(_round_robin(groups))[:quota]


def _portable(task: dict[str, Any]) -> dict[str, Any]:
    task = json.loads(json.dumps(task))
    environment = task.setdefault("environment", {})
    setup = environment.get("setup_commands") or []
    normalized_setup: list[str] = []
    for command in setup:
        if " -m pip " in command and not command.startswith("python -m pip "):
            command = "python -m pip " + command.split(" -m pip ", 1)[1]
        normalized_setup.append(command)
    environment["setup_commands"] = normalized_setup
    for command in (environment.get("test_commands") or {}).values():
        if len(command) >= 3 and command[1:3] == ["-m", "pytest"]:
            command[0] = "python"
    return task


def _pool_view(task: dict[str, Any]) -> dict[str, Any]:
    clean = _portable(task)
    clean["gold_patch"] = None
    clean["mutation"] = None
    return clean


def _manifest(tasks: list[dict[str, Any]], quotas: dict[str, int]) -> dict[str, Any]:
    sources = collections.Counter(_source(task) for task in tasks)
    repos = collections.Counter(str(task["repo"]) for task in tasks)
    generators = collections.Counter(
        str((task.get("metadata") or {}).get("generator_type") or "unknown")
        for task in tasks
    )
    mutation_kinds = collections.Counter(
        str((task.get("mutation") or {}).get("kind") or "none")
        for task in tasks
    )
    total = len(tasks)
    return {
        "version": "mixed-rl-task-pool-v2",
        "total": total,
        "quotas": dict(sorted(quotas.items())),
        "source_counts": dict(sorted(sources.items())),
        "source_ratios": {
            source: round(count / total, 4) if total else 0.0
            for source, count in sorted(sources.items())
        },
        "repo_counts": dict(sorted(repos.items())),
        "generator_counts": dict(sorted(generators.items())),
        "mutation_kinds": dict(sorted(mutation_kinds.items())),
        "f2p_tests": sum(len(task.get("fail_to_pass") or []) for task in tasks),
        "p2p_tests": sum(len(task.get("pass_to_pass") or []) for task in tasks),
    }


def build_mixed_pool(
    public: list[dict[str, Any]],
    ast: list[dict[str, Any]],
    reversal: list[dict[str, Any]],
    *,
    quotas: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)
    selected = [
        *select_quota(public, "PUBLIC_EXECUTABLE", quotas["PUBLIC_EXECUTABLE"]),
        *select_quota(ast, "AST_MUTATION", quotas["AST_MUTATION"]),
        *select_quota(reversal, "REPAIR_REVERSAL", quotas["REPAIR_REVERSAL"]),
    ]
    full = [_portable(task) for task in selected]
    ids = [task["task_id"] for task in full]
    if len(ids) != len(set(ids)):
        duplicates = sorted(task_id for task_id, count in collections.Counter(ids).items() if count > 1)
        raise ValueError(f"duplicate task_id: {duplicates}")
    for task in full:
        if not task.get("fail_to_pass"):
            raise ValueError(f"task has no F2P tests: {task['task_id']}")
        expected = (task.get("metadata") or {}).get("task_source")
        if expected not in quotas:
            raise ValueError(f"unknown task_source={expected!r}: {task['task_id']}")
    pool = [_pool_view(task) for task in full]
    return full, pool, _manifest(full, quotas)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--ast", required=True, type=Path)
    parser.add_argument("--reversal", required=True, type=Path)
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--public-quota", type=int, default=100)
    parser.add_argument("--ast-quota", type=int, default=50)
    parser.add_argument("--reversal-quota", type=int, default=50)
    args = parser.parse_args()

    quotas = {
        "PUBLIC_EXECUTABLE": args.public_quota,
        "AST_MUTATION": args.ast_quota,
        "REPAIR_REVERSAL": args.reversal_quota,
    }
    full, pool, manifest = build_mixed_pool(
        read_jsonl(args.public),
        read_jsonl(args.ast),
        read_jsonl(args.reversal),
        quotas=quotas,
    )
    train_dir = args.out_dir / "train"
    _write_jsonl(train_dir / "full.jsonl", full)
    _write_jsonl(train_dir / "pool.jsonl", pool)

    if args.eval:
        evaluate = [_portable(task) for task in read_jsonl(args.eval)]
        train_repos = {task["repo"] for task in full}
        eval_repos = {task["repo"] for task in evaluate}
        overlap = sorted(train_repos & eval_repos)
        if overlap:
            raise ValueError(f"train/eval repo leakage: {overlap}")
        _write_jsonl(args.out_dir / "eval" / "full.jsonl", evaluate)
        _write_jsonl(args.out_dir / "eval" / "pool.jsonl", [_pool_view(task) for task in evaluate])
        manifest["eval"] = {"count": len(evaluate), "repos": sorted(eval_repos)}
        manifest["train_eval_repo_overlap"] = overlap

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

