"""Workspace path policy: block absolute, escape, symlink, and protected paths."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class PathPolicy:
    def __init__(self, root: Path, protected_paths: Sequence[str] = ()) -> None:
        self.root = root.resolve()
        self.protected_paths = tuple(
            Path(path).as_posix().strip("/") for path in protected_paths if path
        )

    def resolve(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError("path must be relative to the task workspace")
        resolved = (self.root / candidate).resolve()
        try:
            relative = resolved.relative_to(self.root).as_posix()
        except ValueError as error:
            raise ValueError("path escapes the task workspace") from error
        for protected in self.protected_paths:
            if relative == protected or relative.startswith(protected + "/"):
                raise ValueError(f"path is protected: {protected}")
        return resolved
