import pytest

from sweforge.env_server.docker.path_policy import PathPolicy


def test_relative_ok(tmp_path):
    policy = PathPolicy(tmp_path)
    assert policy.resolve("a.txt") == (tmp_path / "a.txt").resolve()


def test_absolute_rejected(tmp_path):
    policy = PathPolicy(tmp_path)
    with pytest.raises(ValueError, match="relative"):
        policy.resolve(str(tmp_path / "x.txt"))


def test_escape_rejected(tmp_path):
    policy = PathPolicy(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        policy.resolve("../secret.txt")


def test_protected_rejected(tmp_path):
    (tmp_path / ".git").mkdir()
    policy = PathPolicy(tmp_path, protected_paths=(".git",))
    with pytest.raises(ValueError, match="protected"):
        policy.resolve(".git/config")


def test_symlink_escape_rejected(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    (tmp_path / "link").symlink_to(outside)
    policy = PathPolicy(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        policy.resolve("link")


def test_symlink_dir_with_trailing_components_rejected(tmp_path):
    outside_dir = tmp_path.parent / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("secret")
    (tmp_path / "link").symlink_to(outside_dir, target_is_directory=True)
    policy = PathPolicy(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        policy.resolve("link/secret.txt")


def test_symlink_inside_root_allowed(tmp_path):
    inside = tmp_path / "real.txt"
    inside.write_text("ok")
    (tmp_path / "alias").symlink_to(inside)
    policy = PathPolicy(tmp_path)
    assert policy.resolve("alias") == inside.resolve()


def test_protected_exact_and_lookalike(tmp_path):
    policy = PathPolicy(tmp_path, protected_paths=(".git",))
    with pytest.raises(ValueError, match="protected"):
        policy.resolve(".git")
    assert policy.resolve(".git.other") == (tmp_path / ".git.other").resolve()
