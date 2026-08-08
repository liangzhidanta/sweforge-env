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
