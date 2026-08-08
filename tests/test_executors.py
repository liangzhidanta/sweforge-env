from sweforge.env_server.docker.executors import DockerExecutor, LocalExecutor


def test_run_argv(tmp_path):
    executor = LocalExecutor(tmp_path)
    result = executor.run_argv(("echo", "hi"))
    assert result.exit_code == 0 and result.stdout.strip() == "hi"


def test_run_shell(tmp_path):
    executor = LocalExecutor(tmp_path)
    result = executor.run_shell("echo hello")
    assert result.exit_code == 0 and result.stdout.strip() == "hello"


def test_run_timeout(tmp_path):
    executor = LocalExecutor(tmp_path)
    result = executor.run_shell("sleep 5", timeout=0.2)
    assert result.timed_out and result.exit_code == 124


def test_output_truncation(tmp_path):
    executor = LocalExecutor(tmp_path, max_output_chars=10)
    result = executor.run_shell("echo 0123456789abcdef")
    assert result.truncated and len(result.stdout) <= 10


def test_read_write_text(tmp_path):
    executor = LocalExecutor(tmp_path)
    executor.write_text("a/b.txt", "hi")
    assert executor.read_text("a/b.txt") == "hi"


def test_from_snapshot(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "f.txt").write_text("x")
    executor = LocalExecutor.from_snapshot(source)
    try:
        assert (executor.root / "f.txt").read_text() == "x"
    finally:
        executor.close()


def test_docker_executor_create_command_has_security_flags():
    executor = DockerExecutor(image="img", container_name="c1", task_id="t1", env_id="e1")
    command = executor.create_command()
    assert "--network" in command and "none" in command
    assert "--pids-limit" in command
    assert "--user" in command
    assert "--label=sweforge.managed=true" in command
    assert "--label=sweforge.task_id=t1" in command
