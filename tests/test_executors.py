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


def test_docker_exec_workdir_precedes_container_name(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        import subprocess as sp
        return sp.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("subprocess.run", fake_run)
    executor = DockerExecutor(image="img", container_name="c1", task_id="t1", env_id="e1")
    executor._started = True  # skip actual container start
    executor.run_argv(("echo", "hi"), cwd="sub")
    command = captured["command"]
    assert "--workdir" in command
    assert command.index("--workdir") < command.index("c1")


def test_docker_read_text_returns_full_file(monkeypatch):
    import subprocess as sp

    big = "x" * 30_000
    def fake_run(command, **kwargs):
        return sp.CompletedProcess(command, 0, big, "")
    monkeypatch.setattr("subprocess.run", fake_run)
    executor = DockerExecutor(image="img", container_name="c1", task_id="t1", env_id="e1",
                              max_output_chars=20_000)
    executor._started = True  # skip actual container start
    assert executor.read_text("some/file.txt") == big
