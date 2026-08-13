from pathlib import Path

from sweforge.env_server.docker.backend import LocalDockerBackend
from sweforge.env_server.docker.executors import DockerExecutor
from sweforge.schemas.task import TaskEnvironment, TaskSpec


def test_task_environment_accepts_image_native_runtime_options():
    environment = TaskEnvironment(
        image="example/public-task:1",
        workspace="/testbed",
        runtime_user="0:0",
        seed_from_snapshot=False,
    )

    assert environment.workspace == "/testbed"
    assert environment.runtime_user == "0:0"
    assert environment.seed_from_snapshot is False


def test_docker_executor_uses_configured_workspace_and_user():
    executor = DockerExecutor(
        image="img",
        container_name="c1",
        task_id="t1",
        env_id="e1",
        workspace="/testbed",
        runtime_user="0:0",
    )

    command = executor.create_command()
    assert command[command.index("--workdir") + 1] == "/testbed"
    assert command[command.index("--user") + 1] == "0:0"


def test_native_image_task_does_not_seed_empty_snapshot(tmp_path: Path):
    task = TaskSpec(
        task_id="public-1",
        repo="public",
        base_commit="0" * 40,
        problem_statement="fix",
        environment=TaskEnvironment(
            image="example/public-task:1",
            workspace="/testbed",
            runtime_user="0:0",
            seed_from_snapshot=False,
        ),
    )
    backend = LocalDockerBackend(tmp_path, use_docker=True)
    executor = backend._make_executor(backend._bundle(task), "task")

    assert isinstance(executor, DockerExecutor)
    assert executor.workspace == "/testbed"
    assert executor.create_command()[executor.create_command().index("--user") + 1] == "0:0"
    assert executor._started is False

