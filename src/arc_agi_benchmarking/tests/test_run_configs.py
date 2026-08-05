import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cli import run_all, run_configs
from arc_agi_benchmarking.utils.concurrency_limiter import ProviderConcurrencyLimiter


@pytest.fixture(autouse=True)
def clear_rate_limiter_cache():
    run_all.PROVIDER_RATE_LIMITERS.clear()
    run_all.PROVIDER_CONCURRENCY_LIMITERS.clear()
    run_all.PROVIDER_CIRCUIT_BREAKERS.clear()
    yield
    run_all.PROVIDER_RATE_LIMITERS.clear()
    run_all.PROVIDER_CONCURRENCY_LIMITERS.clear()
    run_all.PROVIDER_CIRCUIT_BREAKERS.clear()


def test_provider_concurrency_limiter_is_optional():
    assert run_all.get_or_create_concurrency_limiter("xai", None) is None


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "8"])
def test_provider_concurrency_limiter_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="max_concurrency"):
        run_all.get_or_create_concurrency_limiter("openai", value)


@pytest.mark.asyncio
async def test_provider_concurrency_limiter_shares_slots_between_instances(tmp_path):
    first = ProviderConcurrencyLimiter(
        "openai", max_concurrency=2, lock_root=tmp_path, poll_interval=0.01
    )
    second = ProviderConcurrencyLimiter(
        "openai", max_concurrency=2, lock_root=tmp_path, poll_interval=0.01
    )
    release = asyncio.Event()
    two_entered = asyncio.Event()
    entered = 0

    async def hold_slot(limiter):
        nonlocal entered
        async with limiter.slot():
            entered += 1
            if entered == 2:
                two_entered.set()
            await release.wait()

    tasks = [
        asyncio.create_task(hold_slot(first)),
        asyncio.create_task(hold_slot(second)),
        asyncio.create_task(hold_slot(first)),
    ]

    await asyncio.wait_for(two_entered.wait(), timeout=1)
    await asyncio.sleep(0.05)
    assert entered == 2

    release.set()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
    assert entered == 3


def test_provider_rate_limit_is_divided_without_changing_period():
    limiter = run_all.get_or_create_rate_limiter(
        "xai",
        {"xai": {"rate": 300, "period": 1}},
        rate_limit_divisor=3,
    )

    assert limiter._rate == pytest.approx(100)
    assert limiter._capacity == pytest.approx(100)


def test_model_rate_limit_keeps_precedence_and_is_divided():
    model_config = SimpleNamespace(
        name="model-limited",
        kwargs={"rate_limit": {"rate": 90, "period": 30}},
    )

    limiter = run_all.get_or_create_rate_limiter(
        "xai",
        {"xai": {"rate": 300, "period": 1}},
        model_config=model_config,
        rate_limit_divisor=3,
    )

    # 90 requests / 30 seconds / 3 configs = 1 request per second.
    assert limiter._rate == pytest.approx(1)
    assert limiter._capacity == pytest.approx(1)


def test_single_config_keeps_full_rate_limit():
    limiter = run_all.get_or_create_rate_limiter(
        "xai",
        {"xai": {"rate": 300, "period": 60}},
    )

    assert limiter._rate == pytest.approx(5)
    assert limiter._capacity == pytest.approx(5)


def test_rate_limit_divisor_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        run_all.get_or_create_rate_limiter(
            "xai",
            {"xai": {"rate": 300, "period": 60}},
            rate_limit_divisor=0,
        )


@pytest.mark.asyncio
async def test_max_tasks_per_run_limits_sorted_unsubmitted_tasks(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for task_id in ("task-d", "task-b", "task-a", "task-c"):
        (data_dir / f"{task_id}.json").write_text("{}")

    model_config = SimpleNamespace(
        provider="task-limit-provider",
        name="task-limit-config",
        kwargs={},
    )
    run_wrapper = AsyncMock(return_value=True)

    with (
        patch.object(run_all, "get_model_config", return_value=model_config),
        patch.object(
            run_all,
            "read_provider_rate_limits",
            return_value={"task-limit-provider": {"rate": 100, "period": 60}},
        ),
        patch.object(
            run_all,
            "submission_exists",
            side_effect=lambda _directory, task_id: task_id == "task-a",
        ),
        patch.object(
            run_all,
            "submission_is_complete",
            side_effect=lambda _directory, task_id, _pairs, _attempts: (
                task_id == "task-a"
            ),
        ),
        patch.object(run_all, "get_task_test_pair_count", return_value=1),
        patch.object(run_all, "run_single_test_wrapper", run_wrapper),
    ):
        exit_code = await run_all.main(
            task_list_file=None,
            config_to_test="task-limit-config",
            data_dir=str(data_dir),
            save_submission_dir=str(tmp_path / "submissions"),
            overwrite_submission=False,
            print_submission=False,
            num_attempts=1,
            retry_attempts=1,
            logs_base_dir=tmp_path / "logs",
            max_tasks_per_run=2,
        )

    scheduled_task_ids = [call.args[1] for call in run_wrapper.await_args_list]
    assert exit_code == 0
    assert scheduled_task_ids == ["task-b", "task-c"]


@pytest.mark.asyncio
async def test_task_list_deduplicates_ids_before_scheduling(tmp_path):
    task_list = tmp_path / "tasks.txt"
    task_list.write_text("task-b\ntask-a\ntask-b\ntask-c\ntask-a\n")
    model_config = SimpleNamespace(
        provider="dedupe-provider",
        name="dedupe-config",
        kwargs={},
    )
    run_wrapper = AsyncMock(return_value=True)

    with (
        patch.object(run_all, "get_model_config", return_value=model_config),
        patch.object(
            run_all,
            "read_provider_rate_limits",
            return_value={"dedupe-provider": {"rate": 100, "period": 60}},
        ),
        patch.object(run_all, "submission_exists", return_value=False),
        patch.object(run_all, "run_single_test_wrapper", run_wrapper),
    ):
        exit_code = await run_all.main(
            task_list_file=str(task_list),
            config_to_test="dedupe-config",
            data_dir=str(tmp_path / "data"),
            save_submission_dir=str(tmp_path / "submissions"),
            overwrite_submission=False,
            print_submission=False,
            num_attempts=1,
            retry_attempts=1,
            logs_base_dir=tmp_path / "logs",
        )

    scheduled_task_ids = [call.args[1] for call in run_wrapper.await_args_list]
    assert exit_code == 0
    assert scheduled_task_ids == ["task-b", "task-a", "task-c"]


def test_resolve_config_runs_groups_providers_and_isolates_paths():
    args = argparse.Namespace(
        configs=["xai-high", "openai-high", "xai-low"],
        save_submission_root=Path("submissions"),
        run_name="v2",
        datasets=None,
        logs_base_dir=Path("logs"),
        max_concurrency=8,
    )
    providers = {
        "xai-high": SimpleNamespace(provider="xai"),
        "openai-high": SimpleNamespace(provider="openai"),
        "xai-low": SimpleNamespace(provider="xai"),
    }

    with patch.object(
        run_configs,
        "read_models_config",
        side_effect=lambda config: providers[config],
    ):
        runs = run_configs.resolve_config_runs(args, ["--data_dir", "data/v2"])

    assert [run.rate_limit_divisor for run in runs] == [2, 1, 2]
    assert runs[0].submission_dir == Path("submissions/xai-high/v2")
    assert runs[0].logs_dir == Path("logs/xai-high/v2")
    assert "--rate-limit-divisor" in runs[0].command
    assert runs[0].command[runs[0].command.index("--max-concurrency") + 1] == "8"
    assert runs[0].command[-2:] == ("--data_dir", "data/v2")


def test_resolve_config_runs_builds_config_by_dataset_matrix():
    args = argparse.Namespace(
        configs=["xai-high", "openai-high", "xai-low"],
        save_submission_root=Path("submissions"),
        run_name=None,
        datasets=[
            "v1/public_eval=data/v1/public_eval",
            "v2/public_eval=data/v2/public_eval",
        ],
        logs_base_dir=Path("logs"),
    )
    providers = {
        "xai-high": SimpleNamespace(provider="xai"),
        "openai-high": SimpleNamespace(provider="openai"),
        "xai-low": SimpleNamespace(provider="xai"),
    }

    with patch.object(
        run_configs,
        "read_models_config",
        side_effect=lambda config: providers[config],
    ):
        runs = run_configs.resolve_config_runs(args, ["--log-level", "INFO"])

    assert [(run.config, run.dataset) for run in runs] == [
        ("xai-high", "v1/public_eval"),
        ("xai-high", "v2/public_eval"),
        ("openai-high", "v1/public_eval"),
        ("openai-high", "v2/public_eval"),
        ("xai-low", "v1/public_eval"),
        ("xai-low", "v2/public_eval"),
    ]
    assert [run.rate_limit_divisor for run in runs] == [4, 4, 2, 2, 4, 4]
    assert runs[0].submission_dir == Path("submissions/xai-high/v1/public_eval")
    assert runs[1].logs_dir == Path("logs/xai-high/v2/public_eval")
    assert runs[0].command[-2:] == ("--data_dir", "data/v1/public_eval")
    assert runs[1].command[-2:] == ("--data_dir", "data/v2/public_eval")


def test_raw_api_log_root_isolated_by_config_and_dataset():
    args = argparse.Namespace(
        configs=["openai-high"],
        save_submission_root=Path("submissions"),
        run_name=None,
        datasets=["v1/public_eval=data/v1/public_eval"],
        logs_base_dir=Path("logs"),
        raw_api_log_root=Path("raw_api_logs"),
        max_concurrency=None,
    )

    with patch.object(
        run_configs,
        "read_models_config",
        return_value=SimpleNamespace(provider="openai"),
    ):
        run = run_configs.resolve_config_runs(args, [])[0]

    expected = Path("raw_api_logs/openai-high/v1/public_eval")
    assert run.raw_api_log_dir == expected
    option_index = run.command.index("--raw-api-log-dir")
    assert run.command[option_index + 1] == str(expected)


def test_raw_api_child_directory_cannot_be_forwarded():
    with pytest.raises(SystemExit):
        run_configs.parse_args(
            [
                "--configs",
                "openai-high",
                "--run-name",
                "v1/public_eval",
                "--raw-api-log-dir",
                "somewhere",
            ]
        )


def test_parse_args_forwards_run_all_options():
    args, forwarded = run_configs.parse_args(
        [
            "--configs",
            "xai-high",
            "xai-low",
            "--data_dir",
            "data/v2",
            "--save_submission_root",
            "submissions",
            "--run_name",
            "v2",
            "--max-concurrency",
            "8",
            "--log-level",
            "INFO",
        ]
    )

    assert args.configs == ["xai-high", "xai-low"]
    assert args.max_concurrency == 8
    assert forwarded == ["--data_dir", "data/v2", "--log-level", "INFO"]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_parse_args_rejects_invalid_max_concurrency(value):
    with pytest.raises(SystemExit):
        run_configs.parse_args(
            [
                "--configs",
                "xai-high",
                "--run_name",
                "v2",
                "--max-concurrency",
                value,
            ]
        )


def test_parse_args_accepts_named_datasets_without_run_name():
    args, forwarded = run_configs.parse_args(
        [
            "--configs",
            "xai-high",
            "xai-low",
            "--datasets",
            "v1/public_eval=data/v1/public_eval",
            "v2/public_eval=data/v2/public_eval",
            "--save_submission_root",
            "submissions",
            "--log-level",
            "INFO",
        ]
    )

    assert args.datasets == [
        "v1/public_eval=data/v1/public_eval",
        "v2/public_eval=data/v2/public_eval",
    ]
    assert args.run_name is None
    assert forwarded == ["--log-level", "INFO"]


@pytest.mark.parametrize(
    "dataset",
    [
        "/v1/public_eval=data/v1/public_eval",
        "../public_eval=data/v1/public_eval",
        "v1/../../public_eval=data/v1/public_eval",
    ],
)
def test_named_datasets_reject_unsafe_output_paths(dataset):
    with pytest.raises(SystemExit):
        run_configs.parse_args(
            [
                "--configs",
                "xai-high",
                "--datasets",
                dataset,
            ]
        )


def test_named_datasets_reject_forwarded_data_dir():
    with pytest.raises(SystemExit):
        run_configs.parse_args(
            [
                "--configs",
                "xai-high",
                "--datasets",
                "v1=data/v1",
                "--data_dir",
                "data/v2",
            ]
        )


class FakeProcess:
    def __init__(self, release: asyncio.Event, returncode: int = 0):
        self._release = release
        self._planned_returncode = returncode
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_eof()
        self.stderr.feed_eof()

    async def wait(self):
        await self._release.wait()
        self.returncode = self._planned_returncode
        return self.returncode

    def terminate(self):
        self.terminated = True
        self._planned_returncode = -15
        self._release.set()

    def kill(self):
        self.killed = True
        self._planned_returncode = -9
        self._release.set()


def make_run(config: str, dataset: str = "v2") -> run_configs.ConfigRun:
    return run_configs.ConfigRun(
        config=config,
        dataset=dataset,
        provider="xai",
        rate_limit_divisor=2,
        submission_dir=Path("submissions") / config / "v2",
        logs_dir=Path("logs") / config / "v2",
        command=("python", "run_all.py", "--config", config),
    )


@pytest.mark.asyncio
async def test_all_config_processes_start_before_any_finishes():
    release = asyncio.Event()
    started = []

    async def process_factory(*command, **kwargs):
        started.append(command[-1])
        if len(started) == 2:
            release.set()
        return FakeProcess(release)

    results = await asyncio.wait_for(
        run_configs.run_all_configs(
            [make_run("xai-high"), make_run("xai-low")],
            process_factory=process_factory,
        ),
        timeout=1,
    )

    assert started == ["xai-high", "xai-low"]
    assert [result.returncode for result in results] == [0, 0]


@pytest.mark.asyncio
async def test_config_failure_is_reported_without_stopping_other_configs():
    release = asyncio.Event()
    release.set()
    returncodes = iter([0, 7])

    async def process_factory(*command, **kwargs):
        return FakeProcess(release, returncode=next(returncodes))

    results = await run_configs.run_all_configs(
        [make_run("xai-high"), make_run("xai-low")],
        process_factory=process_factory,
    )

    assert [(result.config, result.dataset, result.returncode) for result in results] == [
        ("xai-high", "v2", 0),
        ("xai-low", "v2", 7),
    ]


@pytest.mark.asyncio
async def test_cancelling_config_terminates_its_process():
    release = asyncio.Event()
    process = FakeProcess(release)
    started = asyncio.Event()

    async def process_factory(*command, **kwargs):
        started.set()
        return process

    task = asyncio.create_task(
        run_configs.run_config(make_run("xai-high"), process_factory=process_factory)
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.terminated is True
    assert process.returncode == -15
