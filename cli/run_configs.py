"""Launch multiple ``run_all.py`` benchmark configurations concurrently."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Awaitable, Callable, Sequence, TextIO

from arc_agi_benchmarking.utils.task_utils import read_models_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ALL_PATH = PROJECT_ROOT / "cli" / "run_all.py"


@dataclass(frozen=True)
class ConfigRun:
    """A fully resolved child benchmark invocation."""

    config: str
    dataset: str
    provider: str
    rate_limit_divisor: int
    submission_dir: Path
    logs_dir: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class ConfigResult:
    """The exit status of one child benchmark process."""

    config: str
    dataset: str
    returncode: int


@dataclass(frozen=True)
class DatasetSpec:
    """A named data directory used to construct output paths."""

    name: str
    data_dir: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple ARC model configs concurrently. Options not consumed by "
            "this launcher are forwarded to each cli/run_all.py process."
        ),
        epilog=(
            "Example: uv run cli/run_configs.py --configs model-high model-low "
            "--datasets v1/public_eval=data/v1/public_eval "
            "v2/public_eval=data/v2/public_eval "
            "--save_submission_root submissions"
        ),
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Model configuration names from models.yml.",
    )
    parser.add_argument(
        "--save_submission_root",
        "--save-submission-root",
        dest="save_submission_root",
        type=Path,
        default=Path("submissions"),
        help="Root for per-config submissions (default: submissions).",
    )
    run_mode = parser.add_mutually_exclusive_group(required=True)
    run_mode.add_argument(
        "--run_name",
        "--run-name",
        dest="run_name",
        help="Relative run path appended after each config, such as v2.",
    )
    run_mode.add_argument(
        "--datasets",
        nargs="+",
        metavar="NAME=PATH",
        help=(
            "Named data directories to run as a config-by-dataset matrix, "
            "such as v1/public_eval=data/v1/public_eval. NAME may be a safe "
            "relative path and determines the output directory."
        ),
    )
    parser.add_argument(
        "--logs-base-dir",
        type=Path,
        default=Path("logs"),
        help="Root for isolated per-config logs (default: logs).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help=(
            "Maximum in-flight ARC tasks per provider across all child "
            "processes."
        ),
    )
    return parser


def _option_is_present(arguments: Sequence[str], option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in arguments)


def parse_dataset_specs(values: Sequence[str]) -> list[DatasetSpec]:
    datasets: list[DatasetSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"dataset must use NAME=PATH syntax: {value}")
        name, raw_path = value.split("=", 1)
        name_path = Path(name)
        if (
            not name.strip()
            or name_path.is_absolute()
            or name_path == Path(".")
            or ".." in name_path.parts
        ):
            raise ValueError(
                "dataset name must be a non-empty relative path without '..': "
                f"{name}"
            )
        if not raw_path:
            raise ValueError(f"dataset path must not be empty: {name}")
        datasets.append(
            DatasetSpec(name=name_path.as_posix(), data_dir=Path(raw_path))
        )

    names = [dataset.name for dataset in datasets]
    if len(set(names)) != len(names):
        raise ValueError("dataset names must be unique")
    return datasets


def _validate_launcher_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    forwarded_args: Sequence[str],
) -> None:
    if len(set(args.configs)) != len(args.configs):
        parser.error("--configs must not contain duplicate configuration names")
    if args.max_concurrency is not None and args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")
    if _option_is_present(forwarded_args, "--log-level"):
        parser.error("--log-level is no longer supported; logging is always enabled")

    unsafe_configs = [
        config
        for config in args.configs
        if not config or config in {".", ".."} or Path(config).name != config
    ]
    if unsafe_configs:
        parser.error(
            "configuration names must be non-empty path components: "
            + ", ".join(unsafe_configs)
        )

    if args.run_name:
        run_name = Path(args.run_name)
        if run_name.is_absolute() or not args.run_name.strip() or ".." in run_name.parts:
            parser.error("--run_name must be a non-empty relative path without '..'")
    else:
        try:
            parse_dataset_specs(args.datasets)
        except ValueError as exc:
            parser.error(str(exc))
        if _option_is_present(forwarded_args, "--data_dir"):
            parser.error("--data_dir cannot be combined with --datasets")

    conflicting_options = (
        "--config",
        "--save_submission_dir",
        "--submissions-root",
        "--rate-limit-divisor",
    )
    conflicts = [
        option
        for option in conflicting_options
        if _option_is_present(forwarded_args, option)
    ]
    if conflicts:
        parser.error(
            "these options are managed by run_configs.py and cannot be forwarded: "
            + ", ".join(conflicts)
        )


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = build_parser()
    args, forwarded_args = parser.parse_known_args(argv)
    _validate_launcher_args(parser, args, forwarded_args)
    return args, forwarded_args


def resolve_config_runs(
    args: argparse.Namespace,
    forwarded_args: Sequence[str],
) -> list[ConfigRun]:
    """Resolve providers, per-provider divisors, paths, and child commands."""
    providers: dict[str, str] = {}
    for config in args.configs:
        try:
            providers[config] = read_models_config(config).provider
        except ValueError as exc:
            raise ValueError(f"Cannot launch config '{config}': {exc}") from exc

    if args.datasets:
        datasets = parse_dataset_specs(args.datasets)
    else:
        datasets = [DatasetSpec(name=args.run_name, data_dir=None)]

    provider_counts = Counter(
        providers[config]
        for config in args.configs
        for _dataset in datasets
    )
    runs: list[ConfigRun] = []
    max_concurrency = getattr(args, "max_concurrency", None)
    concurrency_args = (
        ("--max-concurrency", str(max_concurrency))
        if max_concurrency is not None
        else ()
    )

    for config in args.configs:
        provider = providers[config]
        divisor = provider_counts[provider]
        for dataset in datasets:
            submission_dir = args.save_submission_root / config / dataset.name
            logs_dir = args.logs_base_dir / config / dataset.name
            dataset_args = (
                ("--data_dir", str(dataset.data_dir))
                if dataset.data_dir is not None
                else ()
            )
            command = (
                sys.executable,
                str(RUN_ALL_PATH),
                "--config",
                config,
                "--save_submission_dir",
                str(submission_dir),
                "--logs-base-dir",
                str(logs_dir),
                "--rate-limit-divisor",
                str(divisor),
                *concurrency_args,
                *forwarded_args,
                *dataset_args,
            )
            runs.append(
                ConfigRun(
                    config=config,
                    dataset=dataset.name,
                    provider=provider,
                    rate_limit_divisor=divisor,
                    submission_dir=submission_dir,
                    logs_dir=logs_dir,
                    command=command,
                )
            )

    return runs


async def _prefix_stream(
    stream: asyncio.StreamReader,
    config: str,
    destination: TextIO,
) -> None:
    while line := await stream.readline():
        text = line.decode(errors="replace")
        print(f"[{config}] {text}", end="", file=destination, flush=True)


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return

    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


ProcessFactory = Callable[..., Awaitable[asyncio.subprocess.Process]]


async def run_config(
    run: ConfigRun,
    process_factory: ProcessFactory = asyncio.create_subprocess_exec,
) -> ConfigResult:
    """Run one config and ensure cancellation stops its child process."""
    process = await process_factory(
        *run.command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    label = f"{run.config}/{run.dataset}"

    output_tasks = [
        asyncio.create_task(_prefix_stream(process.stdout, label, sys.stdout)),
        asyncio.create_task(_prefix_stream(process.stderr, label, sys.stderr)),
    ]
    try:
        returncode = await process.wait()
        await asyncio.gather(*output_tasks)
        return ConfigResult(
            config=run.config,
            dataset=run.dataset,
            returncode=returncode,
        )
    except asyncio.CancelledError:
        await _stop_process(process)
        for task in output_tasks:
            task.cancel()
        await asyncio.gather(*output_tasks, return_exceptions=True)
        raise


async def run_all_configs(
    runs: Sequence[ConfigRun],
    process_factory: ProcessFactory = asyncio.create_subprocess_exec,
) -> list[ConfigResult]:
    """Start every config immediately and wait for all of them to finish."""
    tasks = [
        asyncio.create_task(run_config(run, process_factory=process_factory))
        for run in runs
    ]
    return await asyncio.gather(*tasks)


def main(argv: Sequence[str] | None = None) -> int:
    args, forwarded_args = parse_args(argv)
    try:
        runs = resolve_config_runs(args, forwarded_args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print("Launching configs concurrently:")
    for run in runs:
        print(
            f"  {run.config}/{run.dataset}: provider={run.provider}, "
            f"rate-share=1/{run.rate_limit_divisor}, output={run.submission_dir}"
        )

    try:
        results = asyncio.run(run_all_configs(runs))
    except KeyboardInterrupt:
        print("Interrupted; stopped active config runs.", file=sys.stderr)
        return 130

    failed = [result for result in results if result.returncode != 0]
    print("Config run summary:")
    for result in results:
        status = "succeeded" if result.returncode == 0 else f"failed ({result.returncode})"
        print(f"  {result.config}/{result.dataset}: {status}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
