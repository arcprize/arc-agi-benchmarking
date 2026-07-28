import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from cli import run_all
from main import ARCTester
from arc_agi_benchmarking.utils.submission_exists import (
    normalize_submission_pairs,
    submission_is_complete,
)


TASK = {
    "train": [{"input": [[1]], "output": [[1]]}],
    "test": [
        {"input": [[2]], "output": [[2]]},
        {"input": [[3]], "output": [[3]]},
    ],
}


class StubAttempt:
    def __init__(self, answer, pair_index):
        self.answer = answer
        self.pair_index = pair_index
        self.correct = None

    def model_dump(self, mode=None):
        return {
            "answer": self.answer,
            "correct": self.correct,
            "metadata": {"pair_index": self.pair_index},
        }


def _write_task_and_submission(tmp_path, submission):
    data_dir = tmp_path / "data"
    submission_dir = tmp_path / "submissions"
    data_dir.mkdir()
    submission_dir.mkdir()
    (data_dir / "task.json").write_text(json.dumps(TASK))
    (submission_dir / "task.json").write_text(json.dumps(submission))
    return data_dir, submission_dir


def _tester(submission_dir, predictions, num_attempts=2):
    tester = object.__new__(ARCTester)
    tester.config = "test-config"
    tester.save_submission_dir = str(submission_dir)
    tester.overwrite_submission = False
    tester.print_submission = False
    tester.num_attempts = num_attempts
    tester.retry_attempts = 1
    tester.get_task_prediction = Mock(side_effect=predictions)
    return tester


def test_normalize_uses_metadata_to_restore_an_omitted_pair():
    saved = [
        {
            "attempt_1": {
                "answer": [[3]],
                "metadata": {"pair_index": 1},
            }
        }
    ]

    normalized = normalize_submission_pairs(saved, expected_num_pairs=2)

    assert normalized[0] == {}
    assert normalized[1] == saved[0]


def test_submission_with_null_or_missing_attempt_is_incomplete(tmp_path):
    _, submission_dir = _write_task_and_submission(
        tmp_path,
        [
            {"attempt_1": {"answer": [[2]]}, "attempt_2": None},
            {"attempt_1": {"answer": [[3]]}, "attempt_2": {"answer": [[3]]}},
        ],
    )

    assert not submission_is_complete(
        str(submission_dir), "task", expected_num_pairs=2, num_attempts=2
    )


def test_rerun_preserves_responses_and_fills_only_missing_slots(tmp_path):
    previous_attempt = {
        "answer": [[2]],
        "correct": True,
        "metadata": {"pair_index": 0},
    }
    data_dir, submission_dir = _write_task_and_submission(
        tmp_path,
        [{"attempt_1": previous_attempt, "attempt_2": None}],
    )
    tester = _tester(
        submission_dir,
        [
            StubAttempt([[9]], 0),
            StubAttempt([[3]], 1),
            StubAttempt([[8]], 1),
        ],
    )

    result = tester.generate_task_solution(str(data_dir), "task")

    assert tester.get_task_prediction.call_count == 3
    assert result[0]["attempt_1"] == previous_attempt
    assert result[0]["attempt_2"]["answer"] == [[9]]
    assert result[1]["attempt_1"]["answer"] == [[3]]
    assert result[1]["attempt_2"]["answer"] == [[8]]
    assert result[1]["attempt_1"]["correct"] is True
    assert result[1]["attempt_2"]["correct"] is False
    assert submission_is_complete(
        str(submission_dir), "task", expected_num_pairs=2, num_attempts=2
    )


def test_failed_retry_remains_null_and_is_eligible_on_next_run(tmp_path):
    data_dir, submission_dir = _write_task_and_submission(
        tmp_path,
        [
            {"attempt_1": {"answer": [[2]], "metadata": {"pair_index": 0}}},
            {"attempt_1": {"answer": [[3]], "metadata": {"pair_index": 1}}},
        ],
    )
    tester = _tester(submission_dir, [None, None])

    tester.generate_task_solution(str(data_dir), "task")

    saved = json.loads((submission_dir / "task.json").read_text())
    assert saved[0]["attempt_2"] is None
    assert saved[1]["attempt_2"] is None
    assert not submission_is_complete(
        str(submission_dir), "task", expected_num_pairs=2, num_attempts=2
    )


@pytest.mark.asyncio
async def test_run_all_schedules_only_submission_with_missing_attempts(tmp_path):
    data_dir = tmp_path / "data"
    submission_dir = tmp_path / "submissions"
    data_dir.mkdir()
    submission_dir.mkdir()
    for task_id in ("complete", "incomplete"):
        (data_dir / f"{task_id}.json").write_text(json.dumps(TASK))

    complete_pair = {
        "attempt_1": {"answer": [[1]]},
        "attempt_2": {"answer": [[1]]},
    }
    (submission_dir / "complete.json").write_text(
        json.dumps([complete_pair, complete_pair])
    )
    (submission_dir / "incomplete.json").write_text(
        json.dumps(
            [
                {"attempt_1": {"answer": [[1]]}, "attempt_2": None},
                complete_pair,
            ]
        )
    )

    model_config = SimpleNamespace(
        provider="submission-resume-provider",
        name="submission-resume-config",
        kwargs={},
    )
    run_wrapper = AsyncMock(return_value=True)
    with (
        patch.object(run_all, "get_model_config", return_value=model_config),
        patch.object(
            run_all,
            "read_provider_rate_limits",
            return_value={
                "submission-resume-provider": {"rate": 100, "period": 60}
            },
        ),
        patch.object(run_all, "run_single_test_wrapper", run_wrapper),
    ):
        exit_code = await run_all.main(
            task_list_file=None,
            config_to_test="submission-resume-config",
            data_dir=str(data_dir),
            save_submission_dir=str(submission_dir),
            overwrite_submission=False,
            print_submission=False,
            num_attempts=2,
            retry_attempts=1,
            logs_base_dir=tmp_path / "logs",
        )

    assert exit_code == 0
    assert [call.args[1] for call in run_wrapper.await_args_list] == ["incomplete"]
