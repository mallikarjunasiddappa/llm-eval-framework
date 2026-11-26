"""
Unit tests for measure_with_retry using a fake metric that replays chosen
judge replies, so the retry logic is tested without calling a real judge.
"""
from pydantic import BaseModel
from deepeval.errors import DeepEvalError
from ai_qa.judge import measure_with_retry


class ReasonScore(BaseModel):
    reason: str
    score: float


GOOD = {"reason": "ok", "score": 1.0}
MISSING = {"reason": "no score"}
BROKEN = "not json"


class FakeMetric:
    def __init__(self, replies):
        self.replies = replies
        self.score = None

    def measure(self, test_case):
        reply = self.replies.pop(0)
        if reply == BROKEN:
            raise DeepEvalError("Evaluation LLM outputted an invalid JSON.")
        self.score = ReasonScore.model_validate(reply).score


def fake_factory(replies):
    return lambda: FakeMetric(replies)


def test_first_try_works():
    assert measure_with_retry(fake_factory([GOOD]), None) == (1.0, 1)


def test_retry_then_works():
    assert measure_with_retry(fake_factory([MISSING, GOOD]), None) == (1.0, 2)


def test_invalid_json_then_works():
    assert measure_with_retry(fake_factory([BROKEN, GOOD]), None) == (1.0, 2)


def test_all_tries_fail():
    assert measure_with_retry(fake_factory([MISSING, BROKEN, MISSING]), None) == (None, 3)