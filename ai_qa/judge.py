# judge.py - the evaluation judge and metric builders, configured in ONE place.
#   JUDGE_MODEL: Claude Haiku (independent from the model under test, cheap).
#   RELEVANCY_THRESHOLD / CORRECTNESS_THRESHOLD: team/risk decisions, can differ per metric.
#   relevancy_metric(): is the answer on topic? (no reference needed)
#   correctness_metric(): G-Eval - is the answer factually consistent with expected_output?

from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.models import AnthropicModel
from deepeval.test_case import SingleTurnParams
from functools import lru_cache

JUDGE_MODEL = "claude-haiku-5-5"
RELEVANCY_THRESHOLD = 0.7
CORRECTNESS_THRESHOLD = 0.7

@lru_cache
def get_judge():
    return AnthropicModel(model=JUDGE_MODEL)

def relevancy_metric():
    return AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model=get_judge())

def correctness_metric():
    return GEval(
        name="correctness",
        criteria="Determine whether the actual output is factually consistent with the expected output. Extra correct details beyond the expected output are fine and must not be penalized, unless they contradict the expected output or are factually wrong.",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT
        ],
        model=get_judge(),
        threshold=CORRECTNESS_THRESHOLD,
    )
