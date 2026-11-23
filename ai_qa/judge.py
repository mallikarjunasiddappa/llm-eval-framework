# judge.py - the evaluation judge and metric builders, configured in ONE place.
#   JUDGE_MODEL: Claude Haiku (independent from the model under test, cheap).
#   RELEVANCY_THRESHOLD / CORRECTNESS_THRESHOLD: team/risk decisions, can differ per metric.
#   relevancy_metric(): is the answer on topic? (no reference needed)
#   correctness_metric(): G-Eval - is the answer factually consistent with expected_output?

from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.metrics.g_eval import Rubric
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

def correctness_metric(verbose=False, strict=False):
    return GEval(
        name="correctness",
        evaluation_steps=[
            "Identify the key facts in the Expected Output that answer the Input.",
            "Check whether the Actual Output states those key facts without contradicting them.",
            "Do not penalize extra details in the Actual Output if they are factually correct and do not contradict the Expected Output.",
            "Heavily penalize any claim that contradicts the Expected Output or is factually wrong.",
        ],
        rubric=[
            Rubric(score_range=(0, 2), expected_outcome="Contradicts the Expected Output or is factually wrong."),
            Rubric(score_range=(3, 6), expected_outcome="No contradiction, but a key fact from the Expected Output is missing."),
            Rubric(score_range=(7, 9), expected_outcome="Key facts correct, with minor issues."),
            Rubric(score_range=(10, 10), expected_outcome="Key facts fully correct, no errors."),
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT
        ],
        model=get_judge(),
        threshold=CORRECTNESS_THRESHOLD,
        verbose_mode=verbose,
        strict_mode=strict
    )
