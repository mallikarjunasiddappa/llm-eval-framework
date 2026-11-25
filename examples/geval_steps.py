"""
Shows the evaluation steps G-Eval generates from our correctness criteria.
Uses a fixed answer (no ask()) so only the judge varies.
"""

from deepeval.test_case import LLMTestCase
from ai_qa.judge import correctness_metric

test_case = LLMTestCase(
    input="Describe Stockholm in one sentence, including which country it is in.",
    actual_output="Stockholm, the capital of Sweden, is spread across 14 islands.",
    expected_output="Stockholm is the capital of Sweden.",
)

metric = correctness_metric(verbose=True)
metric.measure(test_case)