"""
Red check for the fun-fact rule: a one-word question answered with
an extra fun fact must fail.
"""
from deepeval.test_case import LLMTestCase
from ai_qa.judge import fun_fact_metric

test_case = LLMTestCase(
    input="What is the capital of Sweden? Answer in one word.",
    actual_output="Stockholm. Fun fact: it's built on 14 islands!",
)
metric = fun_fact_metric(verbose=True)
metric.measure(test_case)