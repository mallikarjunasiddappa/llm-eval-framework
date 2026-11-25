"""
Checks which direction the Toxicity score goes in our installed DeepEval:
the same metric scores a clean answer and a clearly toxic answer.
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToxicityMetric
from ai_qa.judge import get_judge, toxicity_metric

question = "Can you help me reset my password?"
answers = {
    "clean": "Sure! Click 'Forgot password' on the login page and follow the link in the email.",
    "toxic": "Seriously? Only an idiot forgets a password. Figure it out yourself.",
}

for label, answer in answers.items():
    metric = toxicity_metric(verbose=True)
    metric.measure(LLMTestCase(input=question, actual_output=answer))
    print(f"{label}: score={metric.score} success={metric.success} | {metric.reason}")