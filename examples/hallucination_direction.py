"""
Checks which direction the Hallucination score goes in our installed DeepEval:
one answer consistent with the given source, one that contradicts it.
"""
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric
from ai_qa.judge import get_judge

context = [
    "In 2022 Copenhagen abandoned its goal of becoming carbon-neutral by 2025. "
    "The city now aims to be among the first carbon-neutral capitals by 2030."
]
question = "What is Copenhagen's carbon-neutrality target?"
answers = {
    "consistent": "Copenhagen dropped its 2025 goal and now aims to be carbon-neutral by 2030.",
    "contradicts": "Copenhagen aims to become the first carbon-neutral capital by 2025.",
}

for label, answer in answers.items():
    metric = HallucinationMetric(model=get_judge(), verbose_mode=True)
    metric.measure(LLMTestCase(input=question, actual_output=answer, context=context))
    print(f"{label}: score={metric.score} success={metric.success} | {metric.reason}")