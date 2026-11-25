"""
Tests the blind spot of the Hallucination metric: does it catch a claim
the source never mentions (unsupported), or only claims that contradict it?
"""
from deepeval.test_case import LLMTestCase
from ai_qa.judge import hallucination_metric, groundedness_metric, relevancy_metric

context = [
    "In 2022 Copenhagen abandoned its goal of becoming carbon-neutral by 2025. "
    "The city now aims to be among the first carbon-neutral capitals by 2030."
]
question = "What is Copenhagen's carbon-neutrality target?"
answers = {
    "consistent_plus_unsupported": "Copenhagen now aims for carbon neutrality by 2030. "
                                   "Fun fact: Copenhagen has more bicycles than residents!",
    "unsupported_only": "Copenhagen has more bicycles than residents and the world's oldest amusement park.",
    "on_topic_invented": "Copenhagen now aims for carbon neutrality by 2030, "
                         "and it will ban all petrol cars from the city centre in 2027.",
    "fully_grounded": "In 2022 Copenhagen dropped its 2025 goal; it now aims to be among "
                      "the first carbon-neutral capitals by 2030.",
}

for label, answer in answers.items():
        for metric in [relevancy_metric(), hallucination_metric(), groundedness_metric()]:
            metric.measure(LLMTestCase(input=question, actual_output=answer, context=context))
            print(f"{label} / {metric.__name__}: score={metric.score} success={metric.success} | {metric.reason}")