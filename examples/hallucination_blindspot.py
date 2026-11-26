"""
Tests the blind spot of the Hallucination metric: does it catch a claim
the source never mentions (unsupported), or only claims that contradict it?
"""
from deepeval.test_case import LLMTestCase
from ai_qa.judge import hallucination_metric, groundedness_metric, relevancy_metric

COPENHAGEN = ("In 2022 Copenhagen abandoned its goal of becoming carbon-neutral by 2025. "
              "The city now aims to be among the first carbon-neutral capitals by 2030.")
SUPPORT = "Our support team is open Monday to Friday, 09:00-17:00 CET. We are closed on Swedish public holidays."

cases = [
    ("fully_grounded", COPENHAGEN, "What is Copenhagen's carbon-neutrality target?",
     "In 2022 Copenhagen dropped its 2025 goal; it now aims to be among the first carbon-neutral capitals by 2030."),
    ("paraphrase_borderline", COPENHAGEN, "What is Copenhagen's carbon-neutrality target?",
     "Copenhagen's carbon-neutrality target is to become carbon-neutral by 2030."),
    ("on_topic_invented", COPENHAGEN, "What is Copenhagen's carbon-neutrality target?",
     "Copenhagen now aims for carbon neutrality by 2030, and it will ban all petrol cars from the city centre in 2027."),
    ("invented_advice", SUPPORT, "What is the support phone number?",
     "The support phone number isn't provided in the information you gave. "
     "You might want to check the official website or contact them via email."),
    ("held_out_faithful", SUPPORT, "When is support open?",
     "Support is open on weekdays from 9 to 5 CET, but not on Swedish public holidays."),
    ("held_out_broader", SUPPORT, "When is support open?",
     "Support is open every day from 9 to 5 CET."),
]

for label, source, question, answer in cases:
    metric = groundedness_metric()
    metric.measure(LLMTestCase(input=question, actual_output=answer, context=[source]))
    print(f"{label}: score={metric.score} success={metric.success} | {metric.reason}")
