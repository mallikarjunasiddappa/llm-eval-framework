from dotenv import load_dotenv
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase
from ai_qa.llm import ask

load_dotenv()
judge = AnthropicModel(model="claude-haiku-5-5")     #Temperature = how random the AI is. 0 means as consistent as possible, which is what you want from an examiner
metric = AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True)


question = "Describe Stockholm in one sentence, including which country it is in."

answer = ask(question)
print("answer:", answer)

test_case = LLMTestCase(input= question, actual_output= answer)

metric.measure(test_case)
print("score", metric.score)
print("reason", metric.reason)