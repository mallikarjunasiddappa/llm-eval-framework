
from dotenv import load_dotenv
from deepeval.metrics import GEval
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

load_dotenv()

judge = AnthropicModel(model="claude-haiku-5-5")

correctness = GEval(
    name="correctness",
    criteria="Determine whether the actual output is factually consistent with the expected output.",
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT
    ],
    model=judge,
    threshold=0.7
)

answers = [
    "Stockholm is the capital of Sweden.",
    "Sweden's capital city is Stockholm, located on the east coast.",
    "The capital of Sweden is Oslo.",
    "Stockholm is not the capital of Sweden; the capital is Gothenburg."
]

for answer in answers:
    test_case=LLMTestCase(
        input="What is the capital of Sweden?",
        actual_output=answer,
        expected_output="The capital of Sweden is Stockholm.",
    )
    correctness.measure(test_case)
    print(f"{correctness.score:.2f} | {answer}")
    print("   reason:", correctness.reason)