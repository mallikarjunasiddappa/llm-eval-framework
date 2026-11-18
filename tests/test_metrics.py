import pytest
from ai_qa.metrics import pass_rate, average_score

@pytest.mark.metrics
def test_two_of_three_pass():
    scores = {"login": 0.9, "search": 0.6, "logout": 0.8}
    assert pass_rate(scores, 0.7) == 0.67

@pytest.mark.metrics
def test_empty_scores_raise_error():
    with pytest.raises(ValueError):
        pass_rate({}, 0.7)

@pytest.mark.metrics
@pytest.mark.parametrize("scores, expected", [
    ({"a": 0.9, "b": 0.8}, 1.0),
    ({"a": 0.5, "b": 0.9}, 0.5),
    ({"a": 0.1, "b": 0.2}, 0.0),
])
def test_pass_rate_cases(scores, expected):
    assert pass_rate(scores, 0.7) == expected


@pytest.mark.metrics
def test_float_approx():
    assert 0.1 + 0.2 == pytest.approx(0.3)

@pytest.mark.metrics
def test_average_score():
    scores = {"login": 0.8, "search": 0.6, "logout": 0.9}
    assert average_score(scores) == pytest.approx(0.7667, abs=0.001)