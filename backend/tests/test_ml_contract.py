import pytest
from ml_model import classify_review


def _assert_ml_contract(res: dict):
    assert "classification" in res
    assert res["classification"] in ["REAL", "FAKE", "UNCERTAIN"]

    assert "fake_probability" in res
    fp = float(res["fake_probability"])
    assert 0.0 <= fp <= 1.0

    assert "score" in res
    sc = float(res["score"])
    assert 0.0 <= sc <= 1.0

    assert "reasoning" in res
    assert isinstance(res["reasoning"], str)


def test_classify_review_empty_returns_uncertain():
    res = classify_review("   ")
    assert res["classification"] == "UNCERTAIN"
    _assert_ml_contract(res)


def test_classify_review_hebrew_smoke():
    res = classify_review("קניתי את זה לשבוע. עובד טוב, אבל הגיע עם שריטה קטנה.")
    _assert_ml_contract(res)


def test_classify_review_english_smoke():
    res = classify_review("I used it for a week. It works, but delivery was late.")
    _assert_ml_contract(res)


def test_classify_review_weird_chars():
    res = classify_review("מוצר!!!! 🤖🤖🤖 מעולהההההההה!!!! $$$")
    _assert_ml_contract(res)
