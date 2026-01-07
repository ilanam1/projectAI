import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _assert_single_response_schema(data: dict):
    assert "classification" in data
    assert data["classification"] in ["Real", "Fake", "UNCERTAIN"]

    assert "is_fake" in data
    assert isinstance(data["is_fake"], bool)

    assert "confidence_score" in data
    assert isinstance(data["confidence_score"], (int, float))
    assert 0.0 <= float(data["confidence_score"]) <= 100.0

    assert "fake_probability" in data
    assert isinstance(data["fake_probability"], (int, float))
    assert 0.0 <= float(data["fake_probability"]) <= 1.0

    assert "explanation" in data
    assert isinstance(data["explanation"], str)
    assert len(data["explanation"].strip()) > 0


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"


def test_predict_contract_hebrew_smoke():
    payload = {"text": "מוצר מעולה! הגיע מהר ואני מרוצה מאוד."}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    _assert_single_response_schema(data)


def test_predict_contract_english_smoke():
    payload = {"text": "Great product, works well, fast shipping."}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    _assert_single_response_schema(data)


def test_predict_rejects_empty_by_pydantic():
    r = client.post("/predict", json={"text": ""})
    assert r.status_code in (422, 400)


def test_batch_contract_smoke():
    payload = {
        "reviews": [
            "מוצר מעולה! מומלץ מאוד!",
            "קניתי וזה עבד יומיים ואז התקלקל. שירות לא עזר.",
            "great product highly recommended fast shipping",
        ]
    }
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    data = r.json()

    assert "total_reviews" in data and data["total_reviews"] == 3
    assert "real_count" in data
    assert "fake_count" in data
    assert "uncertain_count" in data

    assert data["real_count"] + data["fake_count"] + data["uncertain_count"] == 3

    assert 0.0 <= float(data["real_percentage"]) <= 100.0
    assert 0.0 <= float(data["fake_percentage"]) <= 100.0

    assert isinstance(data.get("recommendation", ""), str)

    results = data.get("individual_results", [])
    assert len(results) == 3
    for item in results:
        _assert_single_response_schema(item)

    flagged = data.get("flagged_for_review", [])
    assert isinstance(flagged, list)
    for idx in flagged:
        assert isinstance(idx, int)
        assert 0 <= idx < 3
