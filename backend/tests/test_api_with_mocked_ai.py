import pytest
from fastapi.testclient import TestClient

import main  


@pytest.fixture()
def client():
    return TestClient(main.app)


def test_predict_mocked_classify_review(monkeypatch, client):
    def fake_classify_review(text: str):
        return {
            "classification": "FAKE",
            "score": 0.88,
            "fake_probability": 0.91,
            "model_used": "Mock",
            "translated_text": text,
            "reasoning": "Mocked result for test stability."
        }

    monkeypatch.setattr(main, "classify_review", fake_classify_review)

    r = client.post("/predict", json={"text": "מוצר משוגע! ממליץ בחום!"})
    assert r.status_code == 200
    data = r.json()

    assert data["classification"] in ["Real", "Fake", "UNCERTAIN"]
    assert data["classification"] == "Fake"
    assert data["is_fake"] is True
    assert 0.0 <= data["fake_probability"] <= 1.0
    assert 0.0 <= data["confidence_score"] <= 100.0
    assert "Mocked" in data["explanation"]


def test_batch_mocked_classify_review(monkeypatch, client):
    outputs = [
        {"classification": "REAL", "score": 0.80, "fake_probability": 0.10, "reasoning": "real-like"},
        {"classification": "FAKE", "score": 0.90, "fake_probability": 0.95, "reasoning": "fake-like"},
        {"classification": "UNCERTAIN", "score": 0.50, "fake_probability": 0.50, "reasoning": "uncertain"},
    ]

    def fake_classify_review(text: str):
        if len(text) < 10:
            return outputs[2]
        if "מומלץ" in text or "recommend" in text:
            return outputs[1]
        return outputs[0]

    monkeypatch.setattr(main, "classify_review", fake_classify_review)

    r = client.post("/predict/batch", json={"reviews": ["קצר", "מומלץ מאוד!!!", "חוויה אמיתית עם פירוט..."]})
    assert r.status_code == 200
    data = r.json()

    assert data["total_reviews"] == 3
    assert data["real_count"] + data["fake_count"] + data["uncertain_count"] == 3
    assert len(data["individual_results"]) == 3
