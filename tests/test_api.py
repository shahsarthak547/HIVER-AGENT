from fastapi.testclient import TestClient
from app.api.main import app
client = TestClient(app)
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
def test_intents():
    response = client.get("/api/v1/intents")
    assert response.status_code == 200
    assert "battery_charging" in response.json()["intents"]
def test_classify():
    response = client.post(
        "/api/v1/classify",
        json={
            "message": "My iPhone battery is draining really quickly"
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1
def test_invalid_message():
    response = client.post(
        "/api/v1/classify",
        json={"message": ""},
    )
    assert response.status_code == 422