from fastapi.testclient import TestClient
from app import app

# Create test client
client = TestClient(app)


# Test home endpoint
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# Test health endpoint
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# Test ticket prediction
def test_predict_ticket():
    ticket = {"text": "My card was charged twice for the same order. Please help me."}

    response = client.post("/predict", json=ticket)

    assert response.status_code == 200

    result = response.json()

    assert "category" in result
    assert "priority" in result
    assert "category_confidence" in result
    assert "priority_confidence" in result


# Test empty ticket validation
def test_empty_ticket():
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422


# Test missing text validation
def test_missing_text():
    response = client.post("/predict", json={})

    assert response.status_code == 422
