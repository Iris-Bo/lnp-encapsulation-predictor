from fastapi.testclient import TestClient
from app.main import app

# Use FastAPI TestClient to simulate HTTP requests without running a live server
client = TestClient(app)


def test_health_check():
    """Verify that the health check endpoint returns 200 and expected payload."""
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_prediction_success():
    """Verify that a valid LNP formulation returns a prediction within valid range [0, 100]."""
    payload = {
        "particle_size_nm": 85.5,
        "ionizable_ratio": 50.0,
        "helper_ratio": 10.0,
        "sterol_ratio": 38.5,
        "peg_ratio": 1.5,
        "ionizable_lipid": "MC3",
    }
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_encapsulation_efficiency" in data
        assert isinstance(data["predicted_encapsulation_efficiency"], float)
        assert 0.0 <= data["predicted_encapsulation_efficiency"] <= 100.0


def test_prediction_invalid_payload():
    """Verify that negative particle sizes or missing fields trigger validation errors (422)."""
    invalid_payload = {
        "particle_size_nm": -10.0,  # Invalid due to gt=0 constraint
        "ionizable_ratio": 50.0,
        "helper_ratio": 10.0,
        "sterol_ratio": 38.5,
        "peg_ratio": 1.5,
        "ionizable_lipid": "MC3"
    }
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=invalid_payload)
        assert response.status_code == 422

def test_prediction_missing_field():
    """Verify that omitting a required field triggers a validation error (422)"""
    incomplete_payload = { #left out particle size on purpose
        "ionizable_ratio": 50.0,
        "helper_ratio": 10.0,
        "sterol_ratio": 38.5,
        "peg_ratio": 1.5,
        "ionizable_lipid": "MC3"
    }
    with TestClient(app) as test_client:
        response = test_client.post("/predict",json=incomplete_payload)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert data["detail"][0]["type"] == "missing"
