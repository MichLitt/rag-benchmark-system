from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.server import app


def test_token_protects_business_api_but_not_health(monkeypatch):
    monkeypatch.setenv("RAG_API_TOKEN", "test-token")
    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 200
        assert client.post("/v1/retrieve", json={}).status_code == 401
        assert client.post("/v1/retrieve", json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.post("/v1/retrieve", json={}, headers={"Authorization": "Bearer test-token"}).status_code == 422
