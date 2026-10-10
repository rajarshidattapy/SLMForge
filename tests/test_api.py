"""API behaviour: project lifecycle, pipeline metadata and approval gates."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_and_capabilities(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    # Ollama is not running in CI; capabilities must still answer and warn.
    caps = client.get("/api/capabilities")
    assert caps.status_code == 200
    body = caps.json()
    assert body["sandbox_backend"] == "subprocess"
    assert body["planner_llm_configured"] is False
    assert any("planner LLM" in w for w in body["warnings"])


def test_capabilities_report_an_unusable_docker_backend(client, monkeypatch) -> None:
    """A stopped Docker daemon must be visible up front, not at job time."""
    from slmforge.config import settings
    from slmforge.sandbox import docker_backend

    monkeypatch.setattr(settings, "sandbox_backend", "docker")
    monkeypatch.setattr(docker_backend, "docker_available", lambda: False)

    body = client.get("/api/capabilities").json()
    assert body["sandbox_backend"] == "docker"
    assert body["sandbox_error"] is not None
    assert "Start Docker Desktop" in body["sandbox_error"]
    assert any("Docker" in w for w in body["warnings"])


def test_capabilities_are_clean_when_the_backend_works(client, monkeypatch) -> None:
    from slmforge.config import settings
    from slmforge.sandbox import docker_backend

    monkeypatch.setattr(settings, "sandbox_backend", "docker")
    monkeypatch.setattr(docker_backend, "docker_available", lambda: True)

    body = client.get("/api/capabilities").json()
    assert body["sandbox_error"] is None


def test_hf_jobs_backend_is_flagged_as_unavailable(client, monkeypatch) -> None:
    from slmforge.config import settings

    monkeypatch.setattr(settings, "sandbox_backend", "hf_jobs")
    body = client.get("/api/capabilities").json()
    assert body["sandbox_error"] is not None
    assert "credits" in body["sandbox_error"]


def test_create_list_and_fetch_project(client: TestClient, project: dict) -> None:
    assert project["status"] == "draft"
    assert project["spec"] is None
    project_id = project["id"]

    listed = client.get("/api/projects")
    assert listed.status_code == 200
    assert any(p["id"] == project_id for p in listed.json())

    detail = client.get(f"/api/projects/{project_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["prompt"] == project["prompt"]
    # Creating a project emits an audit event the dashboard will show.
    assert any(e["message"] == "Project created" for e in body["events"])


def test_create_project_rejects_empty_prompt(client: TestClient) -> None:
    response = client.post("/api/projects", json={"prompt": "short"})
    assert response.status_code == 422


def test_unknown_project_is_404(client: TestClient) -> None:
    assert client.get("/api/projects/doesnotexist").status_code == 404
    assert client.get("/api/projects/doesnotexist/events").status_code == 404


def test_pipeline_overview_reports_stage_status(client: TestClient) -> None:
    project = client.post(
        "/api/projects", json={"prompt": "Classify tickets into four categories please."}
    ).json()
    overview = client.get(f"/api/projects/{project['id']}/pipeline")
    assert overview.status_code == 200
    stages = overview.json()["stages"]
    assert next(s["name"] for s in stages) == "analyze"
    assert {s["name"] for s in stages} == {
        "analyze",
        "curate_data",
        "select_model",
        "baseline",
        "plan_training",
        "train",
        "evaluate",
        "deploy",
    }
    # The flag must be honest about what is actually built.
    assert [s["name"] for s in stages if s["implemented"]] == ["analyze"]
    assert next(s for s in stages if s["name"] == "train")["requires_approval"] is True


def test_spec_can_reviewed_and_edited(client: TestClient, project: dict) -> None:
    project_id = project["id"]
    empty = client.get(f"/api/projects/{project_id}/spec")
    assert empty.status_code == 200
    assert empty.json()["spec"] is None

    spec = {
        "title": "Ticket classifier",
        "summary": "Route support tickets.",
        "task_type": "classification",
        "input_description": "Ticket text",
        "allowed_labels": ["Billing", "Technical", "Account Access", "General"],
        "output": {"format": "json_object"},
    }
    updated = client.put(f"/api/projects/{project_id}/spec", json={"spec": spec, "note": "edited"})
    assert updated.status_code == 200, updated.text
    assert updated.json()["status"] == "specified"

    fetched = client.get(f"/api/projects/{project_id}/spec").json()
    assert fetched["spec"]["task_type"] == "classification"
    assert fetched["spec"]["allowed_labels"][2] == "Account Access"


def test_unimplemented_stage_returns_501(client: TestClient, project: dict) -> None:
    """An unbuilt stage must say so rather than pretend to succeed."""
    response = client.post(f"/api/projects/{project['id']}/stages/curate_data/run", json={})
    assert response.status_code == 501
    assert "DatasetStage" in response.json()["detail"] or "curate_data" in response.json()["detail"]


def test_analyze_stage_actually_runs(client: TestClient, project: dict) -> None:
    """Stage 1 is implemented, so it must return a real spec, not a 501."""
    response = client.post(f"/api/projects/{project['id']}/stages/analyze/run", json={})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["output"]["task_type"] in {"classification", "unsupported"}
    assert body["output"]["readiness"] in {
        "ready_for_review",
        "needs_clarification",
        "unsupported",
    }

    # The spec must land on the project row for the dashboard to show.
    stored = client.get(f"/api/projects/{project['id']}/spec").json()
    assert stored["spec"] is not None
    assert stored["status"] == "specified"


def test_long_running_stages_are_rejected_synchronously(client: TestClient, project: dict) -> None:
    for stage in ("train", "baseline", "evaluate"):
        response = client.post(
            f"/api/projects/{project['id']}/stages/{stage}/run",
            json={"approved": True},
        )
        assert response.status_code == 409, f"{stage} should not run synchronously"


def test_approval_gate_blocks_gated_stage(client: TestClient, project: dict) -> None:
    # select_model requires approval; without it the request must not execute.
    response = client.post(f"/api/projects/{project['id']}/stages/select_model/run", json={})
    assert response.status_code == 428


def test_next_step_points_at_first_incomplete_stage(client: TestClient, project: dict) -> None:
    response = client.get(f"/api/projects/{project['id']}/next")
    assert response.status_code == 200
    assert response.json()["next"] == "analyze"


def test_delete_project(client: TestClient, project: dict) -> None:
    project_id = project["id"]
    assert client.delete(f"/api/projects/{project_id}").status_code == 204
    assert client.get(f"/api/projects/{project_id}").status_code == 404
