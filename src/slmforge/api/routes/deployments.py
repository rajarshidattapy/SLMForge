"""Deployment endpoints (read-only in the prototype; creation lands in Stage 8)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from ...models import Deployment, DeploymentStatus
from ..deps import ProjectDep, SessionDep

router = APIRouter(prefix="/projects/{project_id}/deployments", tags=["deployments"])


@router.get("")
def list_deployments(project: ProjectDep, session: SessionDep) -> list[dict]:
    """List endpoints published for this project."""
    rows = session.execute(
        select(Deployment)
        .where(Deployment.project_id == project.id)
        .order_by(Deployment.created_at)
    ).scalars()
    return [
        {
            "id": d.id,
            "target": d.target,
            "status": d.status.value,
            "model_id": d.model_id,
            "revision": d.revision,
            "endpoint_url": d.endpoint_url,
            "usage": d.usage,
            "created_at": d.created_at,
            "stopped_at": d.stopped_at,
        }
        for d in rows
    ]


@router.post("/{deployment_id}/stop")
def stop_deployment(deployment_id: str, project: ProjectDep, session: SessionDep) -> dict[str, str]:
    """Stop an endpoint to stop billing. Data and configuration are retained."""
    deployment = session.get(Deployment, deployment_id)
    if deployment is None or deployment.project_id != project.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found")
    if deployment.status == DeploymentStatus.stopped:
        return {"status": "already_stopped", "id": deployment.id}
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Stopping endpoints is implemented in Stage 8 (see docs/ROADMAP.md).",
    )
