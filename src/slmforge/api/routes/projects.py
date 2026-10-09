"""Project lifecycle endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from ...log import get_logger
from ...models import Artifact, Event, Project
from ..deps import ProjectDep, SessionDep
from ..payloads import ArtifactOut, CreateProjectRequest, EventOut, ProjectDetail, ProjectOut

router = APIRouter(prefix="/projects", tags=["projects"])
logger = get_logger("api.projects")

_MAX_NAME = 80


def _derive_name(prompt: str) -> str:
    first_line = prompt.strip().splitlines()[0]
    name = first_line[:_MAX_NAME].strip()
    return name or "Untitled build"


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(payload: CreateProjectRequest, session: SessionDep) -> Project:
    """Create a build from a natural-language prompt.

    The prompt is stored verbatim; nothing is interpreted until the user runs
    the analyze stage, so a mistaken submission costs nothing.
    """
    project = Project(name=payload.name or _derive_name(payload.prompt), prompt=payload.prompt)
    session.add(project)
    session.flush()
    session.add(
        Event(
            project_id=project.id,
            message="Project created",
            data={"prompt_chars": len(payload.prompt)},
        )
    )
    session.commit()
    session.refresh(project)
    logger.info("Created project %s", project.id)
    return project


@router.get("", response_model=list[ProjectOut])
def list_projects(session: SessionDep, limit: int = 50, offset: int = 0) -> list[Project]:
    """List projects, newest first."""
    stmt = select(Project).order_by(Project.created_at.desc()).limit(limit).offset(offset)
    return list(session.execute(stmt).scalars())


@router.get("/{project_id}", response_model=ProjectDetail)
def get_project_detail(project: ProjectDep, session: SessionDep) -> ProjectDetail:
    """Full project record including recent events and artifacts."""
    events = session.execute(
        select(Event).where(Event.project_id == project.id).order_by(Event.at.desc()).limit(100)
    ).scalars()
    artifacts = session.execute(
        select(Artifact)
        .where(Artifact.project_id == project.id)
        .order_by(Artifact.created_at.desc())
    ).scalars()
    detail = ProjectDetail.model_validate(project)
    detail.events = [EventOut.model_validate(e) for e in events]
    detail.artifacts = [ArtifactOut.model_validate(a) for a in artifacts]
    return detail


@router.get("/{project_id}/events", response_model=list[EventOut])
def list_events(project: ProjectDep, session: SessionDep, limit: int = 200) -> list[Event]:
    """Append-only build log for the project dashboard."""
    stmt = (
        select(Event).where(Event.project_id == project.id).order_by(Event.at.desc()).limit(limit)
    )
    return list(session.execute(stmt).scalars())


@router.get("/{project_id}/artifacts", response_model=list[ArtifactOut])
def list_artifacts(project: ProjectDep, session: SessionDep) -> list[Artifact]:
    """Datasets, checkpoints, reports and logs produced for this project."""
    stmt = select(Artifact).where(Artifact.project_id == project.id).order_by(Artifact.created_at)
    return list(session.execute(stmt).scalars())


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project: ProjectDep, session: SessionDep) -> None:
    """Delete a project and its dependent rows.

    NOTE: artifact files on disk are intentionally *not* removed here; the
    retention/deletion policy is a Stage 9 concern (see docs/ROADMAP.md).
    """
    if project.status == "training":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete a project while a training job is running. Cancel it first.",
        )
    session.delete(project)
    session.commit()
