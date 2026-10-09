"""FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Project

SessionDep = Annotated[Session, Depends(get_session)]


def get_project(project_id: str, session: SessionDep) -> Project:
    """Load a project by id or raise 404."""
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


ProjectDep = Annotated[Project, Depends(get_project)]
