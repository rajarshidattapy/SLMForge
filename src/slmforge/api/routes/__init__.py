"""HTTP routers for the SLMForge API."""

from __future__ import annotations

from fastapi import APIRouter

from . import deployments, health, jobs, projects, stages

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(projects.router)
api_router.include_router(stages.router)
api_router.include_router(jobs.router)
api_router.include_router(deployments.router)

__all__ = ["api_router", "deployments", "health", "jobs", "projects", "stages"]
