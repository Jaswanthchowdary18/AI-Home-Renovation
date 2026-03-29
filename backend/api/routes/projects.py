from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from uuid import UUID
import uuid
from db.session import AsyncSession, get_db

router = APIRouter()

class ProjectSummary(BaseModel):
    id: str
    name: str
    status: str
    city: Optional[str]
    budget_inr: Optional[int]
    theme: Optional[str]
    created_at: str
    render_url: Optional[str] = None

@router.get("/", response_model=List[ProjectSummary])
async def list_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List all projects for authenticated user."""
    # In production: filter by current_user.id
    return []

@router.get("/{project_id}")
async def get_project(project_id: str, db: AsyncSession = Depends(get_db)):
    # Return full project with all agent outputs
    return {"id": project_id, "status": "complete"}

@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str, db: AsyncSession = Depends(get_db)):
    pass

@router.post("/{project_id}/version")
async def save_version(project_id: str, db: AsyncSession = Depends(get_db)):
    """Snapshot current project state as a new version."""
    version_id = str(uuid.uuid4())
    return {"version_id": version_id, "version_number": 1}
