"""
API v1 main router
"""

from fastapi import APIRouter

from . import info, models, jobs, upload, user, webhooks

# Create main v1 router
router = APIRouter(prefix="/api/v1")

# Include all sub-routers
router.include_router(info.router, tags=["Info"])
router.include_router(models.router, prefix="/models", tags=["Models"])
router.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
router.include_router(upload.router, prefix="/upload", tags=["Upload"])
router.include_router(user.router, prefix="/user", tags=["User"])
router.include_router(webhooks.router, prefix="/webhooks", tags=["Webhooks"])