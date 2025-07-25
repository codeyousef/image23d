"""
User API v1 endpoints
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from datetime import datetime

from backend.api.middleware.auth import get_current_user
from backend.services.history_service import history_service

router = APIRouter()

@router.get("/profile")
async def get_user_profile(
    current_user: dict = Depends(get_current_user)
):
    """Get user profile"""
    return {
        "user_id": current_user["user_id"],
        "email": current_user["email"],
        "credits": current_user["credits"],
        "created_at": "2024-01-01T00:00:00Z",
        "tier": "pro",
        "usage_stats": {
            "total_generations": 42,
            "credits_used": 150,
            "credits_remaining": current_user["credits"]
        }
    }

@router.get("/credits")
async def get_user_credits(
    current_user: dict = Depends(get_current_user)
):
    """Get user credits balance"""
    return {
        "balance": current_user["credits"],
        "currency": "credits",
        "last_updated": datetime.utcnow().isoformat(),
        "usage_this_month": 150,
        "plan_limit": 1000
    }

@router.get("/history")
async def get_generation_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """Get user generation history"""
    history = await history_service.get_user_history(
        user_id=current_user["user_id"],
        page=page,
        per_page=per_page,
        status=status
    )
    
    return history