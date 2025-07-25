"""
Webhooks API v1 endpoints
"""

import uuid
from typing import List
from fastapi import APIRouter, Depends
from pydantic import BaseModel, HttpUrl

from backend.api.middleware.auth import get_current_user

router = APIRouter()

class WebhookRequest(BaseModel):
    url: HttpUrl
    events: List[str]
    secret: str

@router.post("", status_code=201)
async def create_webhook(
    webhook: WebhookRequest,
    current_user: dict = Depends(get_current_user)
):
    """Register a webhook for job updates"""
    webhook_id = f"webhook-{uuid.uuid4().hex[:8]}"
    
    # Mock webhook creation
    return {
        "webhook_id": webhook_id,
        "url": str(webhook.url),
        "events": webhook.events,
        "created_at": "2024-01-01T00:00:00Z",
        "status": "active"
    }