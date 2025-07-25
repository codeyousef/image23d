"""
History service for user generation history
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from backend.models.generation import JobStatus

logger = logging.getLogger(__name__)


class HistoryService:
    """Service for managing user generation history"""
    
    def __init__(self):
        # In-memory storage for testing
        # In production, this would use a database
        self.user_history: Dict[str, List[Dict[str, Any]]] = {}
        
    async def initialize(self):
        """Initialize the history service"""
        logger.info("History service initialized")
        
    async def get_user_history(self, user_id: str, page: int = 1, per_page: int = 20, 
                             status: Optional[str] = None) -> Dict[str, Any]:
        """Get paginated user history"""
        user_jobs = self.user_history.get(user_id, [])
        
        # Filter by status if provided
        if status:
            user_jobs = [job for job in user_jobs if job.get("status") == status]
            
        # Sort by creation time (newest first)  
        user_jobs.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)
        
        # Calculate pagination
        total = len(user_jobs)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        items = user_jobs[start_idx:end_idx]
        
        return {
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "has_next": end_idx < total,
            "has_prev": page > 1
        }
        
    async def add_job_to_history(self, user_id: str, job_data: Dict[str, Any]):
        """Add a job to user's history"""
        if user_id not in self.user_history:
            self.user_history[user_id] = []
            
        # Create history entry
        history_entry = {
            "job_id": job_data.get("job_id"),
            "type": job_data.get("type"),
            "status": job_data.get("status"),
            "prompt": job_data.get("params", {}).get("prompt"),
            "model": job_data.get("params", {}).get("model"),
            "created_at": job_data.get("created_at"),
            "completed_at": job_data.get("completed_at"),
            "credits_used": job_data.get("credits_used"),
            "output_url": job_data.get("result", {}).get("output_url") if job_data.get("result") else None,
            "thumbnail_url": job_data.get("result", {}).get("thumbnail_url") if job_data.get("result") else None,
            "metadata": job_data.get("result", {}).get("metadata", {}) if job_data.get("result") else {}
        }
        
        # Add to history
        self.user_history[user_id].append(history_entry)
        
        # Keep only last 1000 entries per user
        if len(self.user_history[user_id]) > 1000:
            self.user_history[user_id] = self.user_history[user_id][-1000:]
            
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user generation statistics"""
        user_jobs = self.user_history.get(user_id, [])
        
        if not user_jobs:
            return {
                "total_generations": 0,
                "completed_generations": 0,
                "failed_generations": 0,
                "total_credits_used": 0,
                "generations_by_type": {},
                "generations_by_month": {}
            }
            
        # Calculate stats
        total = len(user_jobs)
        completed = len([j for j in user_jobs if j.get("status") == JobStatus.COMPLETED])
        failed = len([j for j in user_jobs if j.get("status") == JobStatus.FAILED])
        total_credits = sum(j.get("credits_used", 0) or 0 for j in user_jobs)
        
        # Group by type
        by_type = {}
        for job in user_jobs:
            job_type = job.get("type", "unknown")
            by_type[job_type] = by_type.get(job_type, 0) + 1
            
        # Group by month
        by_month = {}
        for job in user_jobs:
            created_at = job.get("created_at")
            if created_at:
                if isinstance(created_at, str):
                    # Handle string dates
                    try:
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except:
                        continue
                        
                month_key = created_at.strftime("%Y-%m")
                by_month[month_key] = by_month.get(month_key, 0) + 1
                
        return {
            "total_generations": total,
            "completed_generations": completed,
            "failed_generations": failed,
            "total_credits_used": total_credits,
            "generations_by_type": by_type,
            "generations_by_month": by_month
        }
        
    async def delete_job_from_history(self, user_id: str, job_id: str) -> bool:
        """Delete a job from user's history"""
        if user_id not in self.user_history:
            return False
            
        user_jobs = self.user_history[user_id]
        for i, job in enumerate(user_jobs):
            if job.get("job_id") == job_id:
                del user_jobs[i]
                return True
                
        return False
        
    async def clear_user_history(self, user_id: str) -> bool:
        """Clear all history for a user"""
        if user_id in self.user_history:
            self.user_history[user_id] = []
            return True
        return False


# Global history service instance
history_service = HistoryService()