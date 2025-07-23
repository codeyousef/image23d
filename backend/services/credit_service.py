"""
Credit service for managing user credits and billing
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal

class CreditService:
    """
    Service for managing user credits and transactions
    """
    
    def __init__(self):
        # In-memory storage for demo
        self.user_credits = {}
        self.transactions = []
        self.credit_packages = {
            "starter": {"credits": 500, "price_usd": 5.00},
            "standard": {"credits": 2000, "price_usd": 18.00},
            "pro": {"credits": 5000, "price_usd": 40.00},
            "enterprise": {"credits": 20000, "price_usd": 150.00}
        }
        
    async def initialize(self):
        """Initialize credit service"""
        # Would connect to payment provider and database
        pass
        
    async def get_user_credits(self, user_id: str) -> int:
        """Get user's current credit balance"""
        return self.user_credits.get(user_id, 50)  # 50 free credits for new users
        
    async def add_credits(
        self,
        user_id: str,
        amount: int,
        transaction_type: str,
        reference: Optional[str] = None
    ) -> bool:
        """Add credits to user account"""
        current = await self.get_user_credits(user_id)
        self.user_credits[user_id] = current + amount
        
        # Record transaction
        self.transactions.append({
            "id": f"txn_{len(self.transactions) + 1}",
            "user_id": user_id,
            "type": transaction_type,
            "amount": amount,
            "balance_after": self.user_credits[user_id],
            "reference": reference,
            "created_at": datetime.utcnow()
        })
        
        return True
        
    async def deduct_credits(
        self,
        user_id: str,
        amount: int,
        job_id: str
    ) -> bool:
        """Deduct credits for a job"""
        current = await self.get_user_credits(user_id)
        
        if current < amount:
            return False
            
        self.user_credits[user_id] = current - amount
        
        # Record transaction
        self.transactions.append({
            "id": f"txn_{len(self.transactions) + 1}",
            "user_id": user_id,
            "type": "usage",
            "amount": -amount,
            "balance_after": self.user_credits[user_id],
            "reference": job_id,
            "created_at": datetime.utcnow()
        })
        
        return True
        
    async def get_transaction_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get user's transaction history"""
        user_transactions = [
            t for t in self.transactions
            if t["user_id"] == user_id
        ]
        
        # Sort by date (newest first)
        user_transactions.sort(
            key=lambda x: x["created_at"],
            reverse=True
        )
        
        return user_transactions[offset:offset + limit]
        
    async def get_credit_packages(self) -> Dict[str, Any]:
        """Get available credit packages"""
        return self.credit_packages
        
    async def process_payment(
        self,
        user_id: str,
        package_id: str,
        payment_method: str,
        payment_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a credit purchase"""
        # In production, this would integrate with payment provider
        
        package = self.credit_packages.get(package_id)
        if not package:
            return {
                "success": False,
                "error": "Invalid package"
            }
            
        # Mock payment processing
        if payment_method == "stripe":
            # Would call Stripe API
            payment_id = f"pi_mock_{user_id}_{package_id}"
        else:
            return {
                "success": False,
                "error": "Unsupported payment method"
            }
            
        # Add credits
        await self.add_credits(
            user_id=user_id,
            amount=package["credits"],
            transaction_type="purchase",
            reference=payment_id
        )
        
        return {
            "success": True,
            "payment_id": payment_id,
            "credits_added": package["credits"],
            "new_balance": await self.get_user_credits(user_id)
        }
        
    async def get_usage_stats(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        usage_transactions = [
            t for t in self.transactions
            if t["user_id"] == user_id
            and t["type"] == "usage"
            and t["created_at"] >= cutoff_date
        ]
        
        total_used = sum(abs(t["amount"]) for t in usage_transactions)
        
        # Group by day
        daily_usage = {}
        for txn in usage_transactions:
            date_key = txn["created_at"].strftime("%Y-%m-%d")
            if date_key not in daily_usage:
                daily_usage[date_key] = 0
            daily_usage[date_key] += abs(txn["amount"])
            
        return {
            "total_credits_used": total_used,
            "daily_usage": daily_usage,
            "average_daily": total_used / days if days > 0 else 0,
            "current_balance": await self.get_user_credits(user_id)
        }
        
    async def estimate_credits_needed(
        self,
        job_type: str,
        parameters: Dict[str, Any]
    ) -> int:
        """Estimate credits needed for a job"""
        # Base costs
        base_costs = {
            "image": 5,
            "3d": 45,
            "video": 30,
            "face_swap": 8
        }
        
        credits = base_costs.get(job_type, 10)
        
        # Adjust for parameters
        if job_type == "image":
            # Higher resolution costs more
            if parameters.get("width", 1024) > 1024:
                credits *= 2
            # More steps cost more
            if parameters.get("steps", 20) > 50:
                credits = int(credits * 1.5)
                
        elif job_type == "3d":
            # Quality presets
            quality_multipliers = {
                "draft": 1.0,
                "standard": 1.5,
                "high": 2.0,
                "ultra": 3.0
            }
            quality = parameters.get("quality_preset", "standard")
            credits = int(credits * quality_multipliers.get(quality, 1.5))
            
        elif job_type == "video":
            # Per second of video
            duration = parameters.get("duration", 1)
            credits = credits * duration
            
        return credits