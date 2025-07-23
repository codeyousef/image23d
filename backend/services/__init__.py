"""
Backend services initialization
"""

from backend.services.model_service import ModelService
from backend.services.queue_service import QueueService
from backend.services.storage_service import StorageService
from backend.services.credit_service import CreditService

# Service instances
model_service = None
queue_service = None
storage_service = None
credit_service = None

async def init_services():
    """Initialize all backend services"""
    global model_service, queue_service, storage_service, credit_service
    
    # Initialize services
    model_service = ModelService()
    queue_service = QueueService()
    storage_service = StorageService()
    credit_service = CreditService()
    
    # Initialize each service
    await model_service.initialize()
    await queue_service.initialize()
    await storage_service.initialize()
    await credit_service.initialize()
    
    print("✅ All services initialized")

def get_model_service() -> ModelService:
    return model_service

def get_queue_service() -> QueueService:
    return queue_service

def get_storage_service() -> StorageService:
    return storage_service

def get_credit_service() -> CreditService:
    return credit_service