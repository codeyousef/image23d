"""
Backend services initialization
"""

try:
    from backend.services.model_service import ModelService
    from backend.services.queue_service import QueueService  
    from backend.services.credit_service import CreditService
except ImportError:
    # Fallback to simple services for testing
    from backend.services.simple_model_service import ModelService
    from backend.services.simple_queue_service import QueueService
    from backend.services.simple_credit_service import CreditService

from backend.services.storage_service import StorageService
from backend.services.job_service import job_service
from backend.services.history_service import history_service

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
    await job_service.initialize()
    await history_service.initialize()
    
    print("✅ All services initialized")

def get_model_service() -> ModelService:
    return model_service

def get_queue_service() -> QueueService:
    return queue_service

def get_storage_service() -> StorageService:
    return storage_service

def get_credit_service() -> CreditService:
    return credit_service