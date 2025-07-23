"""
Generation endpoints for images, 3D models, and videos
"""

from typing import Optional, Dict, Any
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from backend.api.auth import get_current_user
from backend.services.queue_service import QueueService
from backend.models.generation import GenerationJob, JobStatus
from core.models.generation import (
    ImageGenerationRequest,
    ThreeDGenerationRequest,
    VideoGenerationRequest
)
from core.services.gpu_orchestrator import ExecutionMode

router = APIRouter()

# Response models
class GenerationResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    estimated_cost: Optional[float] = None
    estimated_time: Optional[int] = None

class ImageGenerationBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: Optional[str] = None
    model: str = "FLUX.1-schnell"
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    steps: int = Field(20, ge=1, le=150)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    seed: Optional[int] = None
    enhancement_fields: Optional[Dict[str, Any]] = None
    use_enhancement: bool = True
    execution_mode: ExecutionMode = ExecutionMode.AUTO

class ThreeDGenerationBody(BaseModel):
    prompt: Optional[str] = None
    image_url: Optional[str] = None
    model: str = "hunyuan3d-21"
    quality_preset: str = "standard"
    export_formats: list[str] = ["glb"]
    remove_background: bool = True
    enhancement_fields: Optional[Dict[str, Any]] = None
    use_enhancement: bool = True
    execution_mode: ExecutionMode = ExecutionMode.AUTO

# Endpoints
@router.post("/image", response_model=GenerationResponse, summary="Generate Image", description="""
Generate a high-quality image from a text prompt using FLUX models.

## Features:
- Multiple FLUX model variants (schnell, dev, pro)
- Prompt enhancement with local LLM
- Custom resolution up to 2048x2048
- Advanced parameters (steps, guidance, seed)
- Enhancement fields for style, lighting, mood, etc.

## Credit Cost:
- Standard (≤1024px): 10 credits
- High Resolution (>1024px): 20 credits

## Execution Modes:
- **auto**: Automatically choose between local and serverless
- **local**: Force local GPU execution
- **serverless**: Force RunPod serverless execution
""")
async def generate_image(
    body: ImageGenerationBody,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    queue_service: QueueService = Depends()
):
    """Generate an image from text prompt"""
    
    # Check user credits
    required_credits = 10  # Base cost
    if body.width > 1024 or body.height > 1024:
        required_credits = 20  # Higher resolution costs more
        
    if current_user["credits"] < required_credits:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Required: {required_credits}, Available: {current_user['credits']}"
        )
    
    # Create generation request
    request = ImageGenerationRequest(
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        model=body.model,
        width=body.width,
        height=body.height,
        steps=body.steps,
        guidance_scale=body.guidance_scale,
        seed=body.seed,
        enhancement_fields=body.enhancement_fields or {},
        use_enhancement=body.use_enhancement
    )
    
    # Create job
    job_id = str(uuid4())
    job = GenerationJob(
        id=job_id,
        user_id=current_user["id"],
        type="image",
        request=request.dict(),
        execution_mode=body.execution_mode,
        credits_required=required_credits
    )
    
    # Estimate cost and time
    estimate = await queue_service.estimate_job(job)
    
    # Queue the job
    await queue_service.enqueue(job)
    
    # Start processing in background
    background_tasks.add_task(queue_service.process_job, job_id)
    
    return GenerationResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Image generation job queued",
        estimated_cost=estimate.get("cost_usd"),
        estimated_time=estimate.get("time_seconds")
    )

@router.post("/3d", response_model=GenerationResponse, summary="Generate 3D Model", description="""
Generate a high-quality 3D model from text prompt or image using HunyuanVideo 3D.

## Features:
- Text-to-3D and Image-to-3D generation
- Multiple quality presets (draft, standard, high, ultra)
- Multiple export formats (GLB, OBJ, FBX, USDZ, etc.)
- Automatic background removal
- Texture and mesh optimization

## Credit Cost:
- Draft: 50 credits
- Standard: 80 credits
- High: 120 credits
- Ultra: 200 credits

## Input Options:
- **Text Prompt**: Describe the 3D model you want
- **Image URL**: Provide an image to convert to 3D
""")
async def generate_3d(
    body: ThreeDGenerationBody,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    queue_service: QueueService = Depends()
):
    """Generate a 3D model from text or image"""
    
    # Validate input
    if not body.prompt and not body.image_url:
        raise HTTPException(
            status_code=400,
            detail="Either prompt or image_url must be provided"
        )
    
    # Check user credits
    quality_credits = {
        "draft": 50,
        "standard": 80,
        "high": 120,
        "ultra": 200
    }
    required_credits = quality_credits.get(body.quality_preset, 80)
    
    if current_user["credits"] < required_credits:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Required: {required_credits}, Available: {current_user['credits']}"
        )
    
    # Create generation request
    request = ThreeDGenerationRequest(
        prompt=body.prompt,
        input_image=body.image_url,  # Will be downloaded if provided
        model=body.model,
        quality_preset=body.quality_preset,
        export_formats=body.export_formats,
        remove_background=body.remove_background,
        enhancement_fields=body.enhancement_fields or {},
        use_enhancement=body.use_enhancement and bool(body.prompt)
    )
    
    # Create job
    job_id = str(uuid4())
    job = GenerationJob(
        id=job_id,
        user_id=current_user["id"],
        type="3d",
        request=request.dict(),
        execution_mode=body.execution_mode,
        credits_required=required_credits
    )
    
    # Estimate cost and time
    estimate = await queue_service.estimate_job(job)
    
    # Queue the job
    await queue_service.enqueue(job)
    
    # Start processing in background
    background_tasks.add_task(queue_service.process_job, job_id)
    
    return GenerationResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="3D generation job queued",
        estimated_cost=estimate.get("cost_usd"),
        estimated_time=estimate.get("time_seconds")
    )

@router.post("/enhance-prompt")
async def enhance_prompt(
    prompt: str,
    model_type: str = "image",
    fields: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(get_current_user)
):
    """Enhance a prompt using LLM"""
    
    # This is a lightweight operation, minimal credit cost
    if current_user["credits"] < 1:
        raise HTTPException(
            status_code=402,
            detail="Insufficient credits for prompt enhancement"
        )
    
    # Use prompt enhancer from core
    from core.processors.prompt_enhancer import PromptEnhancer
    
    enhancer = PromptEnhancer()
    enhanced = await enhancer.enhance(
        prompt=prompt,
        model_type=model_type,
        fields=fields or {}
    )
    
    return {
        "original": prompt,
        "enhanced": enhanced,
        "credits_used": 1
    }

@router.get("/models")
async def get_available_models(
    current_user: dict = Depends(get_current_user)
):
    """Get list of available models"""
    
    from core.config import FLUX_MODELS, ALL_3D_MODELS
    
    return {
        "image": [
            {
                "id": model_id,
                "name": config["name"],
                "description": config["description"],
                "vram_required": config["vram_required"]
            }
            for model_id, config in FLUX_MODELS.items()
        ],
        "3d": [
            {
                "id": model_id,
                "name": config["name"],
                "description": config["description"],
                "vram_required": config["vram_required"]
            }
            for model_id, config in ALL_3D_MODELS.items()
        ]
    }

@router.get("/enhancement-fields/{model_type}")
async def get_enhancement_fields(
    model_type: str,
    current_user: dict = Depends(get_current_user)
):
    """Get enhancement fields for a model type"""
    
    from core.config import FLUX_ENHANCEMENT_FIELDS, HUNYUAN3D_ENHANCEMENT_FIELDS, SPARC3D_ENHANCEMENT_FIELDS, HI3DGEN_ENHANCEMENT_FIELDS
    
    if model_type == "image":
        return FLUX_ENHANCEMENT_FIELDS
    elif model_type == "3d":
        # Return combined 3D enhancement fields
        return {
            **HUNYUAN3D_ENHANCEMENT_FIELDS,
            **SPARC3D_ENHANCEMENT_FIELDS,
            **HI3DGEN_ENHANCEMENT_FIELDS
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type: {model_type}"
        )