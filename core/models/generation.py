"""
Pydantic models for generation requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path

class GenerationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BaseGenerationRequest(BaseModel):
    """Base class for all generation requests"""
    prompt: str = Field(..., description="The generation prompt")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    enhancement_fields: Dict[str, Any] = Field(default_factory=dict, description="UI enhancement fields")
    use_enhancement: bool = Field(True, description="Whether to use prompt enhancement")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
class ImageGenerationRequest(BaseGenerationRequest):
    """Request model for image generation"""
    model: str = Field(..., description="Model ID to use")
    width: int = Field(1024, ge=256, le=2048, description="Image width")
    height: int = Field(1024, ge=256, le=2048, description="Image height")
    steps: int = Field(20, ge=1, le=150, description="Number of inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    scheduler: Optional[str] = Field(None, description="Scheduler type")
    
class ThreeDGenerationRequest(BaseGenerationRequest):
    """Request model for 3D generation"""
    model: str = Field("hunyuan3d-21", description="3D model to use")
    image_model: Optional[str] = Field(None, description="Image model for text-to-3D generation")
    input_image: Optional[str] = Field(None, description="Optional input image path")
    quality_preset: str = Field("standard", description="Quality preset")
    num_views: int = Field(6, ge=4, le=12, description="Number of views to generate")
    mesh_resolution: int = Field(512, description="Mesh resolution")
    texture_resolution: int = Field(2048, description="Texture resolution")
    export_formats: List[str] = Field(["glb"], description="Export formats")
    remove_background: bool = Field(True, description="Remove background from input")
    
    # Enhanced processing options
    auto_center: bool = Field(True, description="Auto center object in frame")
    view_distance: float = Field(2.0, ge=1.0, le=3.0, description="Camera distance multiplier")
    pbr_materials: bool = Field(False, description="Generate PBR materials")
    depth_enhancement: bool = Field(True, description="Enable depth map enhancement")
    normal_enhancement: bool = Field(True, description="Enable normal map enhancement") 
    multiview_consistency: bool = Field(True, description="Enforce multi-view consistency")
    
    # Performance-critical parameters
    mesh_decode_resolution: int = Field(64, ge=32, le=128, description="SDF decoding resolution (affects quality vs speed)")
    mesh_decode_batch_size: Optional[int] = Field(None, description="Batch size for mesh decoding (auto if None)")
    paint_max_num_view: int = Field(6, ge=4, le=9, description="Max views for texture painting")  
    paint_resolution: int = Field(512, ge=256, le=768, description="Paint pipeline resolution")
    render_size: int = Field(1024, ge=512, le=2048, description="Texture rendering size")
    texture_size: int = Field(1024, ge=512, le=4096, description="Final texture size")
    
class VideoGenerationRequest(BaseGenerationRequest):
    """Request model for video generation"""
    model: str = Field(..., description="Video model to use")
    input_image: Optional[str] = Field(None, description="Optional input image")
    duration: float = Field(4.0, ge=1.0, le=10.0, description="Video duration in seconds")
    fps: int = Field(24, description="Frames per second")
    motion_scale: float = Field(1.0, ge=0.1, le=3.0, description="Motion intensity")
    
class GenerationResponse(BaseModel):
    """Base response model"""
    request_id: str = Field(..., description="Unique request ID")
    status: GenerationStatus = Field(..., description="Generation status")
    created_at: str = Field(..., description="Creation timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")
    
class ImageGenerationResponse(GenerationResponse):
    """Response model for image generation"""
    image_path: Optional[Path] = Field(None, description="Path to generated image")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    enhanced_prompt: Optional[str] = Field(None, description="Enhanced prompt used")
    
class ThreeDGenerationResponse(GenerationResponse):
    """Response model for 3D generation"""
    model_path: Optional[Path] = Field(None, description="Path to generated 3D model")
    preview_images: List[Path] = Field(default_factory=list, description="Preview image paths")
    export_paths: Dict[str, Path] = Field(default_factory=dict, description="Export format to path mapping")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    
class VideoGenerationResponse(GenerationResponse):
    """Response model for video generation"""
    video_path: Optional[Path] = Field(None, description="Path to generated video")
    preview_gif: Optional[Path] = Field(None, description="Preview GIF path")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")