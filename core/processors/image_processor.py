"""
Core image processing logic shared between platforms
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import FluxPipeline, DiffusionPipeline, ControlNetModel
from diffusers.pipelines.flux import FluxControlNetPipeline
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0
import gc

from ..models.generation import ImageGenerationRequest, ImageGenerationResponse, GenerationStatus
from ..models.enhancement import ModelType
from .prompt_enhancer import PromptEnhancer

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image generation with prompt enhancement"""
    
    def __init__(self, model_manager, output_dir: Path, prompt_enhancer: Optional[PromptEnhancer] = None):
        self.model_manager = model_manager
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_enhancer = prompt_enhancer or PromptEnhancer()
        self._pipelines = {}  # Cache for loaded pipelines
        self._compiled_models = {}  # Cache for compiled models
        
    async def generate(self, request: ImageGenerationRequest, progress_callback=None) -> ImageGenerationResponse:
        """
        Generate an image based on the request
        
        Args:
            request: Image generation request
            progress_callback: Optional callback for progress updates
            
        Returns:
            Image generation response
        """
        request_id = str(uuid.uuid4())
        response = ImageGenerationResponse(
            request_id=request_id,
            status=GenerationStatus.IN_PROGRESS,
            created_at=datetime.utcnow().isoformat()
        )
        
        try:
            # Update progress
            if progress_callback:
                progress_callback(0, "Starting image generation...")
                
            # Enhance prompt if enabled
            enhanced_prompt = request.prompt
            if request.use_enhancement:
                model_type = self._get_model_type(request.model)
                enhanced_prompt = await self.prompt_enhancer.enhance(
                    request.prompt,
                    model_type,
                    request.enhancement_fields
                )
                response.enhanced_prompt = enhanced_prompt
                
            if progress_callback:
                progress_callback(10, "Loading model...")
                
            # Load the model pipeline
            pipeline = await self._load_pipeline(request.model)
            
            if progress_callback:
                progress_callback(30, "Generating image...")
                
            # Prepare generation parameters
            gen_params = self._prepare_generation_params(request, enhanced_prompt)
            
            # Generate the image
            with torch.inference_mode():
                result = pipeline(**gen_params)
                
            # Extract image
            if hasattr(result, 'images'):
                image = result.images[0]
            else:
                image = result
                
            if progress_callback:
                progress_callback(90, "Saving image...")
                
            # Save the image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}_{request_id}.png"
            image_path = self.output_dir / filename
            image.save(image_path)
            
            # Update response
            response.status = GenerationStatus.COMPLETED
            response.completed_at = datetime.utcnow().isoformat()
            response.image_path = image_path
            response.metadata = {
                "model": request.model,
                "prompt": request.prompt,
                "enhanced_prompt": enhanced_prompt,
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "guidance_scale": request.guidance_scale,
                "seed": gen_params.get("generator").initial_seed() if "generator" in gen_params else None
            }
            
            if progress_callback:
                progress_callback(100, "Image generation complete!")
                
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            response.status = GenerationStatus.FAILED
            response.error = str(e)
            response.completed_at = datetime.utcnow().isoformat()
            
        return response
        
    async def _load_pipeline(self, model_id: str):
        """Load the model pipeline with GGUF quantization and torch.compile optimization"""
        if model_id in self._pipelines:
            return self._pipelines[model_id]
            
        # Load with proper GGUF configuration
        pipeline = await self._load_gguf_pipeline(model_id)
        
        # Apply torch.compile optimization for FLUX models
        if "flux" in model_id.lower():
            pipeline = self._apply_torch_compile(pipeline)
            
        # Enable memory-efficient attention
        pipeline.enable_attention_slicing()
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except ImportError:
                pass  # xformers not available
                
        self._pipelines[model_id] = pipeline
        return pipeline
        
    async def _load_gguf_pipeline(self, model_id: str):
        """Load FLUX pipeline with GGUF quantization"""
        # Determine GGUF variant based on available VRAM
        gguf_variant = self._select_gguf_variant()
        
        # Check if this is a GGUF model
        if "gguf" in model_id.lower():
            # Load GGUF quantized model
            pipeline = await asyncio.to_thread(
                self._load_gguf_model,
                model_id,
                gguf_variant
            )
        else:
            # Load standard model
            pipeline = await asyncio.to_thread(
                self.model_manager.load_image_model,
                model_id
            )
            
        return pipeline
        
    def _select_gguf_variant(self) -> str:
        """Select GGUF quantization level based on available VRAM"""
        try:
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                if vram_gb >= 24:
                    return "Q8_0"  # Highest quality
                elif vram_gb >= 16:
                    return "Q6_K"  # Balanced
                elif vram_gb >= 12:
                    return "Q4_K_M"  # Medium compression
                else:
                    return "Q4_K_S"  # High compression
            else:
                return "Q4_K_S"  # CPU fallback
        except Exception:
            return "Q6_K"  # Safe default
            
    def _load_gguf_model(self, model_id: str, gguf_variant: str):
        """Load GGUF quantized model"""
        try:
            # Map model_id to GGUF repo
            gguf_repo = self._get_gguf_repo(model_id)
            
            # Load with specific quantization
            pipeline = FluxPipeline.from_pretrained(
                gguf_repo,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                variant=gguf_variant,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
            
            return pipeline
            
        except Exception as e:
            logger.warning(f"Failed to load GGUF variant {gguf_variant}, falling back to standard: {e}")
            # Fallback to standard model
            return self.model_manager.load_image_model(model_id)
            
    def _get_gguf_repo(self, model_id: str) -> str:
        """Map model ID to GGUF repository"""
        gguf_repos = {
            "flux-1-dev": "city96/FLUX.1-dev-gguf",
            "flux-1-schnell": "city96/FLUX.1-schnell-gguf",
        }
        
        for key, repo in gguf_repos.items():
            if key in model_id.lower():
                return repo
                
        # Default fallback
        return model_id
        
    def _apply_torch_compile(self, pipeline):
        """Apply torch.compile optimization to FLUX pipeline"""
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available, skipping optimization")
            return pipeline
            
        try:
            # Only compile the transformer component for best results
            if hasattr(pipeline, 'transformer'):
                logger.info("Applying torch.compile optimization to FLUX transformer")
                pipeline.transformer = torch.compile(
                    pipeline.transformer,
                    mode="max-autotune",
                    fullgraph=True,
                    dynamic=False
                )
                
                # Store compiled model reference
                model_key = f"{pipeline.__class__.__name__}_transformer"
                self._compiled_models[model_key] = pipeline.transformer
                
        except Exception as e:
            logger.warning(f"torch.compile optimization failed, continuing without: {e}")
            
        return pipeline
        
    async def _load_controlnet_pipeline(self, model_id: str, controlnet_type: str):
        """Load FLUX pipeline with ControlNet support"""
        try:
            # Load ControlNet model
            controlnet = ControlNetModel.from_pretrained(
                f"InstantX/FLUX.1-dev-Controlnet-{controlnet_type}",
                torch_dtype=torch.bfloat16
            )
            
            # Load base pipeline
            base_pipeline = await self._load_gguf_pipeline(model_id)
            
            # Create ControlNet pipeline
            controlnet_pipeline = FluxControlNetPipeline(
                transformer=base_pipeline.transformer,
                scheduler=base_pipeline.scheduler,
                vae=base_pipeline.vae,
                text_encoder=base_pipeline.text_encoder,
                text_encoder_2=base_pipeline.text_encoder_2,
                tokenizer=base_pipeline.tokenizer,
                tokenizer_2=base_pipeline.tokenizer_2,
                controlnet=controlnet,
            )
            
            # Apply optimizations
            if "flux" in model_id.lower():
                controlnet_pipeline = self._apply_torch_compile(controlnet_pipeline)
                
            return controlnet_pipeline
            
        except Exception as e:
            logger.warning(f"Failed to load ControlNet {controlnet_type}, falling back to base pipeline: {e}")
            return await self._load_pipeline(model_id)
            
    def _preprocess_control_image(self, control_image, controlnet_type: str):
        """Preprocess control image based on ControlNet type"""
        try:
            import cv2
            import numpy as np
            
            # Convert PIL to numpy
            image_np = np.array(control_image.convert('RGB'))
            
            if controlnet_type == "canny":
                # Canny edge detection
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                control_np = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                
            elif controlnet_type == "depth":
                # Use depth estimation (placeholder - would use actual depth model)
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                # Simple depth approximation from grayscale
                control_np = np.stack([gray, gray, gray], axis=2)
                
            elif controlnet_type == "pose":
                # Pose detection (placeholder - would use actual pose model)
                control_np = image_np  # Pass through for now
                
            else:
                control_np = image_np
                
            # Convert back to PIL
            from PIL import Image
            return Image.fromarray(control_np.astype('uint8'))
            
        except Exception as e:
            logger.warning(f"Control image preprocessing failed: {e}")
            return control_image
        
    def _get_model_type(self, model_id: str) -> ModelType:
        """Map model ID to ModelType enum"""
        model_id_lower = model_id.lower()
        
        if "flux" in model_id_lower:
            if "schnell" in model_id_lower:
                return ModelType.FLUX_1_SCHNELL
            return ModelType.FLUX_1_DEV
        elif "hunyuan3d" in model_id_lower:
            if "mini" in model_id_lower:
                return ModelType.HUNYUAN_3D_MINI
            elif "2.0" in model_id_lower:
                return ModelType.HUNYUAN_3D_20
            return ModelType.HUNYUAN_3D_21
        elif "sdxl" in model_id_lower:
            return ModelType.SDXL
        else:
            return ModelType.SD15
            
    def _prepare_generation_params(self, request: ImageGenerationRequest, prompt: str) -> Dict[str, Any]:
        """Prepare parameters for the generation pipeline"""
        params = {
            "prompt": prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance_scale,
        }
        
        if request.negative_prompt:
            params["negative_prompt"] = request.negative_prompt
            
        if request.seed is not None:
            params["generator"] = torch.Generator().manual_seed(request.seed)
            
        if request.scheduler:
            params["scheduler"] = request.scheduler
            
        # Apply Distilled CFG for FLUX models
        if "flux" in request.model.lower():
            if "schnell" in request.model.lower():
                # FLUX.1-schnell uses guidance_scale=0 (no CFG)
                params["guidance_scale"] = 0.0
                # Reduce steps for schnell variant
                params["num_inference_steps"] = min(params["num_inference_steps"], 4)
            else:
                # FLUX.1-dev uses CFG=1 with Distilled CFG Scale 2.5-3.5
                params["guidance_scale"] = 1.0
                if hasattr(request, 'distilled_cfg_scale'):
                    params["distilled_cfg_scale"] = request.distilled_cfg_scale
                else:
                    # Use guidance_scale as distilled CFG scale for FLUX.1-dev
                    params["distilled_cfg_scale"] = request.guidance_scale or 3.5
                    params["guidance_scale"] = 1.0  # Override CFG to 1.0
                    
        # Add ControlNet parameters if available
        if hasattr(request, 'control_image') and request.control_image:
            params["image"] = request.control_image
            params["controlnet_conditioning_scale"] = getattr(request, 'controlnet_strength', 1.0)
        
        return params
        
    def validate_request(self, request: ImageGenerationRequest) -> Tuple[bool, Optional[str]]:
        """
        Validate an image generation request
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check model availability
        if not self.model_manager.is_model_available(request.model):
            return False, f"Model {request.model} is not available"
            
        # Check resolution limits
        total_pixels = request.width * request.height
        if total_pixels > 2048 * 2048:
            return False, "Resolution too high (max 2048x2048)"
            
        # Check step limits
        if request.steps > 150:
            return False, "Too many inference steps (max 150)"
            
        return True, None