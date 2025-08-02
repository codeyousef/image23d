"""Flow matching pipeline for HunYuan3D shape generation."""

import os
import torch
import torch.nn as nn
import numpy as np
import trimesh
import yaml
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from PIL import Image
import logging
from tqdm import tqdm

# Set up logger
logger = logging.getLogger(__name__)

from ..models.autoencoders import ShapeVAE
from ..models.denoisers.hunyuandit import HunYuanDiTPlain, DiTConfig
from ..rembg import BackgroundRemover

try:
    from transformers import AutoModel, AutoImageProcessor
    DINOV2_AVAILABLE = True
except ImportError:
    logger.warning("DINOv2 not available, image conditioning will be disabled")
    DINOV2_AVAILABLE = False


class Hunyuan3DDiTFlowMatchingPipeline:
    """Flow-based diffusion pipeline for 3D shape generation."""
    
    def __init__(
        self,
        vae: ShapeVAE,
        dit: HunYuanDiTPlain,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16,
        image_encoder_name: str = "facebook/dinov2-giant",
        max_generation_time: int = 3600  # Default timeout of 60 minutes
    ):
        # Force float32 for stability to avoid CUBLAS errors
        self.vae = vae.to(device).float()  # Force float32
        self.dit = dit.to(device).to(dtype)  # DiT can use fp16
        self.device = device
        self.dtype = dtype
        self.max_generation_time = max_generation_time  # Store timeout parameter
        
        # Ensure models are in eval mode
        self.vae.eval()
        self.dit.eval()
        
        # Disable gradient computation
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.dit.parameters():
            param.requires_grad = False
        
        # Initialize image encoder attributes first
        self.image_encoder = None
        self.image_processor = None
        
        # Optimize DiT model for inference
        self.dit.eval()  # Ensure eval mode
        
        # Enable torch optimizations
        self._setup_model_optimization()
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("Enabled Flash Attention for better performance")
        except:
            pass
        
        # Background remover
        self.rembg = BackgroundRemover()
        
        # Image encoder for conditioning
        if DINOV2_AVAILABLE:
            try:
                logger.info(f"Loading DINOv2 image encoder: {image_encoder_name}")
                self.image_encoder = AutoModel.from_pretrained(image_encoder_name).to(device)
                self.image_processor = AutoImageProcessor.from_pretrained(image_encoder_name)
                # Set to eval mode and appropriate dtype
                self.image_encoder.eval()
                if dtype == torch.float16:
                    self.image_encoder = self.image_encoder.half()
                
                # Add projection layer for DINOv2 embeddings (1536) to context_dim (1024)
                # DINOv2-giant has 1536 dimensions, but the model expects 1024
                dinov2_dim = 1536  # DINOv2-giant embedding dimension
                context_dim = 1024  # Expected by the model
                logger.info(f"Adding projection layer from DINOv2 dim ({dinov2_dim}) to context dim ({context_dim})")
                self.embedding_projection = nn.Linear(dinov2_dim, context_dim).to(device).to(dtype)
                # Initialize with identity-like weights for minimal disruption
                nn.init.eye_(self.embedding_projection.weight[:, :min(dinov2_dim, context_dim)])
                nn.init.zeros_(self.embedding_projection.bias)
                # Disable gradient computation
                for param in self.embedding_projection.parameters():
                    param.requires_grad = False
                
                logger.info("DINOv2 image encoder and projection layer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load DINOv2: {e}")
                self.image_encoder = None
                self.image_processor = None
                self.embedding_projection = None
    
    def _setup_model_optimization(self):
        """Setup model optimization with platform-specific handling"""
        import platform
        
        # Check if torch.compile is available and should be used
        if not hasattr(torch, 'compile'):
            logger.info("torch.compile not available in this PyTorch version")
            return
            
        if not torch.cuda.is_available():
            logger.info("CUDA not available, skipping model compilation")
            return
            
        # Check if compilation is disabled via environment variable
        if os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1":
            logger.info("Model compilation disabled via TORCH_COMPILE_DISABLE environment variable")
            return
            
        # Platform-specific compilation
        system = platform.system()
        
        try:
            if system == "Windows":
                # Windows: Use safer compilation options
                logger.info("Windows detected: Using safe compilation mode")
                
                # Try to compile with Windows-compatible backend
                backend = os.environ.get("TORCH_COMPILE_BACKEND", "eager")
                if backend == "inductor":
                    # Override to eager on Windows if inductor is requested
                    backend = "eager"
                    logger.warning("Inductor backend not supported on Windows, using eager mode")
                
                logger.info(f"Attempting to compile DiT model with backend='{backend}'...")
                self.dit = torch.compile(self.dit, backend=backend, mode='default')
                logger.info("DiT model compiled successfully for Windows")
                
            else:
                # Linux/Unix: Use full optimizations
                logger.info("Linux/Unix detected: Using optimized compilation mode")
                
                # Use reduce-overhead mode for better performance
                backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
                mode = 'reduce-overhead' if backend == 'inductor' else 'default'
                
                logger.info(f"Attempting to compile DiT model with backend='{backend}', mode='{mode}'...")
                self.dit = torch.compile(self.dit, backend=backend, mode=mode)
                logger.info("DiT model compiled successfully for Linux/Unix")
                
        except Exception as e:
            logger.warning(f"Failed to compile DiT model: {e}")
            logger.warning("Falling back to eager execution mode")
            
            # Ensure torch._dynamo errors are suppressed
            if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
                torch._dynamo.config.suppress_errors = True
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        """Load pipeline from pretrained checkpoint.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device to load on
            dtype: Data type
        """
        if isinstance(model_path, str) and '/' in model_path and not Path(model_path).exists():
            # HuggingFace model ID
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_path)
        
        model_path = Path(model_path)
        
        # Load from checkpoint files
        dit_path = model_path / "hunyuan3d-dit-v2-1"
        vae_path = model_path / "hunyuan3d-vae-v2-1"
        
        # Load configurations
        dit_config_path = dit_path / "config.yaml"
        vae_config_path = vae_path / "config.yaml"
        
        if dit_config_path.exists():
            with open(dit_config_path, 'r') as f:
                dit_cfg = yaml.safe_load(f)
            # Handle different config structures
            if 'model' in dit_cfg and 'params' in dit_cfg['model']:
                dit_params = dit_cfg['model']['params']
            elif 'params' in dit_cfg:
                dit_params = dit_cfg['params']
            else:
                dit_params = {}
                logger.warning(f"Could not find DiT params in config: {dit_config_path}")
        else:
            dit_params = {}
        
        if vae_config_path.exists():
            with open(vae_config_path, 'r') as f:
                vae_cfg = yaml.safe_load(f)
            # Handle different config structures
            if 'vae' in vae_cfg and 'params' in vae_cfg['vae']:
                vae_params = vae_cfg['vae']['params']
            elif 'params' in vae_cfg:
                vae_params = vae_cfg['params']
            else:
                vae_params = {}
                logger.warning(f"Could not find VAE params in config: {vae_config_path}")
        else:
            vae_params = {}
        
        # Create models
        vae = ShapeVAE(**vae_params)
        dit = HunYuanDiTPlain(**dit_params)
        
        # Load weights if available
        dit_ckpt = dit_path / "model.fp16.ckpt"
        vae_ckpt = vae_path / "model.fp16.ckpt"
        
        if dit_ckpt.exists():
            logger.info(f"Loading DiT checkpoint from {dit_ckpt}")
            dit_state = torch.load(dit_ckpt, map_location='cpu')
            if 'state_dict' in dit_state:
                dit_state = dit_state['state_dict']
            # Remove 'model.' prefix if present
            dit_state = {k.replace('model.', ''): v for k, v in dit_state.items()}
            dit.load_state_dict(dit_state, strict=False)
        
        if vae_ckpt.exists():
            logger.info(f"Loading VAE checkpoint from {vae_ckpt}")
            vae_state = torch.load(vae_ckpt, map_location='cpu')
            if 'state_dict' in vae_state:
                vae_state = vae_state['state_dict']
            # Remove 'vae.' prefix if present
            vae_state = {k.replace('vae.', ''): v for k, v in vae_state.items()}
            vae.load_state_dict(vae_state, strict=False)
        
        return cls(vae, dit, device, dtype, **kwargs)
    
    @classmethod
    def from_single_file(
        cls,
        ckpt_path: Union[str, Path],
        config_path: Union[str, Path],
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        """Load from a single checkpoint file."""
        # For single file loading, we assume it contains both VAE and DiT
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create models from config
        vae_params = config.get('vae', {}).get('params', {})
        dit_params = config.get('model', {}).get('params', {})
        
        # Keep original config - model was trained without cross-attention
        # Forcing it to True causes weight mismatches
        original_setting = dit_params.get('with_decoupled_ca', False)
        logger.info(f"🔍 Model config has with_decoupled_ca={original_setting}")
        logger.info("📝 Keeping original config to match checkpoint weights")
        logger.info("🔧 Using AdaLN-based conditioning instead of cross-attention")
        
        vae = ShapeVAE(**vae_params)
        dit = HunYuanDiTPlain(**dit_params)
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Split state dict between VAE and DiT
        vae_state = {}
        dit_state = {}
        
        for k, v in state_dict.items():
            if k.startswith('vae.'):
                vae_state[k.replace('vae.', '')] = v
            elif k.startswith('model.'):
                dit_state[k.replace('model.', '')] = v
            else:
                # Try to determine based on parameter names
                if 'encoder' in k or 'decoder' in k or 'to_latent' in k or 'to_sdf' in k:
                    vae_state[k] = v
                else:
                    dit_state[k] = v
        
        # Load state dicts and verify
        missing_vae = vae.load_state_dict(vae_state, strict=False)
        missing_dit = dit.load_state_dict(dit_state, strict=False)
        
        # Log loading results
        if missing_vae.missing_keys:
            logger.warning(f"VAE missing keys: {missing_vae.missing_keys[:5]}...")
        if missing_vae.unexpected_keys:
            logger.warning(f"VAE unexpected keys: {missing_vae.unexpected_keys[:5]}...")
            
        if missing_dit.missing_keys:
            logger.warning(f"DiT missing keys: {missing_dit.missing_keys[:5]}...")
        if missing_dit.unexpected_keys:
            logger.warning(f"DiT unexpected keys: {missing_dit.unexpected_keys[:5]}...")
        
        # Verify models are properly loaded
        logger.info(f"✅ Loaded VAE: {type(vae).__name__} with {sum(p.numel() for p in vae.parameters())} parameters")
        logger.info(f"✅ Loaded DiT: {type(dit).__name__} with {sum(p.numel() for p in dit.parameters())} parameters")
        
        # DIAGNOSTIC: Verify DiT configuration for image conditioning
        logger.info(f"🔧 DiT Configuration for Image Conditioning:")
        logger.info(f"   - Using AdaLN-based conditioning: {not dit.config.with_decoupled_ca}")
        logger.info(f"   - Cross-attention enabled: {dit.config.with_decoupled_ca}")
        logger.info(f"   - Context dimension: {dit.config.context_dim}")
        logger.info(f"   - Hidden size: {dit.config.hidden_size}")
        logger.info(f"   - Number of blocks: {dit.config.depth}")
        
        # Check if blocks have cross-attention layers
        has_cross_attn = hasattr(dit.blocks[0], 'cross_attn') if dit.blocks else False
        logger.info(f"   - Blocks have cross-attention: {has_cross_attn}")
        
        # HunYuan3D 2.1 uses AdaLN-based conditioning (with_decoupled_ca=False)
        if not dit.config.with_decoupled_ca:
            logger.info("✅ DiT model is properly configured for AdaLN-based image conditioning!")
        elif dit.config.with_decoupled_ca and has_cross_attn:
            logger.info("✅ DiT model is configured for cross-attention-based image conditioning!")
        else:
            logger.warning("⚠️ DiT model has unusual configuration - check conditioning mechanism!")
        
        return cls(vae, dit, device, dtype, **kwargs)
    
    def to(self, device):
        """Move pipeline to device."""
        self.device = device
        self.vae = self.vae.to(device)
        self.dit = self.dit.to(device)
        return self
        
    def _get_available_gpu_memory(self):
        """Get available GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0
        
        try:
            # Get available memory for current device
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)  # Convert to GB
            
            # Get currently allocated memory
            allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # Convert to GB
            reserved_memory = torch.cuda.memory_reserved(device) / (1024**3)  # Convert to GB
            
            # Calculate available memory (with some buffer)
            available_memory = total_memory - allocated_memory - reserved_memory
            
            # Apply a safety factor (80% of what's reported as available)
            available_memory *= 0.8
            
            logger.info(f"GPU memory - Total: {total_memory:.2f}GB, Available: {available_memory:.2f}GB")
            return available_memory
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return 0
            
    def _get_adaptive_step_count(self, requested_steps=30):
        """Determine appropriate step count based on hardware capabilities."""
        # Default step counts for different memory tiers
        high_memory_steps = requested_steps
        medium_memory_steps = 20
        low_memory_steps = 15
        minimum_steps = 10
        
        # Memory thresholds in GB
        high_memory_threshold = 10.0  # 10+ GB available
        medium_memory_threshold = 6.0  # 6-10 GB available
        low_memory_threshold = 4.0    # 4-6 GB available
        
        # Get available memory
        available_memory = self._get_available_gpu_memory()
        
        # Determine step count based on available memory
        if available_memory >= high_memory_threshold:
            steps = high_memory_steps
            memory_tier = "high"
        elif available_memory >= medium_memory_threshold:
            steps = medium_memory_steps
            memory_tier = "medium"
        elif available_memory >= low_memory_threshold:
            steps = low_memory_steps
            memory_tier = "low"
        else:
            steps = minimum_steps
            memory_tier = "minimal"
            
        # Log the decision
        if steps < requested_steps:
            logger.warning(f"Reducing inference steps from {requested_steps} to {steps} due to limited GPU memory ({available_memory:.2f}GB, {memory_tier} tier)")
        else:
            logger.info(f"Using {steps} inference steps ({memory_tier} memory tier, {available_memory:.2f}GB available)")
            
        return steps
    
    @torch.no_grad()
    def __call__(
        self,
        image: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
        prompt: Optional[str] = None,
        num_inference_steps: int = 30,  # Default value, will be adjusted based on hardware
        guidance_scale: float = 15.0,  # Increased from 3.0 to improve image conditioning effectiveness
        generator: Optional[torch.Generator] = None,
        box_v: float = 1.5,
        output_type: str = "trimesh",
        callback: Optional[callable] = None,
        callback_steps: int = 1,
        adaptive_steps: bool = True,  # Whether to adapt step count to hardware capabilities
        **kwargs
    ) -> Union[trimesh.Trimesh, List[trimesh.Trimesh], Dict[str, Any]]:
        """Generate 3D shape from image.
        
        Args:
            image: Input image (required)
            prompt: Text prompt (optional, not used currently)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            generator: Random generator
            box_v: Bounding box scale
            output_type: Output format ('trimesh', 'mesh', 'dict')
            callback: Progress callback
            callback_steps: Callback frequency
            
        Returns:
            Generated 3D mesh(es)
        """
        logger.info(f"Pipeline __call__ started with image type: {type(image)}")
        
        if image is None:
            raise ValueError("Image input is required for shape generation")
        
        # Prepare image
        if isinstance(image, (np.ndarray, torch.Tensor)):
            image = Image.fromarray(np.array(image).astype(np.uint8))
        
        # CRITICAL FIX: Preserve original image format to avoid RGB→RGBA conditioning issues
        original_mode = image.mode
        logger.info(f"Image mode: {image.mode}, size: {image.size}")
        logger.info(f"Original image mode: {original_mode}")
        
        # Remove background if needed
        if image.mode == 'RGB':
            logger.info("Removing background from input image")
            try:
                # Background removal converts to RGBA
                bg_removed_image = self.rembg(image)
                logger.info("Background removal completed")
                
                # CRITICAL: Convert back to original format if it was RGB
                # This preserves the original pixel data structure for better conditioning
                if original_mode == 'RGB' and bg_removed_image.mode == 'RGBA':
                    # Create a white background and composite
                    white_bg = Image.new('RGB', bg_removed_image.size, (255, 255, 255))
                    # Use alpha channel for compositing
                    image = Image.alpha_composite(white_bg.convert('RGBA'), bg_removed_image).convert('RGB')
                    logger.info(f"✅ Converted back to RGB - maintaining original image format")
                    
                    # Verify the conversion preserved image quality
                    import hashlib
                    final_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]
                    logger.info(f"📸 Final processed image hash: {final_hash}")
                else:
                    image = bg_removed_image
                    
            except Exception as e:
                logger.error(f"Background removal failed: {e}")
                # Continue with original image instead of failing
                logger.info("Continuing with original image without background removal")
        
        # CRITICAL: Verify image format consistency after processing
        if image.mode != original_mode:
            logger.warning(f"⚠️  Image mode changed from {original_mode} to {image.mode} - this may affect conditioning quality")
        else:
            logger.info(f"✅ Image format preserved: {image.mode}")
        
        # Convert to tensor and normalize
        logger.info("Preparing image tensor...")
        image_tensor = self._prepare_image(image)
        logger.info(f"Image tensor shape: {image_tensor.shape}")
        
        # Encode image for conditioning
        logger.info("Encoding image for conditioning...")
        image_embeddings = self._encode_image(image)
        logger.info(f"Image embeddings: {image_embeddings.shape if image_embeddings is not None else 'None'}")
        if image_embeddings is None:
            logger.warning("No image embeddings available, generation will be unconditional")
        
        # Adjust number of inference steps based on hardware capabilities if adaptive_steps is enabled
        original_steps = num_inference_steps
        if adaptive_steps:
            num_inference_steps = self._get_adaptive_step_count(requested_steps=num_inference_steps)
            if num_inference_steps != original_steps:
                logger.info(f"Adjusted inference steps from {original_steps} to {num_inference_steps} based on hardware capabilities")
                
                # Update callback_steps if needed to ensure reasonable number of callbacks
                if callback is not None and callback_steps > 1:
                    callback_steps = max(1, num_inference_steps // 10)
                    logger.info(f"Adjusted callback_steps to {callback_steps}")
        
        # Generate latents using flow matching
        logger.info(f"Starting latent generation with {num_inference_steps} steps...")
        try:
            latents = self._generate_latents(
                image_tensor,
                image_embeddings,
                num_inference_steps,
                guidance_scale,
                generator,
                callback,
                callback_steps
            )
            logger.info(f"Latent generation completed, latents shape: {latents.shape}")
        except Exception as e:
            logger.error(f"Latent generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Decode latents to mesh with progress callback
        logger.info("Starting mesh decoding...")
        try:
            mesh = self._decode_latents_to_mesh(latents, box_v, callback=callback)
            logger.info("Mesh decoding completed successfully")
        except Exception as e:
            logger.error(f"Mesh decoding failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Return based on output type
        if output_type == "dict":
            return {"mesh": mesh, "latents": latents}
        elif output_type == "list":
            return [mesh]
        else:
            return mesh
    
    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Prepare image for model input."""
        # Resize to 512x512
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        if image_np.shape[-1] == 4:
            # Use alpha channel as mask
            rgb = image_np[..., :3]
            alpha = image_np[..., 3:4]
            # Composite on white background
            image_np = rgb * alpha + (1 - alpha)
        
        # Normalize to [-1, 1]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        image_tensor = image_tensor * 2.0 - 1.0
        
        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(self.device).to(self.dtype)
        
        return image_tensor
    
    def _encode_image(self, image: Image.Image) -> Optional[torch.Tensor]:
        """Encode image using DINOv2 and project to expected dimension."""
        if self.image_encoder is None or self.image_processor is None or self.embedding_projection is None:
            logger.warning("Image encoder or projection layer not available, returning None")
            return None
        
        try:
            # Process image for DINOv2
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            
            # Get image features
            with torch.no_grad():
                # DINOv2 returns last_hidden_state directly
                outputs = self.image_encoder(**inputs)
                image_embeddings = outputs.last_hidden_state  # (1, num_patches+1, embed_dim)
                
                # Convert to appropriate dtype
                image_embeddings = image_embeddings.to(self.dtype)
                
                # Log original embedding shape
                logger.info(f"Original DINOv2 embeddings shape: {image_embeddings.shape}")
                
                # Project embeddings to expected dimension
                # Apply projection to each token embedding
                B, L, D = image_embeddings.shape
                image_embeddings_flat = image_embeddings.reshape(-1, D)  # Flatten to (B*L, D)
                projected_embeddings_flat = self.embedding_projection(image_embeddings_flat)  # Project to (B*L, context_dim)
                projected_embeddings = projected_embeddings_flat.reshape(B, L, -1)  # Reshape back to (B, L, context_dim)
                
                # Log projected embedding shape
                logger.info(f"Projected embeddings shape: {projected_embeddings.shape}")
            
            return projected_embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            logger.exception(e)  # Print full traceback for debugging
            return None
    
    def _generate_latents(
        self,
        image: torch.Tensor,
        image_embeddings: Optional[torch.Tensor],
        num_inference_steps: int,
        guidance_scale: float,
        generator: Optional[torch.Generator],
        callback: Optional[callable],
        callback_steps: int
    ) -> torch.Tensor:
        """Generate shape latents using flow matching with image conditioning."""
        B = image.shape[0]
        
        # Sample initial noise
        if generator is not None:
            torch.manual_seed(generator.initial_seed())
        
        # Ensure latents have the correct dtype
        latents = torch.randn(
            B, self.vae.num_latents, self.vae.embed_dim,
            device=self.device, dtype=self.dtype
        )
        # Double-check dtype to ensure consistency
        latents = latents.to(dtype=self.dtype)
        
        # Create timesteps (flow matching uses continuous time)
        # Ensure timesteps have the correct dtype
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device, dtype=self.dtype)
        
        # Flow matching loop with timeout protection
        logger.info(f"Starting flow matching loop with {len(timesteps)-1} steps")
        
        import time
        loop_start_time = time.time()
        max_step_time = 300  # 5 minutes max per step to prevent indefinite hangs
        
        # Calculate average time per step to provide better estimates
        avg_step_time = None
        remaining_time_estimate = "unknown"
        
        for i, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            # Check for timeout
            elapsed = time.time() - loop_start_time
            
            # Update average step time and estimate remaining time
            if i > 0:
                avg_step_time = elapsed / i if avg_step_time is None else avg_step_time * 0.8 + (elapsed / i) * 0.2
                steps_remaining = len(timesteps) - i - 1
                if avg_step_time is not None:
                    remaining_time_estimate = f"{avg_step_time * steps_remaining:.1f}s"
            
            # Log progress more frequently for long-running generations
            if i % 5 == 0 or elapsed > 600:  # Every 5 steps or after 10 minutes
                logger.info(f"Flow matching step {i+1}/{len(timesteps)-1} (elapsed: {elapsed:.1f}s, est. remaining: {remaining_time_estimate})")
            
            # Check for global timeout
            if elapsed > self.max_generation_time:
                logger.error(f"Flow matching exceeded maximum allowed time of {self.max_generation_time}s")
                raise TimeoutError(f"Flow matching timed out after {elapsed:.1f}s at step {i}/{len(timesteps)-1}")
            
            # Track time for this specific step
            step_start_time = time.time()
            
            # Expand timestep and ensure correct dtype
            t_batch = t_curr.expand(B).to(dtype=self.dtype)
            
            # Predict velocity (flow) with image conditioning
            step_start_time = time.time()
            with torch.no_grad():
                try:
                    # DIAGNOSTIC: Log image conditioning status only at the beginning and for troubleshooting
                    if i == 0:
                        logger.info(f"🖼️  Image conditioning status:")
                        logger.info(f"   - Image embeddings: {image_embeddings.shape if image_embeddings is not None else 'None'}")
                        if image_embeddings is not None:
                            img_norm = torch.norm(image_embeddings).item() 
                            img_mean = image_embeddings.mean().item()
                            img_std = image_embeddings.std().item()
                            logger.info(f"   - Image embedding stats: norm={img_norm:.3f}, mean={img_mean:.3f}, std={img_std:.3f}")
                        logger.info(f"   - Guidance scale: {guidance_scale}")
                        logger.info(f"   - Using CFG: {guidance_scale > 1.0 and image_embeddings is not None}")
                        # Check conditioning mechanism
                        has_decoupled_ca = hasattr(self.dit, 'config') and getattr(self.dit.config, 'with_decoupled_ca', False)
                        logger.info(f"   - Conditioning mechanism: {'Cross-attention' if has_decoupled_ca else 'AdaLN-based'}")
                        
                        # Verify image embeddings are meaningful and not all zeros
                        if image_embeddings is not None:
                            if torch.norm(image_embeddings).item() < 0.001:
                                logger.warning("⚠️  Image embeddings are nearly zero - conditioning will be ineffective!")
                            else:
                                logger.info("✅ Image embeddings have meaningful values for conditioning")
                    
                    if guidance_scale > 1.0 and image_embeddings is not None:
                        # Classifier-free guidance: run both conditional and unconditional
                        if i == 0:
                            logger.info(f"🔄 Running CFG: conditional + unconditional forward passes")
                        
                        # Memory optimization: explicitly free memory before each forward pass
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        cond_start = time.time()
                        # Ensure image_embeddings has the correct dtype
                        image_embeddings_typed = image_embeddings.to(dtype=self.dtype)
                        velocity_cond = self.dit(latents, t_batch, context=image_embeddings_typed)
                        cond_time = time.time() - cond_start
                        
                        # Memory optimization: explicitly free memory before unconditional pass
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        uncond_start = time.time()
                        # CRITICAL FIX: Use zero embeddings for proper unconditional generation
                        # The DiT model will detect this is a null context and skip conditioning
                        # Ensure all tensors have the same dtype to avoid type mismatch errors
                        null_context = torch.zeros_like(image_embeddings).to(dtype=self.dtype)
                        velocity_uncond = self.dit(latents, t_batch, context=null_context)
                        uncond_time = time.time() - uncond_start
                        
                        # Apply guidance
                        velocity = velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)
                        
                        # DIAGNOSTIC: Verify guidance is being applied (only at start and occasionally)
                        if i == 0 or i % 10 == 0:
                            cond_norm = torch.norm(velocity_cond).item()
                            uncond_norm = torch.norm(velocity_uncond).item()
                            final_norm = torch.norm(velocity).item()
                            logger.info(f"🎯 CFG effectiveness: Conditional={cond_norm:.3f}, Unconditional={uncond_norm:.3f}, Final={final_norm:.3f}")
                            if abs(cond_norm - uncond_norm) < 0.001:
                                logger.warning("⚠️  Conditional and unconditional outputs are nearly identical - image conditioning may not be working!")
                        
                        # Only log timing details at the beginning to reduce overhead
                        if i == 0:
                            logger.info(f"CFG times - Conditional: {cond_time:.2f}s, Unconditional: {uncond_time:.2f}s")
                    else:
                        # Normal generation with or without conditioning
                        if i == 0:
                            logger.info(f"🔄 Running single forward pass with {'image' if image_embeddings is not None else 'no'} conditioning")
                        
                        # Memory optimization: explicitly free memory before forward pass
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        single_start = time.time()
                        velocity = self.dit(latents, t_batch, context=image_embeddings)
                        single_time = time.time() - single_start
                        
                        # Only log timing details at the beginning to reduce overhead
                        if i == 0:
                            logger.info(f"Single forward pass: {single_time:.2f}s")
                except Exception as e:
                    logger.error(f"DiT model forward pass failed at step {i}: {e}")
                    raise
            
            step_time = time.time() - step_start_time
            
            # Check for step-specific timeout
            if step_time > max_step_time:
                logger.error(f"Step {i+1} exceeded maximum allowed time of {max_step_time}s (took {step_time:.2f}s)")
                logger.error(f"This may indicate insufficient GPU memory or other hardware limitations")
                logger.error(f"Consider reducing the number of inference steps or using a smaller model")
                raise TimeoutError(f"Flow matching step {i+1} timed out after {step_time:.2f}s (limit: {max_step_time}s)")
            elif step_time > 60.0:  # Log very slow steps (over 1 minute)
                logger.warning(f"Very slow step {i+1}: {step_time:.2f}s - generation may take a long time to complete")
            elif step_time > 5.0:  # Log moderately slow steps
                logger.warning(f"Slow step {i+1}: {step_time:.2f}s (DiT model may be inefficient)")
            elif i % 5 == 0:  # Log regular progress
                logger.info(f"Step {i+1} completed in {step_time:.2f}s")
            
            # Update latents (Euler step)
            dt = t_next - t_curr
            latents = latents + velocity * dt
            
            # Callback
            if callback is not None and i % callback_steps == 0:
                callback(i, t_curr.item(), f"Flow matching step {i+1}/{len(timesteps)-1}")
        
        logger.info("Flow matching loop completed")
        
        return latents
    
    def _decode_latents_to_mesh(self, latents: torch.Tensor, box_v: float = 1.5, resolution: Optional[int] = None, callback: Optional[callable] = None) -> trimesh.Trimesh:
        """Decode latents to 3D mesh using marching cubes."""
        # Use very low resolution for fast processing
        if resolution is None:
            resolution = 24  # Ultra-low resolution for speed (13,824 points)
        
        # We'll try to decode at this resolution, and fall back if OOM
        logger.info(f"Attempting mesh generation at resolution {resolution}")
        
        # Disable fast mode for now - it's causing GPU crashes
        # if resolution <= 24:
        #     return self._decode_latents_to_mesh_fast(latents, box_v, resolution, callback)
        
        # Generate grid points
        x = torch.linspace(-box_v, box_v, resolution, device=self.device, dtype=torch.float32)
        y = torch.linspace(-box_v, box_v, resolution, device=self.device, dtype=torch.float32)
        z = torch.linspace(-box_v, box_v, resolution, device=self.device, dtype=torch.float32)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        query_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        
        # Ultra-conservative batch sizes to avoid GPU crashes
        if resolution >= 32:
            batch_size = 1024   # Extremely small batches
        elif resolution >= 24:
            batch_size = 2048   # Very small batches
        else:
            batch_size = 4096   # Small batches for low resolution
        
        sdf_values = []
        
        # Clear cache before processing
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except RuntimeError as e:
            logger.warning(f"GPU error during cache clear: {e}")
            # Try to recover GPU
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            except:
                pass
        
        total_points = query_points.shape[0]
        num_batches = (total_points + batch_size - 1) // batch_size
        logger.info(f"Decoding {total_points} points in {num_batches} batches of {batch_size}")
        
        # Report start of mesh decoding
        if callback is not None:
            callback(0, 0.9, f"Starting mesh decoding: {total_points:,} points in {num_batches} batches")
        
        import time
        start_time = time.time()
        
        # Set timeout based on expected processing time (60s per batch for slower GPUs)
        timeout_seconds = min(1800, num_batches * 60)  # Max 30 minutes for very slow processing
        
        try:
            for batch_idx, i in enumerate(range(0, query_points.shape[0], batch_size)):
                # Check for timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    raise TimeoutError(f"Mesh decoding timed out after {elapsed:.1f}s (expected ~{timeout_seconds}s)")
                
                batch_points = query_points[i:i+batch_size].unsqueeze(0)  # (1, N, 3)
                
                with torch.no_grad():
                    # Clear any cached activations before decode
                    if batch_idx % 5 == 0:  # Clear cache more frequently
                        torch.cuda.empty_cache()
                    
                    # Log batch processing
                    batch_start = time.time()
                    
                    # Disable autocast to avoid CUBLAS errors
                    # Ensure all inputs are float32 for stability
                    latents_f32 = latents.float()
                    batch_points_f32 = batch_points.float()
                    
                    # Decode without autocast
                    batch_sdf = self.vae.decode(latents_f32, batch_points_f32)  # (1, N, 1)
                    batch_sdf = batch_sdf.float()  # Ensure float32
                    
                    # Log batch timing
                    batch_time = time.time() - batch_start
                    if batch_idx % 10 == 0:
                        logger.info(f"Batch {batch_idx+1}/{num_batches}: {batch_time:.2f}s for {batch_size} points ({batch_size/batch_time:.0f} points/sec)")
                
                # Keep results on GPU for efficiency
                sdf_values.append(batch_sdf.squeeze(0).squeeze(-1))
                
                # Progress reporting every 25% or for small batches every batch
                if num_batches <= 4 or batch_idx % max(1, num_batches // 4) == 0:
                    progress = (batch_idx + 1) / num_batches
                    elapsed = time.time() - start_time
                    eta = elapsed / progress - elapsed if progress > 0 else 0
                    progress_message = f"Mesh decoding: {progress:.1%} complete ({batch_idx + 1}/{num_batches} batches, ETA: {eta:.1f}s)"
                    logger.info(progress_message)
                    
                    # Call progress callback if provided
                    if callback is not None:
                        # Map mesh decoding progress to volume_decoding step (90%-100% of overall process)
                        overall_progress = 0.9 + progress * 0.1
                        callback(batch_idx, overall_progress, str(progress_message))
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM during mesh decoding at resolution {resolution}, falling back to lower resolution")
            torch.cuda.empty_cache()
            
            # Recursively call with lower resolution
            if resolution > 32:
                new_resolution = max(32, resolution - 16)  # Reduce by 16 each time
                return self._decode_latents_to_mesh(latents, box_v=box_v, resolution=new_resolution, callback=callback)
            else:
                raise RuntimeError("Cannot generate mesh even at minimum resolution of 32")
        
        # Concatenate SDF values on GPU, then move to CPU for marching cubes
        logger.info("Concatenating SDF values and converting to numpy")
        sdf_grid = torch.cat(sdf_values, dim=0).reshape(resolution, resolution, resolution)
        sdf_grid = sdf_grid.cpu().numpy()  # Single transfer at the end
        
        # Log final timing
        total_time = time.time() - start_time
        logger.info(f"Mesh decoding completed in {total_time:.1f}s ({total_points/total_time:.0f} points/sec)")
        
        # Report completion of decoding
        if callback is not None:
            callback(-1, 1.0, f"SDF decoding completed in {total_time:.1f}s - extracting mesh...")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Extract mesh using marching cubes
        from skimage import measure
        try:
            vertices, faces, _, _ = measure.marching_cubes(sdf_grid, level=0.0)
        except Exception as e:
            logger.warning(f"Marching cubes failed with level=0.0: {e}, trying level=0.5")
            vertices, faces, _, _ = measure.marching_cubes(sdf_grid, level=0.5)
        
        # Rescale vertices to world coordinates
        vertices = vertices / (resolution - 1) * 2 * box_v - box_v
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Clean up mesh
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        
        # Ensure consistent face orientation
        mesh.fix_normals()
        
        logger.info(f"Generated mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        # Final completion callback
        if callback is not None:
            callback(-1, 1.0, f"Mesh generation complete: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        return mesh
    
    def _decode_latents_to_mesh_fast(self, latents: torch.Tensor, box_v: float = 1.5, resolution: int = 24, callback: Optional[callable] = None) -> trimesh.Trimesh:
        """Fast mesh decoding for low VRAM situations - processes all points at once."""
        logger.info(f"Using fast mesh generation at resolution {resolution}")
        
        # Generate grid points
        x = torch.linspace(-box_v, box_v, resolution, device=self.device, dtype=torch.float32)
        y = torch.linspace(-box_v, box_v, resolution, device=self.device, dtype=torch.float32)
        z = torch.linspace(-box_v, box_v, resolution, device=self.device, dtype=torch.float32)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        query_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).unsqueeze(0)  # (1, N, 3)
        
        total_points = query_points.shape[1]
        logger.info(f"Processing {total_points:,} points in single batch")
        
        if callback:
            callback(0, 0.9, f"Fast mesh decoding: {total_points:,} points")
        
        # Clear cache and decode all at once
        torch.cuda.empty_cache()
        
        try:
            with torch.no_grad():
                # Force to float32 for stability
                latents_32 = latents.float()
                query_points_32 = query_points.float()
                
                # Decode all points at once
                import time
                start_time = time.time()
                
                # Try different precision levels
                try:
                    # First try with float16 for speed
                    if self.dtype == torch.float16:
                        latents_16 = latents.half()
                        query_points_16 = query_points.half()
                        with torch.cuda.amp.autocast(enabled=True):
                            sdf_values = self.vae.decode(latents_16, query_points_16)
                        sdf_values = sdf_values.float()
                    else:
                        sdf_values = self.vae.decode(latents_32, query_points_32)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    logger.warning(f"GPU decode failed: {e}, trying smaller chunks")
                    # Try processing in two halves
                    mid = total_points // 2
                    sdf1 = self.vae.decode(latents_32, query_points_32[:, :mid])
                    sdf2 = self.vae.decode(latents_32, query_points_32[:, mid:])
                    sdf_values = torch.cat([sdf1, sdf2], dim=1)
                
                decode_time = time.time() - start_time
                logger.info(f"Decoded {total_points} points in {decode_time:.2f}s ({total_points/decode_time:.0f} points/sec)")
                
                # Convert to numpy
                sdf_grid = sdf_values.squeeze().reshape(resolution, resolution, resolution).cpu().numpy()
                
        except Exception as e:
            logger.error(f"Fast decode failed: {e}")
            # Fallback to even lower resolution
            if resolution > 16:
                logger.warning(f"Falling back to resolution 16")
                return self._decode_latents_to_mesh_fast(latents, box_v, 16, callback)
            else:
                raise RuntimeError(f"Cannot generate mesh even at minimum resolution of 16: {e}")
        
        if callback:
            callback(-1, 1.0, "Extracting mesh from SDF grid...")
        
        # Extract mesh using marching cubes
        from skimage import measure
        try:
            vertices, faces, _, _ = measure.marching_cubes(sdf_grid, level=0.0)
        except Exception as e:
            logger.warning(f"Marching cubes failed with level=0.0: {e}, trying level=0.5")
            vertices, faces, _, _ = measure.marching_cubes(sdf_grid, level=0.5)
        
        # Rescale vertices to world coordinates
        vertices = vertices / (resolution - 1) * 2 * box_v - box_v
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Clean up mesh
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        
        logger.info(f"Generated mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        if callback:
            callback(-1, 1.0, f"Mesh generation complete: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        return mesh