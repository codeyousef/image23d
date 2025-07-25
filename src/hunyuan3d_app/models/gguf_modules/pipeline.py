"""GGUF pipeline generation functionality"""

import logging
from typing import Optional, Dict, Any, Union
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class GGUFPipelineGenerator:
    """Handles generation with GGUF pipelines"""
    
    def __init__(self, pipeline: Any, device: str = "cuda"):
        self.pipeline = pipeline
        self.device = device
        
    def generate(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate images using the GGUF pipeline"""
        logger.info(f"Generating with GGUF pipeline: prompt='{prompt[:50]}...', steps={num_inference_steps}")
        
        # Prepare generation arguments
        gen_kwargs = {
            "prompt": prompt,
            "prompt_2": prompt_2 or prompt,
            "height": height or 1024,
            "width": width or 1024,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator,
            "output_type": output_type,
            "return_dict": return_dict
        }
        
        # Add any additional kwargs
        gen_kwargs.update(kwargs)
        
        # Handle Q8 models differently
        is_q8 = hasattr(self.pipeline, '_is_placeholder') and "q8" in getattr(self.pipeline, 'model_name', '').lower()
        
        if is_q8:
            logger.info("Q8 model detected - ensuring proper device placement")
            # Ensure transformer is on GPU for inference
            if hasattr(self.pipeline, 'transformer') and self.device == "cuda":
                device_check = next(self.pipeline.transformer.parameters()).device
                if device_check.type != 'cuda':
                    logger.info(f"Moving transformer from {device_check} to {self.device} for inference")
                    self.pipeline.transformer = self.pipeline.transformer.to(self.device)
        
        try:
            # For Q6/Q8 models, use lower memory batch size
            is_large_quant = any(q in getattr(self.pipeline, 'model_name', '').lower() 
                               for q in ["q6", "q8", "Q6", "Q8"])
            
            if is_large_quant and gen_kwargs.get('num_images_per_prompt', 1) > 1:
                logger.info(f"Q6/Q8 model: Reducing batch size from {gen_kwargs['num_images_per_prompt']} to 1")
                gen_kwargs['num_images_per_prompt'] = 1
            
            # Generate images
            result = self.pipeline(**gen_kwargs)
            
            logger.info("Generation completed successfully")
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory during generation: {e}")
            
            # Try to recover
            from ...utils.memory_optimization import get_memory_manager
            memory_manager = get_memory_manager()
            memory_manager.aggressive_memory_clear()
            
            # Retry with reduced settings
            logger.info("Retrying with reduced settings...")
            gen_kwargs['num_inference_steps'] = max(10, num_inference_steps // 2)
            gen_kwargs['height'] = min(768, gen_kwargs['height'])
            gen_kwargs['width'] = min(768, gen_kwargs['width'])
            
            return self.pipeline(**gen_kwargs)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    @staticmethod
    def create_placeholder_result(
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_images_per_prompt: int = 1
    ) -> Dict[str, Any]:
        """Create a placeholder result for testing"""
        # Create gradient placeholder images
        images = []
        for i in range(num_images_per_prompt):
            # Create a gradient image
            img = Image.new('RGB', (width, height))
            pixels = img.load()
            
            for y in range(height):
                for x in range(width):
                    # Create a diagonal gradient
                    r = int((x / width) * 255)
                    g = int((y / height) * 255)
                    b = int(((x + y) / (width + height)) * 255)
                    pixels[x, y] = (r, g, b)
            
            images.append(img)
        
        return {"images": images}