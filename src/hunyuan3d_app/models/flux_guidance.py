"""Advanced guidance and sampling techniques for FLUX models.

This module implements distilled CFG, dynamic guidance scheduling,
multi-stage generation, and optimal sampler/scheduler combinations.
"""

import torch
import time
import logging
from typing import Optional, Dict, List, Any, Union
from PIL import Image

logger = logging.getLogger(__name__)


class AdvancedFluxGuidance:
    """Advanced guidance techniques for Flux.1 models.
    
    CRITICAL: Flux.1-dev uses Distilled CFG, not regular CFG!
    Always set CFG=1.0 and use distilled_cfg_scale instead.
    """
    
    def __init__(self, model_variant: str = "dev"):
        self.model_variant = model_variant
        
        # Guidance strategies for different styles
        self.guidance_strategies = {
            "photorealistic": {"distilled_cfg": 2.0, "regular_cfg": 1.0},
            "artistic": {"distilled_cfg": 4.5, "regular_cfg": 1.0},
            "complex_scene": {"distilled_cfg": 6.0, "regular_cfg": 1.0},
            "simple_portrait": {"distilled_cfg": 2.5, "regular_cfg": 1.0},
            "cinematic": {"distilled_cfg": 3.0, "regular_cfg": 1.0},
            "anime": {"distilled_cfg": 4.0, "regular_cfg": 1.0},
            "3d_render": {"distilled_cfg": 3.5, "regular_cfg": 1.0}
        }
    
    def get_guidance_config(self, style: str = "photorealistic") -> Dict[str, float]:
        """Get optimal guidance configuration for a style."""
        return self.guidance_strategies.get(style, self.guidance_strategies["photorealistic"])
    
    def dynamic_guidance_scheduling(self, 
                                   pipe: Any,
                                   prompt: str,
                                   style: str = "photorealistic",
                                   height: int = 1024,
                                   width: int = 1024,
                                   num_inference_steps: int = 28,
                                   seed: Optional[int] = None) -> Image.Image:
        """Implement dynamic guidance scheduling for optimal results."""
        
        guidance_config = self.get_guidance_config(style)
        
        # For Flux.1-dev (distilled model)
        if self.model_variant == "dev":
            logger.info(f"Using distilled CFG guidance for style: {style}")
            logger.info(f"Distilled CFG Scale: {guidance_config['distilled_cfg']}")
            
            # Use distilled CFG guidance
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_config['distilled_cfg'],  # This is distilled CFG
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cuda").manual_seed(seed) if seed else None
            ).images[0]
        
        return image
    
    def guidance_rescaling(self, 
                          base_guidance: float, 
                          complexity_factor: float = 1.0, 
                          realism_factor: float = 1.0) -> float:
        """Rescale guidance based on prompt complexity and desired realism.
        
        Args:
            base_guidance: Base distilled CFG scale
            complexity_factor: 0-2, where 2 is very complex scene
            realism_factor: 0-2, where 2 is maximum realism
        
        Returns:
            Adjusted distilled CFG scale
        """
        # Complex scenes need higher guidance for prompt adherence
        complexity_adjustment = base_guidance * (1 + (complexity_factor - 1) * 0.5)
        
        # Photorealistic images often benefit from lower guidance
        realism_adjustment = complexity_adjustment * (2.0 - realism_factor * 0.5)
        
        # Clamp to reasonable range for distilled CFG
        return max(1.0, min(8.0, realism_adjustment))
    
    def multi_stage_generation(self, 
                              pipe: Any,
                              prompt: str,
                              negative_prompt: Optional[str] = None,
                              stages: Optional[List[Dict]] = None,
                              height: int = 1024,
                              width: int = 1024,
                              seed: Optional[int] = None) -> Image.Image:
        """Multi-stage generation with different guidance at each stage.
        
        This implements the three-phase optimization strategy:
        Phase 1: Structure (high guidance)
        Phase 2: Details (medium guidance)
        Phase 3: Polish (low guidance)
        """
        
        if stages is None:
            stages = [
                {"steps": 8, "guidance": 4.0, "denoise": 1.0},    # Structure
                {"steps": 12, "guidance": 2.5, "denoise": 0.75},  # Details
                {"steps": 8, "guidance": 1.5, "denoise": 0.5}     # Polish
            ]
        
        generator = torch.Generator("cuda").manual_seed(seed) if seed else None
        
        logger.info("Starting multi-stage generation")
        
        # Phase 1: Initial generation with high guidance for structure
        logger.info(f"Phase 1: Structure generation (guidance={stages[0]['guidance']})")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=stages[0]["guidance"],
            num_inference_steps=stages[0]["steps"],
            generator=generator
        ).images[0]
        
        # Phase 2 & 3: Refinement stages with progressively lower guidance
        for i, stage in enumerate(stages[1:], 2):
            logger.info(f"Phase {i}: {'Details' if i == 2 else 'Polish'} "
                       f"(guidance={stage['guidance']}, strength={stage['denoise']})")
            
            # Convert to img2img for refinement
            image = pipe(
                prompt=prompt,
                image=image,
                strength=stage["denoise"],
                guidance_scale=stage["guidance"],
                num_inference_steps=stage["steps"],
                generator=generator
            ).images[0]
        
        return image
    
    def calculate_complexity_factor(self, prompt: str) -> float:
        """Estimate prompt complexity for guidance adjustment."""
        # Simple heuristic based on prompt characteristics
        complexity_indicators = [
            "multiple", "several", "many", "various", "complex",
            "detailed", "intricate", "elaborate", "crowded",
            "busy", "filled with", "surrounded by"
        ]
        
        # Count objects/subjects
        object_indicators = ["and", "with", "including", "featuring", ","]
        
        complexity_score = 0.0
        prompt_lower = prompt.lower()
        
        # Check for complexity indicators
        for indicator in complexity_indicators:
            if indicator in prompt_lower:
                complexity_score += 0.2
        
        # Count potential objects
        for indicator in object_indicators:
            complexity_score += prompt_lower.count(indicator) * 0.1
        
        # Normalize to 0-2 range
        return min(2.0, complexity_score)


class FluxSamplerOptimizer:
    """Optimize samplers and schedulers for different use cases.
    
    Note: Current diffusers implementation has fixed samplers,
    but these settings represent optimal configurations.
    """
    
    OPTIMAL_COMBINATIONS = {
        "photorealistic": {
            "sampler": "heun",
            "scheduler": "beta",
            "steps": 25,
            "distilled_cfg": 2.5,
            "description": "Best for realistic photos, portraits"
        },
        "artistic": {
            "sampler": "deis", 
            "scheduler": "ddim_uniform",
            "steps": 30,
            "distilled_cfg": 4.0,
            "description": "Best for artistic, stylized images"
        },
        "speed_optimized": {
            "sampler": "euler",
            "scheduler": "simple", 
            "steps": 20,
            "distilled_cfg": 3.5,
            "description": "Fastest generation with good quality"
        },
        "high_quality": {
            "sampler": "dpmpp_2m",
            "scheduler": "beta",
            "steps": 40,
            "distilled_cfg": 3.0,
            "description": "Maximum quality, slower generation"
        },
        "anime": {
            "sampler": "deis",
            "scheduler": "ddim_uniform",
            "steps": 28,
            "distilled_cfg": 4.0,
            "description": "Optimized for anime/manga style"
        },
        "3d_render": {
            "sampler": "heun",
            "scheduler": "beta",
            "steps": 30,
            "distilled_cfg": 3.5,
            "description": "Best for 3D renders and CGI"
        }
    }
    
    @classmethod
    def get_optimal_settings(cls, use_case: str = "photorealistic") -> Dict[str, Any]:
        """Get optimal sampler/scheduler combination for specific use case."""
        return cls.OPTIMAL_COMBINATIONS.get(use_case, cls.OPTIMAL_COMBINATIONS["photorealistic"])
    
    @classmethod
    def auto_select_settings(cls, prompt: str) -> Dict[str, Any]:
        """Automatically select optimal settings based on prompt analysis."""
        prompt_lower = prompt.lower()
        
        # Check for style indicators
        if any(word in prompt_lower for word in ["photo", "realistic", "real", "photograph"]):
            return cls.get_optimal_settings("photorealistic")
        elif any(word in prompt_lower for word in ["anime", "manga", "cartoon"]):
            return cls.get_optimal_settings("anime")
        elif any(word in prompt_lower for word in ["3d", "render", "cgi", "unreal"]):
            return cls.get_optimal_settings("3d_render")
        elif any(word in prompt_lower for word in ["art", "painting", "artistic", "style"]):
            return cls.get_optimal_settings("artistic")
        else:
            return cls.get_optimal_settings("speed_optimized")
    
    @classmethod
    def benchmark_samplers(cls, pipe: Any, prompt: str, test_cases: Optional[List[str]] = None) -> Dict:
        """Benchmark different sampler combinations."""
        if test_cases is None:
            test_cases = ["photorealistic", "artistic", "speed_optimized"]
        
        results = {}
        
        for use_case in test_cases:
            settings = cls.get_optimal_settings(use_case)
            logger.info(f"\nTesting {use_case} settings...")
            
            start_time = time.time()
            
            # Note: In practice, you would need to configure the scheduler
            # This is a simplified version
            image = pipe(
                prompt=prompt,
                num_inference_steps=settings["steps"],
                guidance_scale=settings["distilled_cfg"],
                height=512,
                width=512
            ).images[0]
            
            generation_time = time.time() - start_time
            
            results[use_case] = {
                "time": generation_time,
                "image": image,
                "settings": settings
            }
            
            logger.info(f"{use_case}: {generation_time:.2f}s")
        
        return results


class DistilledCFGScheduler:
    """Manage distilled CFG scheduling throughout generation process."""
    
    def __init__(self):
        self.schedules = {
            "linear_decrease": lambda step, total: 4.0 - (step / total) * 2.0,
            "cosine_decrease": lambda step, total: 2.0 + 2.0 * torch.cos(torch.tensor(step / total * torch.pi)).item(),
            "stepped": self._stepped_schedule,
            "adaptive": self._adaptive_schedule
        }
    
    def _stepped_schedule(self, step: int, total_steps: int) -> float:
        """Stepped schedule matching the three-phase approach."""
        progress = step / total_steps
        
        if progress < 0.3:  # Structure phase
            return 4.0
        elif progress < 0.7:  # Detail phase
            return 2.5
        else:  # Polish phase
            return 1.5
    
    def _adaptive_schedule(self, step: int, total_steps: int, complexity: float = 1.0) -> float:
        """Adaptive schedule based on complexity."""
        base_schedule = self._stepped_schedule(step, total_steps)
        return base_schedule * (0.5 + 0.5 * complexity)
    
    def get_schedule(self, schedule_type: str = "stepped") -> callable:
        """Get a CFG schedule function."""
        return self.schedules.get(schedule_type, self.schedules["stepped"])


# Example usage functions
def demonstrate_guidance_strategies(pipe: Any):
    """Demonstrate different guidance strategies."""
    guidance = AdvancedFluxGuidance()
    
    test_prompts = {
        "simple": "A serene mountain landscape",
        "complex": "A bustling cyberpunk marketplace with multiple vendors, "
                  "neon signs, flying cars, and crowds of diverse people",
        "portrait": "Professional headshot of a business person",
        "artistic": "Abstract painting in the style of Kandinsky"
    }
    
    results = {}
    
    for prompt_type, prompt in test_prompts.items():
        # Calculate complexity
        complexity = guidance.calculate_complexity_factor(prompt)
        
        # Get auto-selected settings
        sampler_settings = FluxSamplerOptimizer.auto_select_settings(prompt)
        
        # Adjust guidance based on complexity
        base_guidance = sampler_settings["distilled_cfg"]
        adjusted_guidance = guidance.guidance_rescaling(
            base_guidance, 
            complexity_factor=complexity,
            realism_factor=1.5 if "photo" in prompt_type else 0.5
        )
        
        logger.info(f"\n{prompt_type.upper()} prompt:")
        logger.info(f"Complexity: {complexity:.2f}")
        logger.info(f"Base guidance: {base_guidance}")
        logger.info(f"Adjusted guidance: {adjusted_guidance:.2f}")
        logger.info(f"Optimal steps: {sampler_settings['steps']}")
        
        results[prompt_type] = {
            "complexity": complexity,
            "guidance": adjusted_guidance,
            "settings": sampler_settings
        }
    
    return results