"""Advanced prompt optimization for FLUX models.

This module implements prompt engineering techniques specifically optimized
for FLUX models, including style injection, negative prompt generation,
and prompt token optimization.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Structured prompt template for consistent generation."""
    subject: str
    style: Optional[str] = None
    quality_tags: Optional[List[str]] = None
    lighting: Optional[str] = None
    camera: Optional[str] = None
    negative_base: Optional[str] = None
    
    def build(self) -> Tuple[str, str]:
        """Build positive and negative prompts from template."""
        # Build positive prompt
        positive_parts = [self.subject]
        
        if self.style:
            positive_parts.append(self.style)
        
        if self.quality_tags:
            positive_parts.extend(self.quality_tags)
        
        if self.lighting:
            positive_parts.append(self.lighting)
        
        if self.camera:
            positive_parts.append(self.camera)
        
        positive_prompt = ", ".join(positive_parts)
        
        # Build negative prompt
        negative_parts = []
        if self.negative_base:
            negative_parts.append(self.negative_base)
        
        # Add common negative tags based on style
        if self.style and "realistic" in self.style.lower():
            negative_parts.extend([
                "cartoon", "anime", "illustration", "painting",
                "low quality", "blurry", "distorted"
            ])
        elif self.style and "anime" in self.style.lower():
            negative_parts.extend([
                "realistic", "photograph", "3d render",
                "low quality", "bad anatomy"
            ])
        
        negative_prompt = ", ".join(negative_parts) if negative_parts else ""
        
        return positive_prompt, negative_prompt


class FluxPromptOptimizer:
    """Advanced prompt optimization specifically for FLUX models."""
    
    # FLUX-specific quality enhancers
    QUALITY_ENHANCERS = {
        "general": [
            "high quality", "detailed", "professional",
            "award winning", "masterpiece"
        ],
        "photorealistic": [
            "photorealistic", "hyperrealistic", "8k uhd",
            "dslr", "film grain", "Fujifilm XT3", "photograph"
        ],
        "artistic": [
            "artstation", "concept art", "highly detailed",
            "digital painting", "matte painting", "trending on artstation"
        ],
        "anime": [
            "anime style", "manga style", "pixiv",
            "detailed anime", "kawaii", "studio anime"
        ],
        "3d": [
            "octane render", "unreal engine", "3d render",
            "volumetric lighting", "ray tracing", "cinema 4d"
        ]
    }
    
    # Style-specific negative prompts
    NEGATIVE_PROMPTS = {
        "general": [
            "low quality", "worst quality", "normal quality",
            "jpeg artifacts", "compression artifacts", "blurry",
            "film grain", "chromatic aberration"
        ],
        "photorealistic": [
            "cartoon", "anime", "illustration", "painting",
            "drawing", "sketch", "3d render", "cgi",
            "oversaturated", "unnatural colors"
        ],
        "artistic": [
            "photograph", "realistic", "photo",
            "bad art", "amateur", "poorly drawn"
        ],
        "anatomy": [
            "bad anatomy", "wrong anatomy", "deformed",
            "mutation", "mutated", "extra limbs",
            "missing limbs", "floating limbs", "disconnected limbs"
        ],
        "face": [
            "bad face", "ugly face", "deformed face",
            "bad eyes", "crossed eyes", "lazy eye",
            "bad mouth", "crooked teeth"
        ]
    }
    
    # FLUX-specific style mappings
    STYLE_MAPPINGS = {
        "cinematic": {
            "positive": "cinematic lighting, movie still, film photography, shallow depth of field",
            "negative": "flat lighting, amateur photography",
            "guidance_boost": 0.5
        },
        "portrait": {
            "positive": "portrait photography, face focus, shallow dof, professional lighting",
            "negative": "full body, wide shot, busy background",
            "guidance_boost": 0.3
        },
        "landscape": {
            "positive": "landscape photography, wide angle, scenic view, golden hour",
            "negative": "portrait, close-up, macro, people",
            "guidance_boost": 0.2
        },
        "product": {
            "positive": "product photography, clean background, studio lighting, commercial",
            "negative": "busy background, harsh shadows, amateur",
            "guidance_boost": 0.4
        },
        "fantasy": {
            "positive": "fantasy art, magical, ethereal, dreamlike, enchanted",
            "negative": "realistic, mundane, ordinary, photograph",
            "guidance_boost": 0.7
        }
    }
    
    def __init__(self, model_type: str = "flux-dev"):
        self.model_type = model_type
        self.tokenizer = None
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """Setup tokenizer for token counting."""
        try:
            # FLUX uses T5 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google/t5-v1_1-xxl",
                model_max_length=512
            )
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None
    
    def optimize_prompt(self,
                       prompt: str,
                       style: Optional[str] = None,
                       quality_preset: str = "high",
                       add_negative: bool = True) -> Dict[str, Any]:
        """Optimize prompt for FLUX generation.
        
        Returns:
            Dict containing:
            - prompt: Optimized positive prompt
            - negative_prompt: Generated negative prompt
            - guidance_adjustment: Suggested guidance scale adjustment
            - token_count: Number of tokens in prompt
        """
        
        # Clean and prepare base prompt
        prompt = self._clean_prompt(prompt)
        
        # Add style-specific enhancements
        if style and style in self.STYLE_MAPPINGS:
            style_config = self.STYLE_MAPPINGS[style]
            prompt = f"{prompt}, {style_config['positive']}"
            guidance_adjustment = style_config['guidance_boost']
        else:
            guidance_adjustment = 0.0
        
        # Detect prompt type and add quality tags
        prompt_type = self._detect_prompt_type(prompt)
        quality_tags = self._get_quality_tags(prompt_type, quality_preset)
        
        if quality_tags:
            prompt = f"{prompt}, {', '.join(quality_tags)}"
        
        # Generate negative prompt
        negative_prompt = ""
        if add_negative:
            negative_prompt = self._generate_negative_prompt(prompt, style, prompt_type)
        
        # Count tokens
        token_count = self._count_tokens(prompt) if self.tokenizer else -1
        
        # Optimize token usage if needed
        if token_count > 380:  # Leave room for other tokens
            prompt = self._compress_prompt(prompt, max_tokens=380)
            token_count = self._count_tokens(prompt) if self.tokenizer else -1
        
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_adjustment": guidance_adjustment,
            "token_count": token_count,
            "prompt_type": prompt_type,
            "quality_preset": quality_preset
        }
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and normalize prompt."""
        # Remove extra whitespace
        prompt = " ".join(prompt.split())
        
        # Fix common issues
        prompt = prompt.replace(" ,", ",")
        prompt = prompt.replace(",,", ",")
        
        # Remove duplicate words
        words = prompt.split(", ")
        seen = set()
        unique_words = []
        for word in words:
            word_lower = word.lower().strip()
            if word_lower not in seen:
                seen.add(word_lower)
                unique_words.append(word)
        
        return ", ".join(unique_words)
    
    def _detect_prompt_type(self, prompt: str) -> str:
        """Detect the type of prompt for appropriate enhancement."""
        prompt_lower = prompt.lower()
        
        # Check for specific types
        if any(word in prompt_lower for word in ["photo", "photograph", "realistic", "real"]):
            return "photorealistic"
        elif any(word in prompt_lower for word in ["anime", "manga", "chibi", "kawaii"]):
            return "anime"
        elif any(word in prompt_lower for word in ["3d", "render", "cgi", "octane"]):
            return "3d"
        elif any(word in prompt_lower for word in ["art", "painting", "illustration"]):
            return "artistic"
        else:
            return "general"
    
    def _get_quality_tags(self, prompt_type: str, quality_preset: str) -> List[str]:
        """Get appropriate quality tags based on prompt type and preset."""
        tags = []
        
        # Base quality tags
        if quality_preset in ["high", "ultra"]:
            tags.extend(self.QUALITY_ENHANCERS.get("general", [])[:3])
        
        # Type-specific tags
        type_tags = self.QUALITY_ENHANCERS.get(prompt_type, [])
        if quality_preset == "ultra":
            tags.extend(type_tags[:4])
        elif quality_preset == "high":
            tags.extend(type_tags[:2])
        elif quality_preset == "medium":
            tags.extend(type_tags[:1])
        
        return tags
    
    def _generate_negative_prompt(self, positive_prompt: str, style: Optional[str], prompt_type: str) -> str:
        """Generate appropriate negative prompt."""
        negative_parts = []
        
        # Add general negatives
        negative_parts.extend(self.NEGATIVE_PROMPTS["general"][:4])
        
        # Add type-specific negatives
        if prompt_type in self.NEGATIVE_PROMPTS:
            negative_parts.extend(self.NEGATIVE_PROMPTS[prompt_type][:3])
        
        # Add anatomy negatives if people/characters detected
        if any(word in positive_prompt.lower() for word in ["person", "people", "man", "woman", "character"]):
            negative_parts.extend(self.NEGATIVE_PROMPTS["anatomy"][:3])
            negative_parts.extend(self.NEGATIVE_PROMPTS["face"][:2])
        
        # Add style-specific negatives
        if style and style in self.STYLE_MAPPINGS:
            style_negative = self.STYLE_MAPPINGS[style].get("negative", "")
            if style_negative:
                negative_parts.append(style_negative)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_negatives = []
        for item in negative_parts:
            if item.lower() not in seen:
                seen.add(item.lower())
                unique_negatives.append(item)
        
        return ", ".join(unique_negatives)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not self.tokenizer:
            return -1
        
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            return -1
    
    def _compress_prompt(self, prompt: str, max_tokens: int = 380) -> str:
        """Compress prompt to fit within token limit."""
        if not self.tokenizer:
            # Simple truncation if no tokenizer
            parts = prompt.split(", ")
            while len(", ".join(parts)) > max_tokens * 4:  # Rough estimate
                parts.pop()
            return ", ".join(parts)
        
        # Smart compression with tokenizer
        parts = prompt.split(", ")
        essential_parts = parts[:3]  # Keep first 3 parts as essential
        optional_parts = parts[3:]
        
        # Start with essential parts
        compressed = ", ".join(essential_parts)
        current_tokens = self._count_tokens(compressed)
        
        # Add optional parts while under limit
        for part in optional_parts:
            test_prompt = f"{compressed}, {part}"
            test_tokens = self._count_tokens(test_prompt)
            
            if test_tokens <= max_tokens:
                compressed = test_prompt
                current_tokens = test_tokens
            else:
                break
        
        return compressed
    
    def create_variations(self, base_prompt: str, num_variations: int = 4) -> List[str]:
        """Create prompt variations for diverse outputs."""
        variations = []
        
        # Variation strategies
        style_variations = ["photorealistic", "artistic", "cinematic", "professional"]
        lighting_variations = ["natural lighting", "studio lighting", "golden hour", "dramatic lighting"]
        angle_variations = ["front view", "three quarter view", "profile view", "dynamic angle"]
        
        for i in range(num_variations):
            variation_parts = [base_prompt]
            
            # Add style variation
            if i < len(style_variations):
                variation_parts.append(style_variations[i])
            
            # Add lighting variation
            lighting_idx = i % len(lighting_variations)
            variation_parts.append(lighting_variations[lighting_idx])
            
            # Add angle variation for subjects
            if any(word in base_prompt.lower() for word in ["portrait", "character", "person"]):
                angle_idx = i % len(angle_variations)
                variation_parts.append(angle_variations[angle_idx])
            
            variation = ", ".join(variation_parts)
            variations.append(self._clean_prompt(variation))
        
        return variations
    
    def enhance_for_3d(self, prompt: str) -> str:
        """Enhance prompt specifically for 3D asset generation."""
        # Add 3D-specific enhancements
        enhancements = [
            "centered composition",
            "clean background",
            "full object visible",
            "consistent lighting",
            "neutral pose"
        ]
        
        # Add material/texture hints
        if "metal" in prompt.lower():
            enhancements.append("metallic surface")
        elif "wood" in prompt.lower():
            enhancements.append("wood texture")
        elif "fabric" in prompt.lower():
            enhancements.append("fabric texture")
        
        enhanced = f"{prompt}, {', '.join(enhancements)}"
        
        # Add negative prompt for 3D
        negative = "multiple objects, cluttered, partial view, extreme perspective, motion blur"
        
        return enhanced


# Utility functions
def demonstrate_prompt_optimization():
    """Demonstrate prompt optimization capabilities."""
    optimizer = FluxPromptOptimizer()
    
    test_prompts = [
        "a cat",
        "portrait of a warrior",
        "futuristic city at night",
        "anime girl with blue hair",
        "product photo of a watch"
    ]
    
    for prompt in test_prompts:
        print(f"\nOriginal: {prompt}")
        print("-" * 50)
        
        # Try different styles
        for style in ["photorealistic", "cinematic", "artistic"]:
            result = optimizer.optimize_prompt(
                prompt,
                style=style,
                quality_preset="high"
            )
            
            print(f"\n{style.upper()}:")
            print(f"Positive: {result['prompt'][:100]}...")
            print(f"Negative: {result['negative_prompt'][:80]}...")
            print(f"Tokens: {result['token_count']}")
            print(f"Guidance adjustment: +{result['guidance_adjustment']}")


def create_prompt_library() -> Dict[str, PromptTemplate]:
    """Create a library of proven prompt templates."""
    return {
        "portrait_professional": PromptTemplate(
            subject="professional headshot of a person",
            style="photorealistic portrait photography",
            quality_tags=["8k", "dslr", "85mm lens"],
            lighting="soft studio lighting",
            camera="shallow depth of field, bokeh",
            negative_base="cartoon, anime, 3d render"
        ),
        "product_hero": PromptTemplate(
            subject="product photography",
            style="commercial product shot",
            quality_tags=["professional", "clean", "minimal"],
            lighting="gradient background, studio lighting",
            camera="macro lens, sharp focus",
            negative_base="busy background, reflections"
        ),
        "character_fantasy": PromptTemplate(
            subject="fantasy character",
            style="concept art, digital painting",
            quality_tags=["detailed", "artstation", "trending"],
            lighting="dramatic lighting, rim light",
            camera="dynamic pose, full body",
            negative_base="photograph, realistic"
        ),
        "environment_scifi": PromptTemplate(
            subject="futuristic environment",
            style="sci-fi concept art",
            quality_tags=["detailed", "atmospheric", "moody"],
            lighting="volumetric lighting, neon accents",
            camera="wide angle, establishing shot",
            negative_base="medieval, fantasy, old"
        )
    }