"""
Prompt enhancement engine using Ollama for LLM-based enhancement
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

try:
    import ollama
    # Also check if Ollama server is actually accessible
    try:
        # Test connection with timeout to avoid hanging
        import socket
        
        # First check if the service is listening on default port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)  # 2 second timeout
        result = sock.connect_ex(('localhost', 11434))
        sock.close()
        
        if result == 0:
            # Port is open, try to list models with limited retries
            try:
                # Use threading to timeout the ollama call
                import threading
                import queue
                
                def check_ollama(q):
                    try:
                        ollama.list()
                        q.put(True)
                    except Exception as e:
                        q.put(False)
                
                q = queue.Queue()
                thread = threading.Thread(target=check_ollama, args=(q,))
                thread.daemon = True
                thread.start()
                thread.join(timeout=3)  # 3 second timeout
                
                if thread.is_alive():
                    # Thread is still running, ollama is hanging
                    OLLAMA_AVAILABLE = False
                    ollama = None
                else:
                    # Thread completed, check result
                    try:
                        result = q.get_nowait()
                        OLLAMA_AVAILABLE = result
                        if not result:
                            ollama = None
                    except queue.Empty:
                        OLLAMA_AVAILABLE = False
                        ollama = None
                        
            except Exception:
                OLLAMA_AVAILABLE = False
                ollama = None
        else:
            OLLAMA_AVAILABLE = False
            ollama = None
            
    except Exception:
        OLLAMA_AVAILABLE = False
        ollama = None
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None
    
from ..models.enhancement import ModelType, EnhancementFields
from ..config import (
    FLUX_ENHANCEMENT_FIELDS, 
    HUNYUAN_ENHANCEMENT_FIELDS, 
    SPARC3D_ENHANCEMENT_FIELDS,
    HI3DGEN_ENHANCEMENT_FIELDS,
    CACHE_DIR
)

logger = logging.getLogger(__name__)

# LLM Templates
FLUX_TEMPLATE = """You are an expert at writing prompts for Flux.1 image generation AI. Your task is to transform a basic prompt into a detailed, professional prompt that will generate stunning images.

Transform this basic prompt into a detailed Flux.1 prompt that includes ALL of the following elements:

1. **Subject Description**: Expand the subject with specific details about appearance, pose, expression, clothing, etc.
2. **Art Style/Medium**: Specify the artistic medium (e.g., "digital painting", "oil on canvas", "photorealistic render", "watercolor illustration")
3. **Lighting Setup**: Describe the lighting (e.g., "golden hour sunlight", "dramatic chiaroscuro", "soft studio lighting", "neon cyberpunk glow")
4. **Camera/Composition**: Include camera angle and framing (e.g., "low angle shot", "rule of thirds composition", "extreme close-up", "aerial view")
5. **Atmosphere/Mood**: Set the emotional tone (e.g., "mysterious and ethereal", "vibrant and energetic", "melancholic", "triumphant")
6. **Color Palette**: Describe the color scheme (e.g., "muted earth tones", "vibrant neon palette", "monochromatic blue", "warm sunset hues")
7. **Technical Quality**: Add quality modifiers (e.g., "8k resolution", "highly detailed", "sharp focus", "ray tracing", "octane render")
8. **Additional Details**: Include environmental elements, background, or contextual details

Basic prompt: "{user_prompt}"

Write ONLY the enhanced prompt in a single paragraph. Do not include explanations, lists, or formatting. The output should be ready to paste directly into Flux.1.

Enhanced prompt:"""

HUNYUAN_TEMPLATE = """You are an expert at writing prompts for HunYuan 3D 2.1 model generation. Your task is to transform a basic concept into a detailed prompt that will generate high-quality, professional 3D assets.

Transform this basic prompt into a detailed HunYuan 3D prompt that includes ALL of the following specifications:

1. **Geometric Description**: Detailed shape, proportions, and structural elements
2. **Materials & Textures**: Specify PBR materials (e.g., "brushed metal with rust", "polished marble", "worn leather")
3. **Surface Details**: Describe fine details, patterns, engravings, or surface features
4. **Style Classification**: Define the style (e.g., "photorealistic", "stylized cartoon", "low-poly", "sculptural")
5. **Technical Specifications**: Include polygon density preference, UV mapping needs, rigging requirements
6. **Use Case**: Specify intended use (e.g., "game asset", "architectural visualization", "3D printing", "film production")
7. **Scale & Proportions**: Provide size context and proportional relationships
8. **Additional Features**: Any moving parts, modular elements, or special requirements

Basic prompt: "{user_prompt}"

Write ONLY the enhanced prompt in a single paragraph. Focus on clarity and technical precision for 3D generation.

Enhanced prompt:"""

SPARC3D_TEMPLATE = """You are an expert at writing prompts for Sparc3D high-resolution 3D reconstruction. Your task is to transform a basic concept into a detailed prompt that leverages Sparc3D's unique sparse representation capabilities.

Transform this basic prompt into a detailed Sparc3D prompt that includes ALL of the following specifications:

1. **Structural Complexity**: Describe the overall structural complexity and topology (e.g., "complex interconnected geometry", "simple watertight form", "multi-component assembly")
2. **Resolution Requirements**: Specify the level of detail needed (e.g., "ultra-high resolution capturing millimeter details", "balanced resolution for real-time use")
3. **Surface Characteristics**: Detail surface properties (e.g., "sharp edges with precise corners", "organic flowing surfaces", "mixed sharp and smooth regions")
4. **Sparse Representation**: Guide the sparse cube distribution (e.g., "dense sampling at detail areas", "adaptive sparse representation", "uniform distribution")
5. **Reconstruction Goals**: Define the reconstruction objectives (e.g., "preserve all fine details", "optimize for 3D printing", "game-ready optimization")
6. **Topology Handling**: Specify topology requirements (e.g., "handle multiple disconnected parts", "ensure manifold geometry", "preserve holes and cavities")
7. **Output Requirements**: Define output needs (e.g., "watertight mesh for manufacturing", "textured model for rendering", "clean topology for animation")

Basic prompt: "{user_prompt}"

Write ONLY the enhanced prompt in a single paragraph focused on leveraging Sparc3D's sparse representation strengths.

Enhanced prompt:"""

HI3DGEN_TEMPLATE = """You are an expert at writing prompts for Hi3DGen normal-bridging 3D generation. Your task is to transform a basic concept into a detailed prompt that maximizes Hi3DGen's normal map estimation capabilities.

Transform this basic prompt into a detailed Hi3DGen prompt that includes ALL of the following specifications:

1. **Surface Normal Details**: Describe surface orientation and normal variations (e.g., "complex normal variations with subtle surface undulations", "sharp normal transitions at edges")
2. **Material Properties**: Specify material characteristics that affect normals (e.g., "brushed metal with directional grooves", "rough stone with micro-detail", "smooth reflective surface")
3. **Geometric Fidelity**: Define the level of geometric accuracy needed (e.g., "pixel-perfect fidelity to reference", "enhanced geometric interpretation", "stylized but accurate")
4. **Normal Map Quality**: Specify normal map requirements (e.g., "16-bit precision normals", "tangent-space normal mapping", "world-space normals")
5. **Detail Preservation**: Describe which details are critical (e.g., "preserve all surface imperfections", "maintain fabric weave patterns", "capture engraved text")
6. **Lighting Interaction**: Consider how normals will interact with lighting (e.g., "optimized for dramatic lighting", "subtle normal variations for realism")
7. **Integration Requirements**: Specify pipeline integration needs (e.g., "game engine ready", "film production quality", "real-time rendering optimized")

Basic prompt: "{user_prompt}"

Write ONLY the enhanced prompt in a single paragraph that emphasizes Hi3DGen's strength in normal-based reconstruction.

Enhanced prompt:"""

TEMPLATES = {
    ModelType.FLUX_1_DEV: FLUX_TEMPLATE,
    ModelType.FLUX_1_SCHNELL: FLUX_TEMPLATE,
    ModelType.HUNYUAN_3D_21: HUNYUAN_TEMPLATE,
    ModelType.HUNYUAN_3D_20: HUNYUAN_TEMPLATE,
    ModelType.HUNYUAN_3D_MINI: HUNYUAN_TEMPLATE,
    ModelType.SPARC3D: SPARC3D_TEMPLATE,
    ModelType.HI3DGEN: HI3DGEN_TEMPLATE,
}

class PromptEnhancer:
    """Enhances prompts using LLM and field-based UI selections"""
    
    def __init__(self, llm_model: str = "mistral:latest", cache_dir: Optional[Path] = None):
        self.llm_model = llm_model
        self.cache_dir = cache_dir or CACHE_DIR / "prompt_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if Ollama is available
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama not available (not installed or server not running). Prompt enhancement will use field-based enhancement only.")
            logger.info("To enable LLM-based prompt enhancement, install Ollama and start the server: 'ollama serve'")
            
        # Load field configurations
        self.field_configs = {
            ModelType.FLUX_1_DEV: FLUX_ENHANCEMENT_FIELDS,
            ModelType.FLUX_1_SCHNELL: FLUX_ENHANCEMENT_FIELDS,
            ModelType.HUNYUAN_3D_21: HUNYUAN_ENHANCEMENT_FIELDS,
            ModelType.HUNYUAN_3D_20: HUNYUAN_ENHANCEMENT_FIELDS,
            ModelType.HUNYUAN_3D_MINI: HUNYUAN_ENHANCEMENT_FIELDS,
            ModelType.SPARC3D: SPARC3D_ENHANCEMENT_FIELDS,
            ModelType.HI3DGEN: HI3DGEN_ENHANCEMENT_FIELDS,
        }
        
    def _get_cache_key(self, prompt: str, model_type: ModelType, fields: Dict[str, Any]) -> str:
        """Generate cache key for enhanced prompt"""
        data = {
            "prompt": prompt,
            "model_type": model_type.value,
            "fields": fields,
            "llm_model": self.llm_model
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load enhanced prompt from cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Check if cache is still valid (24 hours)
                    cached_time = datetime.fromisoformat(data['timestamp'])
                    if datetime.now() - cached_time < timedelta(hours=24):
                        return data['enhanced_prompt']
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
        
    def _save_to_cache(self, cache_key: str, enhanced_prompt: str):
        """Save enhanced prompt to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'enhanced_prompt': enhanced_prompt,
                    'timestamp': datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
            
    async def enhance_with_llm(self, prompt: str, model_type: ModelType) -> Optional[str]:
        """Enhance prompt using LLM"""
        if not OLLAMA_AVAILABLE:
            return None
            
        template = TEMPLATES.get(model_type)
        if not template:
            return None
            
        try:
            # Format the template with user prompt
            llm_prompt = template.format(user_prompt=prompt)
            
            # Call Ollama
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.llm_model,
                prompt=llm_prompt,
                options={"temperature": 0.7, "num_predict": 500}
            )
            
            enhanced = response['response'].strip()
            
            # Clean up the response
            if enhanced.startswith('"') and enhanced.endswith('"'):
                enhanced = enhanced[1:-1]
                
            return enhanced
            
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return None
            
    def apply_field_enhancements(self, prompt: str, model_type: ModelType, fields: Dict[str, Any]) -> str:
        """Apply field-based enhancements to prompt"""
        field_config = self.field_configs.get(model_type, {})
        
        additions = []
        for field_id, value in fields.items():
            if field_id not in field_config or not value:
                continue
                
            field = field_config[field_id]
            
            # Handle different field types
            if field.get('type') == 'multi_checkbox':
                # Multi-checkbox: value is a list of selected options
                if isinstance(value, list):
                    for option in value:
                        if option in field.get('options', {}):
                            additions.append(field['options'][option])
            else:
                # Dropdown: value is a single selection
                if value in field.get('options', {}):
                    additions.append(field['options'][value])
                    
        # Combine prompt with additions
        if additions:
            return f"{prompt}, {', '.join(additions)}"
        return prompt
        
    async def enhance(self, prompt: str, model_type: ModelType, fields: Optional[Dict[str, Any]] = None, use_llm: bool = True) -> str:
        """
        Enhance a prompt using both LLM and field-based enhancements
        
        Args:
            prompt: Base prompt to enhance
            model_type: Type of model the prompt is for
            fields: UI field selections
            use_llm: Whether to use LLM enhancement
            
        Returns:
            Enhanced prompt
        """
        fields = fields or {}
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, model_type, fields)
        cached = self._load_from_cache(cache_key)
        if cached:
            logger.info("Using cached enhanced prompt")
            return cached
            
        enhanced_prompt = prompt
        
        # Apply LLM enhancement if enabled
        if use_llm:
            if OLLAMA_AVAILABLE:
                llm_enhanced = await self.enhance_with_llm(prompt, model_type)
                if llm_enhanced:
                    enhanced_prompt = llm_enhanced
                    logger.info("Applied LLM enhancement")
                else:
                    logger.warning("LLM enhancement failed, using original prompt")
            else:
                logger.debug("LLM enhancement requested but Ollama not available, skipping")
                
        # Apply field enhancements
        enhanced_prompt = self.apply_field_enhancements(enhanced_prompt, model_type, fields)
        logger.info("Applied field enhancements")
        
        # Cache the result
        self._save_to_cache(cache_key, enhanced_prompt)
        
        return enhanced_prompt
        
    def get_fields_for_model(self, model_type: ModelType) -> Dict[str, Any]:
        """Get field configuration for a specific model type"""
        return self.field_configs.get(model_type, {})