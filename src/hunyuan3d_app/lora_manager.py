"""LoRA (Low-Rank Adaptation) model management and loading"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
from diffusers import StableDiffusionXLPipeline, FluxPipeline
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


@dataclass
class LoRAInfo:
    """Information about a LoRA model"""
    name: str
    path: Path
    base_model: str  # SD1.5, SDXL, FLUX, etc.
    trigger_words: List[str]
    weight_default: float = 1.0
    weight_min: float = -2.0
    weight_max: float = 2.0
    description: str = ""
    metadata: Dict[str, Any] = None
    file_size_mb: float = 0.0
    
    @property
    def is_compatible_with_flux(self) -> bool:
        """Check if LoRA is compatible with FLUX models"""
        return self.base_model.upper() in ["FLUX", "FLUX.1", "FLUX.1-DEV", "FLUX.1-SCHNELL"]
    
    @property
    def is_compatible_with_sdxl(self) -> bool:
        """Check if LoRA is compatible with SDXL models"""
        return self.base_model.upper() in ["SDXL", "SDXL 1.0", "SDXL-TURBO"]


class LoRAManager:
    """Manages LoRA loading, stacking, and application to pipelines"""
    
    SUPPORTED_FORMATS = [".safetensors", ".pt", ".bin", ".ckpt"]
    
    def __init__(self, lora_dir: Path):
        self.lora_dir = Path(lora_dir)
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded LoRAs
        self.loaded_loras: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # Metadata cache
        self.metadata_file = self.lora_dir / "lora_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load LoRA metadata from cache"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save LoRA metadata to cache"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def scan_lora_directory(self) -> List[LoRAInfo]:
        """Scan directory for LoRA files and extract metadata"""
        loras = []
        
        for lora_path in self.lora_dir.rglob("*"):
            if lora_path.is_file() and lora_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    lora_info = self._extract_lora_info(lora_path)
                    if lora_info:
                        loras.append(lora_info)
                except Exception as e:
                    logger.error(f"Error processing {lora_path}: {e}")
                    
        return loras
    
    def _extract_lora_info(self, lora_path: Path) -> Optional[LoRAInfo]:
        """Extract information from a LoRA file"""
        try:
            # Check cached metadata first
            path_str = str(lora_path)
            if path_str in self.metadata:
                cached = self.metadata[path_str]
                return LoRAInfo(
                    name=cached.get("name", lora_path.stem),
                    path=lora_path,
                    base_model=cached.get("base_model", "SDXL"),
                    trigger_words=cached.get("trigger_words", []),
                    weight_default=cached.get("weight_default", 1.0),
                    weight_min=cached.get("weight_min", -2.0),
                    weight_max=cached.get("weight_max", 2.0),
                    description=cached.get("description", ""),
                    metadata=cached.get("metadata", {}),
                    file_size_mb=lora_path.stat().st_size / (1024 * 1024)
                )
            
            # Try to extract from file
            metadata = {}
            base_model = "SDXL"  # Default assumption
            
            if lora_path.suffix == ".safetensors":
                # Try to read safetensors metadata
                try:
                    from safetensors import safe_open
                    with safe_open(lora_path, framework="pt", device="cpu") as f:
                        metadata = f.metadata() or {}
                except:
                    pass
                    
            # Parse metadata for model info
            if "modelspec.architecture" in metadata:
                arch = metadata["modelspec.architecture"]
                if "flux" in arch.lower():
                    base_model = "FLUX"
                elif "xl" in arch.lower():
                    base_model = "SDXL"
                elif "sd15" in arch.lower() or "sd1.5" in arch.lower():
                    base_model = "SD1.5"
                    
            # Extract trigger words
            trigger_words = []
            if "trigger_words" in metadata:
                trigger_words = metadata["trigger_words"].split(",")
            elif "trigger" in metadata:
                trigger_words = [metadata["trigger"]]
                
            # Create LoRA info
            lora_info = LoRAInfo(
                name=metadata.get("name", lora_path.stem),
                path=lora_path,
                base_model=base_model,
                trigger_words=[t.strip() for t in trigger_words],
                weight_default=float(metadata.get("weight_default", 1.0)),
                weight_min=float(metadata.get("weight_min", -2.0)),
                weight_max=float(metadata.get("weight_max", 2.0)),
                description=metadata.get("description", ""),
                metadata=metadata,
                file_size_mb=lora_path.stat().st_size / (1024 * 1024)
            )
            
            # Cache metadata
            self.metadata[path_str] = {
                "name": lora_info.name,
                "base_model": lora_info.base_model,
                "trigger_words": lora_info.trigger_words,
                "weight_default": lora_info.weight_default,
                "weight_min": lora_info.weight_min,
                "weight_max": lora_info.weight_max,
                "description": lora_info.description,
                "metadata": lora_info.metadata
            }
            self._save_metadata()
            
            return lora_info
            
        except Exception as e:
            logger.error(f"Error extracting LoRA info from {lora_path}: {e}")
            return None
    
    def load_lora(self, lora_path: Path, device: str = "cuda") -> Dict[str, torch.Tensor]:
        """Load LoRA weights from file
        
        Args:
            lora_path: Path to LoRA file
            device: Device to load weights to
            
        Returns:
            Dictionary of LoRA weights
        """
        path_str = str(lora_path)
        
        # Check cache
        if path_str in self.loaded_loras:
            logger.info(f"Using cached LoRA: {lora_path.name}")
            return self.loaded_loras[path_str]
            
        try:
            logger.info(f"Loading LoRA: {lora_path}")
            
            if lora_path.suffix == ".safetensors":
                state_dict = load_file(lora_path, device=device)
            else:
                state_dict = torch.load(lora_path, map_location=device)
                
            # Process and validate LoRA weights
            lora_weights = {}
            for key, value in state_dict.items():
                # LoRA weights typically have patterns like:
                # - lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.alpha
                # - lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight
                # - lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight
                
                if "lora" in key.lower():
                    lora_weights[key] = value
                    
            if not lora_weights:
                raise ValueError("No LoRA weights found in file")
                
            # Cache loaded weights
            self.loaded_loras[path_str] = lora_weights
            
            logger.info(f"Loaded {len(lora_weights)} LoRA weight tensors")
            return lora_weights
            
        except Exception as e:
            logger.error(f"Error loading LoRA {lora_path}: {e}")
            raise
    
    def apply_lora_to_pipeline(
        self,
        pipeline: Union[StableDiffusionXLPipeline, FluxPipeline],
        lora_info: LoRAInfo,
        weight: float = 1.0,
        adapter_name: Optional[str] = None
    ) -> None:
        """Apply a single LoRA to a pipeline
        
        Args:
            pipeline: Diffusion pipeline
            lora_info: LoRA information
            weight: LoRA weight/scale
            adapter_name: Optional adapter name for the LoRA
        """
        try:
            # Load LoRA weights
            lora_weights = self.load_lora(lora_info.path)
            
            # Use adapter name or generate from path
            if adapter_name is None:
                adapter_name = lora_info.name or lora_info.path.stem
                
            # Apply to pipeline using diffusers LoRA support
            pipeline.load_lora_weights(
                lora_info.path.parent,
                weight_name=lora_info.path.name,
                adapter_name=adapter_name
            )
            
            # Set the scale
            pipeline.set_adapters(adapter_name, adapter_weights=weight)
            
            logger.info(f"Applied LoRA '{adapter_name}' with weight {weight}")
            
        except Exception as e:
            logger.error(f"Error applying LoRA: {e}")
            raise
    
    def apply_multiple_loras(
        self,
        pipeline: Union[StableDiffusionXLPipeline, FluxPipeline],
        lora_configs: List[Tuple[LoRAInfo, float]]
    ) -> None:
        """Apply multiple LoRAs to a pipeline with different weights
        
        Args:
            pipeline: Diffusion pipeline
            lora_configs: List of (LoRAInfo, weight) tuples
        """
        adapter_names = []
        adapter_weights = []
        
        for i, (lora_info, weight) in enumerate(lora_configs):
            adapter_name = f"{lora_info.name}_{i}"
            
            # Load each LoRA
            pipeline.load_lora_weights(
                lora_info.path.parent,
                weight_name=lora_info.path.name,
                adapter_name=adapter_name
            )
            
            adapter_names.append(adapter_name)
            adapter_weights.append(weight)
            
        # Apply all LoRAs with their weights
        pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
        
        logger.info(f"Applied {len(lora_configs)} LoRAs")
    
    def remove_loras_from_pipeline(
        self,
        pipeline: Union[StableDiffusionXLPipeline, FluxPipeline]
    ) -> None:
        """Remove all LoRAs from a pipeline"""
        try:
            pipeline.unload_lora_weights()
            logger.info("Removed all LoRAs from pipeline")
        except Exception as e:
            logger.error(f"Error removing LoRAs: {e}")
    
    def get_compatible_loras(
        self,
        base_model: str,
        lora_list: Optional[List[LoRAInfo]] = None
    ) -> List[LoRAInfo]:
        """Get LoRAs compatible with a specific base model
        
        Args:
            base_model: Base model type (FLUX, SDXL, etc.)
            lora_list: Optional list to filter (defaults to scanning directory)
            
        Returns:
            List of compatible LoRAs
        """
        if lora_list is None:
            lora_list = self.scan_lora_directory()
            
        compatible = []
        base_model_upper = base_model.upper()
        
        for lora in lora_list:
            if base_model_upper in ["FLUX", "FLUX.1", "FLUX.1-DEV", "FLUX.1-SCHNELL"]:
                if lora.is_compatible_with_flux:
                    compatible.append(lora)
            elif base_model_upper in ["SDXL", "SDXL 1.0", "SDXL-TURBO"]:
                if lora.is_compatible_with_sdxl:
                    compatible.append(lora)
            elif lora.base_model.upper() == base_model_upper:
                compatible.append(lora)
                
        return compatible
    
    def merge_loras(
        self,
        lora_configs: List[Tuple[LoRAInfo, float]],
        output_path: Path,
        base_model: Optional[str] = None
    ) -> bool:
        """Merge multiple LoRAs into a single file
        
        Args:
            lora_configs: List of (LoRAInfo, weight) tuples
            output_path: Where to save merged LoRA
            base_model: Optional base model override
            
        Returns:
            Success status
        """
        try:
            merged_state_dict = {}
            merged_metadata = {
                "merged_from": [],
                "base_model": base_model or lora_configs[0][0].base_model,
                "description": "Merged LoRA"
            }
            
            # Merge weights
            for lora_info, weight in lora_configs:
                lora_weights = self.load_lora(lora_info.path, device="cpu")
                
                for key, value in lora_weights.items():
                    if key in merged_state_dict:
                        # Add weighted values
                        merged_state_dict[key] = merged_state_dict[key] + (value * weight)
                    else:
                        merged_state_dict[key] = value * weight
                        
                merged_metadata["merged_from"].append({
                    "name": lora_info.name,
                    "weight": weight
                })
                
            # Save merged LoRA
            if output_path.suffix == ".safetensors":
                from safetensors.torch import save_file
                save_file(
                    merged_state_dict,
                    output_path,
                    metadata=merged_metadata
                )
            else:
                torch.save(merged_state_dict, output_path)
                
            logger.info(f"Merged {len(lora_configs)} LoRAs to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error merging LoRAs: {e}")
            return False
    
    def update_lora_metadata(
        self,
        lora_path: Path,
        name: Optional[str] = None,
        base_model: Optional[str] = None,
        trigger_words: Optional[List[str]] = None,
        description: Optional[str] = None,
        weight_default: Optional[float] = None
    ) -> bool:
        """Update metadata for a LoRA
        
        Args:
            lora_path: Path to LoRA file
            name: Optional new name
            base_model: Optional base model
            trigger_words: Optional trigger words
            description: Optional description
            weight_default: Optional default weight
            
        Returns:
            Success status
        """
        try:
            path_str = str(lora_path)
            
            # Get existing metadata or create new
            if path_str in self.metadata:
                meta = self.metadata[path_str]
            else:
                meta = {
                    "name": lora_path.stem,
                    "base_model": "SDXL",
                    "trigger_words": [],
                    "description": "",
                    "weight_default": 1.0,
                    "weight_min": -2.0,
                    "weight_max": 2.0
                }
                
            # Update fields
            if name is not None:
                meta["name"] = name
            if base_model is not None:
                meta["base_model"] = base_model
            if trigger_words is not None:
                meta["trigger_words"] = trigger_words
            if description is not None:
                meta["description"] = description
            if weight_default is not None:
                meta["weight_default"] = weight_default
                
            # Save
            self.metadata[path_str] = meta
            self._save_metadata()
            
            logger.info(f"Updated metadata for {lora_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            return False
    
    def clear_cache(self):
        """Clear loaded LoRA cache to free memory"""
        self.loaded_loras.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Cleared LoRA cache")