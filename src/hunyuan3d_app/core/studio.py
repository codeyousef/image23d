import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import gradio as gr
import torch
from PIL import Image

from ..models.manager import ModelManager
from ..generation.image import ImageGenerator
# from ..generation.threed import ThreeDConverter  # Replaced by new 3D system
from ..utils.system import SystemRequirementsChecker, get_system_requirements_html

from ..config import (
    ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, QUALITY_PRESETS,
    MODELS_DIR, OUTPUT_DIR, CACHE_DIR
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Hunyuan3DStudio:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Models & State ---
        self.image_model = None
        self.image_model_name = None
        self.hunyuan3d_model = None
        self.hunyuan3d_model_name = None
        self.background_remover = None

        for dir_path in [MODELS_DIR, OUTPUT_DIR, CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.model_manager = ModelManager(MODELS_DIR, Path(__file__).parent / "models")
        self.image_generator = ImageGenerator(self.device, OUTPUT_DIR)
        # self.three_d_converter = ThreeDConverter(CACHE_DIR, OUTPUT_DIR)  # Replaced by new 3D system

    def set_hf_token(self, token):
        return self.model_manager.set_hf_token(token)

    def get_model_status(self):
        return self.model_manager.get_model_status()

    def check_model_complete(self, model_path, model_type, model_name):
        return self.model_manager.check_model_complete(model_path, model_type, model_name)

    def check_missing_components(self, model_type, model_name):
        """Check for missing components required for optimized image generation"""
        return self.model_manager.check_missing_components(model_type, model_name)

    def stop_download(self):
        return self.model_manager.stop_download()

    def stop_generation(self):
        """Stop the current image generation or 3D conversion process"""
        # Stop both image generation and 3D conversion
        image_result = self.image_generator.stop_generation()
        conversion_result = self.three_d_converter.stop_conversion()

        # Return a combined message
        return f"""
<div class="warning-box">
    <h4>⚠️ Processing Stopped</h4>
    <p>The current operation is being stopped. Please wait for it to complete.</p>
</div>
"""




    def load_image_model(self, model_name, progress):
        # Check if the model name has the "(not downloaded)" suffix and handle it
        if "(not downloaded)" in model_name:
            # Extract the actual model name
            actual_model_name = model_name.split(" (not downloaded)")[0]
            return f"❌ Model {actual_model_name} is not downloaded. Please download it first."

        load_status, self.image_model, self.image_model_name = self.model_manager.load_image_model(
            model_name, self.image_model, self.image_model_name, self.device, progress
        )
        return load_status

    def load_hunyuan3d_model(self, model_name, progress):
        # Check if the model name has the "(not downloaded)" suffix and handle it
        if "(not downloaded)" in model_name:
            # Extract the actual model name
            actual_model_name = model_name.split(" (not downloaded)")[0]
            return f"❌ Model {actual_model_name} is not downloaded. Please download it first."

        load_status, self.hunyuan3d_model, self.hunyuan3d_model_name = self.model_manager.load_hunyuan3d_model(
            model_name, self.hunyuan3d_model, self.hunyuan3d_model_name, self.device, progress
        )
        return load_status

    def unload_image_model(self):
        self.model_manager.unload_image_model()

    def unload_3d_model(self):
        self.model_manager.unload_3d_model()

    def unload_models(self):
        self.model_manager.unload_models()

    def check_system_requirements(self):
        """Check system requirements and return HTML report"""
        return get_system_requirements_html()
    
    def download_model(self, model_name, progress):
        """Download a standard image model"""
        return self.model_manager.download_model("image", model_name, progress)
    
    def download_gguf_model(self, model_name, force_redownload, progress):
        """Download a GGUF model and components - generator version for UI updates"""
        from .config import GGUF_IMAGE_MODELS
        
        # Start with initial message
        yield """
<div class="info-box">
    <h4>🚀 Starting GGUF Model Download</h4>
    <p>Preparing to download model components...</p>
</div>
"""
        
        if model_name in GGUF_IMAGE_MODELS:
            config = GGUF_IMAGE_MODELS[model_name]
            model_path = self.model_manager.models_dir / "gguf" / model_name
            
            # If force redownload, delete existing first
            if force_redownload and model_path.exists():
                import shutil
                shutil.rmtree(model_path)
                if progress:
                    progress(0.05, desc="Removed existing model for re-download...")
                yield """
<div class="info-box">
    <h4>🗑️ Cleaned Up</h4>
    <p>Removed existing model for fresh download...</p>
</div>
"""
            
            # Delegate to model manager's GGUF download method
            # We'll create a generator version of this
            yield from self.model_manager.download_gguf_model(model_name, force_redownload, progress)
        else:
            yield f"❌ Unknown GGUF model: {model_name}"
    
    def download_component(self, component_name, progress):
        """Download a FLUX component (VAE, text encoder)"""
        from .config import FLUX_COMPONENTS
        if component_name in FLUX_COMPONENTS:
            comp = FLUX_COMPONENTS[component_name]
            try:
                from huggingface_hub import hf_hub_download
                
                # Determine target directory
                if component_name == "vae":
                    target_dir = self.model_manager.models_dir / "vae"
                else:
                    target_dir = self.model_manager.models_dir / "text_encoders"
                
                target_dir.mkdir(parents=True, exist_ok=True)
                file_path = target_dir / comp["filename"]
                
                if file_path.exists():
                    return f"✅ {comp['name']} already downloaded"
                
                if progress:
                    progress(0.5, desc=f"Downloading {comp['name']}...")
                hf_hub_download(
                    repo_id=comp["repo_id"],
                    filename=comp["filename"],
                    local_dir=target_dir,
                    token=self.model_manager.hf_token
                )
                
                return f"✅ Successfully downloaded {comp['name']}"
                
            except Exception as e:
                return f"❌ Failed to download {comp['name']}: {str(e)}"
        
        return f"❌ Unknown component: {component_name}"
    
    def delete_model(self, model_type, model_name):
        """Delete a downloaded model"""
        import shutil
        
        try:
            models_deleted = []
            
            if model_type == "image":
                # Get model config to find repo_id
                from .config import ALL_IMAGE_MODELS
                model_config = ALL_IMAGE_MODELS.get(model_name, {})
                repo_id = model_config.get("repo_id", "")
                
                # Check standard directory structure first
                model_path = self.model_manager.models_dir / "image" / model_name
                
                # For FLUX models, also check src directory if not found in cache
                if not model_path.exists() and model_name.startswith("FLUX"):
                    model_path = self.model_manager.src_models_dir / "image" / model_name
                
                if model_path.exists():
                    shutil.rmtree(model_path)
                    models_deleted.append(str(model_path))
                
                # Also check for HuggingFace cache directory structure
                # When using cache_dir with snapshot_download, HF creates:
                # cache_dir/models--{org}--{model}/snapshots/{hash}/
                if repo_id:
                    # Convert repo_id (e.g., "black-forest-labs/FLUX.1-schnell") 
                    # to HF cache format (e.g., "models--black-forest-labs--FLUX.1-schnell")
                    hf_cache_name = f"models--{repo_id.replace('/', '--')}"
                    
                    # Check in the image model cache directory
                    hf_cache_path = self.model_manager.models_dir / "image" / model_name / hf_cache_name
                    if hf_cache_path.exists():
                        shutil.rmtree(hf_cache_path)
                        models_deleted.append(str(hf_cache_path))
                    
                    # Also check if the entire model directory contains only HF cache
                    parent_path = self.model_manager.models_dir / "image" / model_name
                    if parent_path.exists():
                        # Check if it only contains HF cache directories
                        subdirs = [d for d in parent_path.iterdir() if d.is_dir()]
                        if all(d.name.startswith("models--") for d in subdirs):
                            shutil.rmtree(parent_path)
                            models_deleted.append(str(parent_path))
                            
            elif model_type == "3d":
                # Get model config to find repo_id
                from .config import HUNYUAN3D_MODELS
                model_config = HUNYUAN3D_MODELS.get(model_name, {})
                repo_id = model_config.get("repo_id", "")
                
                model_path = self.model_manager.models_dir / "3d" / model_name
                if model_path.exists():
                    shutil.rmtree(model_path)
                    models_deleted.append(str(model_path))
                
                # Also check for HuggingFace cache directory structure
                if repo_id:
                    hf_cache_name = f"models--{repo_id.replace('/', '--')}"
                    hf_cache_path = self.model_manager.models_dir / "3d" / model_name / hf_cache_name
                    if hf_cache_path.exists():
                        shutil.rmtree(hf_cache_path)
                        models_deleted.append(str(hf_cache_path))
                    
                    # Also check if the entire model directory contains only HF cache
                    parent_path = self.model_manager.models_dir / "3d" / model_name
                    if parent_path.exists():
                        subdirs = [d for d in parent_path.iterdir() if d.is_dir()]
                        if all(d.name.startswith("models--") for d in subdirs):
                            shutil.rmtree(parent_path)
                            models_deleted.append(str(parent_path))
            else:
                return f"❌ Unknown model type: {model_type}"
            
            if models_deleted:
                return f"✅ Successfully deleted {model_name} from:\n" + "\n".join(f"  • {path}" for path in models_deleted)
            else:
                return f"❌ Model {model_name} not found"
                
        except Exception as e:
            return f"❌ Failed to delete {model_name}: {str(e)}"
    
    def delete_gguf_model(self, model_name):
        """Delete a GGUF model and optionally its components"""
        import shutil
        from .config import GGUF_IMAGE_MODELS
        
        try:
            deleted_items = []
            
            # Delete standard GGUF model directory
            gguf_path = self.model_manager.models_dir / "gguf" / model_name
            if gguf_path.exists():
                shutil.rmtree(gguf_path)
                deleted_items.append(f"standard directory: {gguf_path}")
            
            # Also check for HuggingFace cache structure
            if model_name in GGUF_IMAGE_MODELS:
                config = GGUF_IMAGE_MODELS[model_name]
                repo_id = config.repo_id
                
                # Convert repo_id to HF cache format
                hf_cache_name = f"models--{repo_id.replace('/', '--')}"
                hf_cache_path = self.model_manager.models_dir / "gguf" / model_name / hf_cache_name
                
                if hf_cache_path.exists():
                    shutil.rmtree(hf_cache_path)
                    deleted_items.append(f"HF cache: {hf_cache_path}")
            
            if deleted_items:
                return f"✅ Successfully deleted GGUF model {model_name} ({', '.join(deleted_items)}). Components (VAE, text encoders) preserved for other models."
            else:
                return f"❌ GGUF model {model_name} not found"
                
        except Exception as e:
            return f"❌ Failed to delete GGUF model {model_name}: {str(e)}"
    
    def cleanup_orphaned_caches(self):
        """Clean up orphaned cache files from old directory structures"""
        import shutil
        import os
        
        cleanup_results = []
        total_freed_space = 0
        
        try:
            # Check for old HuggingFace cache in user home directory
            hf_cache_home = Path.home() / ".cache" / "huggingface" / "hub"
            if hf_cache_home.exists():
                # Look for models that might be duplicated in our local cache
                for item in hf_cache_home.iterdir():
                    if item.is_dir() and item.name.startswith("models--"):
                        # Check if this model exists in our local cache
                        model_name_parts = item.name.replace("models--", "").split("--")
                        if len(model_name_parts) >= 2:
                            # Check if we have this model locally
                            for model_type in ["image", "3d", "gguf"]:
                                local_paths = []
                                
                                # Check various possible local paths
                                if model_type == "image":
                                    # Check for FLUX models
                                    if "FLUX" in item.name or "flux" in item.name:
                                        for flux_name in ["FLUX.1-dev", "FLUX.1-schnell"]:
                                            local_paths.append(self.model_manager.models_dir / model_type / flux_name)
                                            local_paths.append(self.model_manager.src_models_dir / model_type / flux_name)
                                
                                # Check if any local path exists with HF cache structure
                                for local_path in local_paths:
                                    hf_cache_local = local_path / item.name
                                    if hf_cache_local.exists():
                                        # We have this model locally, safe to suggest cleanup
                                        size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                                        size_gb = size / (1024**3)
                                        cleanup_results.append({
                                            "path": str(item),
                                            "size_gb": size_gb,
                                            "reason": "Duplicate in HuggingFace cache (model exists locally)"
                                        })
                                        total_freed_space += size_gb
                                        break
            
            # Check for orphaned src/models directory
            src_models_path = Path(__file__).parent / "models"
            if src_models_path.exists():
                # Check if any models exist here that also exist in cache
                for model_type in ["image", "3d"]:
                    type_path = src_models_path / model_type
                    if type_path.exists():
                        for model_dir in type_path.iterdir():
                            if model_dir.is_dir():
                                # Check if this model exists in cache
                                cache_path = self.model_manager.models_dir / model_type / model_dir.name
                                if cache_path.exists() or (cache_path.parent / f"models--{model_dir.name.replace('/', '--')}").exists():
                                    size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                                    size_gb = size / (1024**3)
                                    cleanup_results.append({
                                        "path": str(model_dir),
                                        "size_gb": size_gb,
                                        "reason": "Duplicate in src/models (model exists in cache)"
                                    })
                                    total_freed_space += size_gb
            
            if cleanup_results:
                result_text = f"<div class='info-box'><h4>🧹 Found {len(cleanup_results)} orphaned items</h4>"
                result_text += f"<p>Total space that can be freed: {total_freed_space:.2f} GB</p>"
                result_text += "<p>Orphaned items found:</p><ul>"
                
                for item in cleanup_results:
                    result_text += f"<li>{item['path']} ({item['size_gb']:.2f} GB) - {item['reason']}</li>"
                
                result_text += "</ul>"
                result_text += "<p><strong>Note:</strong> These are duplicate files that exist in multiple locations. "
                result_text += "You can safely delete them from the HuggingFace cache or src/models directories if the models are working correctly from the models directory.</p>"
                result_text += "</div>"
                
                return result_text
            else:
                return "<div class='success-box'><h4>✅ No orphaned cache files found</h4><p>Your model directories are clean!</p></div>"
                
        except Exception as e:
            return f"<div class='error-box'><h4>❌ Error during cleanup scan</h4><p>{str(e)}</p></div>"
    
    def delete_component(self, component_name):
        """Delete a FLUX component"""
        import os
        from .config import FLUX_COMPONENTS
        
        if component_name not in FLUX_COMPONENTS:
            return f"❌ Unknown component: {component_name}"
            
        try:
            comp = FLUX_COMPONENTS[component_name]
            
            # Determine file path
            if component_name == "vae":
                file_path = self.model_manager.models_dir / "vae" / comp["filename"]
            else:
                file_path = self.model_manager.models_dir / "text_encoders" / comp["filename"]
            
            if file_path.exists():
                os.remove(file_path)
                return f"✅ Successfully deleted {comp['name']}"
            else:
                return f"❌ Component {comp['name']} not found"
                
        except Exception as e:
            return f"❌ Failed to delete component: {str(e)}"

    def get_model_selection_data(self):
        """Returns updated choices and disabled status for model dropdowns."""
        # Get all model names
        image_model_choices = list(ALL_IMAGE_MODELS.keys())
        hunyuan_model_choices = list(HUNYUAN3D_MODELS.keys())

        # Track which models are downloaded
        downloaded_image_models = []
        downloaded_hunyuan_models = []

        # Check which image models are downloaded
        for name in image_model_choices:
            # First check in cache directory
            model_path_cache = self.model_manager.models_dir / "image" / name
            is_downloaded = self.model_manager.check_model_complete(model_path_cache, "image", name)
            logger.debug(f"Checking {name}: cache path {model_path_cache}, downloaded: {is_downloaded}")

            # For FLUX models, also check in src directory if not found in cache
            if not is_downloaded and name.startswith("FLUX"):
                model_path_src = self.model_manager.src_models_dir / "image" / name
                is_downloaded = self.model_manager.check_model_complete(model_path_src, "image", name)
                logger.debug(f"Checking {name}: src path {model_path_src}, downloaded: {is_downloaded}")

            if is_downloaded:
                downloaded_image_models.append(name)
                logger.info(f"Model {name} is downloaded")

        # Check which 3D models are downloaded
        for name in hunyuan_model_choices:
            model_path = self.model_manager.models_dir / "3d" / name
            is_downloaded = self.model_manager.check_model_complete(model_path, "3d", name)
            if is_downloaded:
                downloaded_hunyuan_models.append(name)

        # Create dropdown updates with appropriate values
        # For image models - show all models but mark non-downloaded ones
        image_model_choices_with_status = []
        for name in image_model_choices:
            if name in downloaded_image_models:
                image_model_choices_with_status.append(name)
            else:
                # Add a disabled indicator to non-downloaded models
                image_model_choices_with_status.append(f"{name} (not downloaded)")

        default_image_model = None
        if downloaded_image_models:
            default_image_model = downloaded_image_models[0]

        # For 3D models - show all models but mark non-downloaded ones
        hunyuan_model_choices_with_status = []
        for name in hunyuan_model_choices:
            if name in downloaded_hunyuan_models:
                hunyuan_model_choices_with_status.append(name)
            else:
                # Add a disabled indicator to non-downloaded models
                hunyuan_model_choices_with_status.append(f"{name} (not downloaded)")

        default_hunyuan_model = None
        if downloaded_hunyuan_models:
            default_hunyuan_model = downloaded_hunyuan_models[0]

        # Return simple values instead of gr.update objects to avoid schema issues
        return (
            image_model_choices_with_status,
            hunyuan_model_choices_with_status,
            image_model_choices_with_status,
            hunyuan_model_choices_with_status
        )



    def generate_image(
            self,
            prompt,
            negative_prompt,
            model_name,
            width,
            height,
            steps,
            guidance_scale,
            seed,
            progress
    ):
        # Check if the model name has the "(not downloaded)" suffix and handle it
        if "(not downloaded)" in model_name:
            # Extract the actual model name
            actual_model_name = model_name.split(" (not downloaded)")[0]
            return None, f"""
<div class="error-box">
    <h4>❌ Model Not Downloaded</h4>
    <p>The model <strong>{actual_model_name}</strong> is not downloaded.</p>
    <p>Please go to the Model Manager tab and download it first.</p>
</div>
"""

        # Check for missing components
        missing_components = self.check_missing_components("image", model_name)
        if missing_components:
            # Create a user-friendly message about missing components
            if "complete model" in missing_components:
                return None, f"""
<div class="error-box">
    <h4>❌ Model Not Downloaded</h4>
    <p>The model <strong>{model_name}</strong> is not downloaded.</p>
    <p>Please go to the Model Manager tab and download it first.</p>
</div>
"""
            else:
                components_list = ", ".join(missing_components)
                return None, f"""
<div class="error-box">
    <h4>❌ Missing Components</h4>
    <p>The model <strong>{model_name}</strong> is missing required components for optimized image generation:</p>
    <ul>
        <li>{components_list}</li>
    </ul>
    <p>You can either:</p>
    <ol>
        <li>Go to the <strong>Model Manager</strong> tab → <strong>Download Missing Components</strong> section to download only the missing components</li>
        <li>Or re-download the entire model with the "Force re-download" option checked</li>
    </ol>
</div>
"""

        # Load model if needed
        if not self.image_model or self.image_model_name != model_name:
            load_status = self.load_image_model(model_name, progress)
            if load_status and "❌" in load_status:
                return None, load_status

        return self.image_generator.generate_image(
            image_model=self.image_model,
            image_model_name=self.image_model_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            progress=progress
        )

    def convert_to_3d(
            self,
            image,
            model_name,
            remove_bg,
            num_views,
            mesh_resolution,
            texture_resolution,
            progress
    ):
        # Load model if needed
        if not self.hunyuan3d_model or self.hunyuan3d_model_name != model_name:
            load_status = self.load_hunyuan3d_model(model_name, progress)
            if load_status and "❌" in load_status:
                return None, None, load_status

        # Remove background if requested
        if remove_bg:
            progress(0.2, desc="Removing background...")
            image = self.image_generator.remove_background(image)

        return self.three_d_converter.convert_to_3d(
            hunyuan3d_model=self.hunyuan3d_model,
            hunyuan3d_model_name=self.hunyuan3d_model_name,
            image=image,
            remove_bg=remove_bg,
            num_views=num_views,
            mesh_resolution=mesh_resolution,
            texture_resolution=texture_resolution,
            progress=progress
        )

    def full_pipeline(
            self,
            prompt,
            negative_prompt,
            image_model,
            width,
            height,
            seed,
            quality_preset,
            hunyuan_model,
            keep_image_loaded,
            save_intermediate,
            only_generate_image,
            progress
    ):
        """Run the complete text-to-3D pipeline or just generate an image if only_generate_image is True"""
        try:
            # Check for missing components in the image model
            missing_components = self.check_missing_components("image", image_model)
            if missing_components:
                # Create a user-friendly message about missing components
                if "complete model" in missing_components:
                    return None, None, None, f"""
<div class="error-box">
    <h4>❌ Image Model Not Downloaded</h4>
    <p>The model <strong>{image_model}</strong> is not downloaded.</p>
    <p>Please go to the Model Manager tab and download it first.</p>
</div>
"""
                else:
                    components_list = ", ".join(missing_components)
                    return None, None, None, f"""
<div class="error-box">
    <h4>❌ Missing Components</h4>
    <p>The image model <strong>{image_model}</strong> is missing required components for optimized image generation:</p>
    <ul>
        <li>{components_list}</li>
    </ul>
    <p>You can either:</p>
    <ol>
        <li>Go to the <strong>Model Manager</strong> tab → <strong>Download Missing Components</strong> section to download only the missing components</li>
        <li>Or re-download the entire model with the "Force re-download" option checked</li>
    </ol>
</div>
"""

            # If not only generating an image, also check 3D model components
            if not only_generate_image:
                missing_3d_components = self.check_missing_components("3d", hunyuan_model)
                if missing_3d_components:
                    if "complete model" in missing_3d_components:
                        return None, None, None, f"""
<div class="error-box">
    <h4>❌ 3D Model Not Downloaded</h4>
    <p>The 3D model <strong>{hunyuan_model}</strong> is not downloaded.</p>
    <p>Please go to the Model Manager tab and download it first.</p>
</div>
"""

            preset = QUALITY_PRESETS[quality_preset]

            # Step 1: Generate image
            progress(0.0, desc="Starting image generation...")
            image, img_info = self.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_name=image_model,
                width=width,
                height=height,
                steps=preset.image_steps,
                guidance_scale=preset.image_guidance,
                seed=seed,
                progress=progress
            )

            if image is None:
                return None, None, None, img_info

            # If only generating image, skip 3D conversion
            if only_generate_image:
                # Unload image model if requested
                if not keep_image_loaded:
                    self.unload_image_model()

                return image, None, None, img_info

            # Step 2: Convert to 3D
            progress(0.5, desc="Converting to 3D...")
            mesh_path, preview, model_info = self.convert_to_3d(
                image=image,
                model_name=hunyuan_model,
                num_views=preset.num_3d_views,
                mesh_resolution=preset.mesh_resolution,
                texture_resolution=preset.texture_resolution,
                progress=progress
            )

            # Unload image model if requested
            if not keep_image_loaded:
                self.unload_image_model()

            # Combine info
            full_info = f"""
<div class="pipeline-info">
    <h3>🎉 Pipeline Complete!</h3>
    {img_info}
    {model_info if mesh_path else ""}
</div>
"""

            return image, preview, mesh_path, full_info

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            return None, None, None, f"❌ Pipeline error: {str(e)}"
