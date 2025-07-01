import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import gradio as gr
import torch
from PIL import Image

from .model_manager import ModelManager
from .image_generation import ImageGenerator
from ._3d_conversion import ThreeDConverter
from .system_checker import SystemRequirementsChecker, get_system_requirements_html

from .config import (
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
        self.three_d_converter = ThreeDConverter(CACHE_DIR, OUTPUT_DIR)

    def set_hf_token(self, token: str):
        return self.model_manager.set_hf_token(token)

    def get_model_status(self) -> str:
        return self.model_manager.get_model_status()

    def check_model_complete(self, model_path: Path, model_type: str, model_name: str) -> bool:
        return self.model_manager.check_model_complete(model_path, model_type, model_name)

    def check_missing_components(self, model_type: str, model_name: str) -> list:
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




    def load_image_model(self, model_name: str, progress=gr.Progress()) -> str:
        # Check if the model name has the "(not downloaded)" suffix and handle it
        if "(not downloaded)" in model_name:
            # Extract the actual model name
            actual_model_name = model_name.split(" (not downloaded)")[0]
            return f"❌ Model {actual_model_name} is not downloaded. Please download it first."

        load_status, self.image_model, self.image_model_name = self.model_manager.load_image_model(
            model_name, self.image_model, self.image_model_name, self.device, progress
        )
        return load_status

    def load_hunyuan3d_model(self, model_name: str, progress=gr.Progress()) -> str:
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

    def check_system_requirements(self) -> str:
        """Check system requirements and return HTML report"""
        return get_system_requirements_html()

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

            # For FLUX models, also check in src directory if not found in cache
            if not is_downloaded and name.startswith("FLUX"):
                model_path_src = self.model_manager.src_models_dir / "image" / name
                is_downloaded = self.model_manager.check_model_complete(model_path_src, "image", name)

            if is_downloaded:
                downloaded_image_models.append(name)

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

        # Always make the dropdown interactive
        image_model_update = gr.update(
            choices=image_model_choices_with_status,
            value=default_image_model,
            interactive=True
        )

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

        # Always make the dropdown interactive
        hunyuan_model_update = gr.update(
            choices=hunyuan_model_choices_with_status,
            value=default_hunyuan_model,
            interactive=True
        )

        # For manual pipeline dropdowns (same updates)
        manual_img_model_update = image_model_update
        manual_3d_model_update = hunyuan_model_update

        return image_model_update, hunyuan_model_update, manual_img_model_update, manual_3d_model_update



    def generate_image(
            self,
            prompt: str,
            negative_prompt: str,
            model_name: str,
            width: int,
            height: int,
            steps: int,
            guidance_scale: float,
            seed: int,
            progress=gr.Progress()
    ) -> Tuple[Image.Image, str]:
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
            image: Image.Image,
            model_name: str,
            remove_bg: bool,
            num_views: int,
            mesh_resolution: int,
            texture_resolution: int,
            progress=gr.Progress()
    ) -> Tuple[Optional[str], Optional[Image.Image], str]:
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
            prompt: str,
            negative_prompt: str,
            image_model: str,
            width: int,
            height: int,
            seed: int,
            quality_preset: str,
            hunyuan_model: str,
            keep_image_loaded: bool,
            save_intermediate: bool,
            only_generate_image: bool = False,
            progress=gr.Progress()
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[str], str]:
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
