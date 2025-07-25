"""Face Swap UI

Handles face swapping with options to upload or generate target images.
"""

import gradio as gr
from pathlib import Path
from typing import Any
import logging
import time
from PIL import Image

from ...components.common import create_action_button

logger = logging.getLogger(__name__)


def create_face_swap_subtab(app: Any) -> None:
    """Create face swap interface with generate-then-swap option"""
    
    with gr.Column():
        gr.Markdown("### Swap faces in images with AI precision")
        
        # Mode selection
        mode = gr.Radio(
            choices=["Upload Both Images", "Generate Then Swap"],
            value="Upload Both Images",
            label="Mode",
            info="Choose whether to upload both images or generate the target image first"
        )
        
        with gr.Group(visible=True) as upload_mode_group:
            with gr.Row():
                # Source face image
                source_image = gr.Image(
                    label="Source Face",
                    type="pil",
                    height=300
                )
                
                # Target image
                target_image = gr.Image(
                    label="Target Image",
                    type="pil",
                    height=300
                )
        
        with gr.Group(visible=False) as generate_mode_group:
            # Generation prompt
            gen_prompt = gr.Textbox(
                label="Generate Target Image",
                placeholder="A professional business photo with formal attire...",
                lines=2
            )
            
            with gr.Row():
                # Model selection for generation
                downloaded_models = []
                try:
                    downloaded_models = app.model_manager.get_downloaded_models("image")
                except:
                    pass
                
                if downloaded_models:
                    gen_model = gr.Dropdown(
                        choices=downloaded_models,
                        value=downloaded_models[0] if downloaded_models else None,
                        label="Image Model"
                    )
                else:
                    gen_model = gr.Dropdown(
                        choices=[],
                        label="Image Model (No models downloaded)",
                        interactive=False
                    )
                
                generate_target_btn = create_action_button("🎨 Generate Target", size="sm")
            
            # Generated target display
            generated_target = gr.Image(
                label="Generated Target Image",
                type="pil",
                height=300,
                visible=False
            )
            
            # Source face for generate mode
            source_face_gen = gr.Image(
                label="Source Face",
                type="pil",
                height=300
            )
        
        # Swap settings
        with gr.Accordion("Face Swap Settings", open=True):
            with gr.Row():
                face_restore = gr.Checkbox(
                    label="Restore Face Quality",
                    value=True,
                    info="Apply face restoration after swapping"
                )
                
                blend_ratio = gr.Slider(
                    0.0, 1.0, 0.95,
                    step=0.05,
                    label="Blend Ratio",
                    info="How much of the source face to blend (1.0 = full swap)"
                )
            
            with gr.Row():
                face_index = gr.Number(
                    value=0,
                    label="Target Face Index",
                    info="Which face to swap in target image (0 = first/largest)"
                )
                
                similarity_threshold = gr.Slider(
                    0.0, 1.0, 0.6,
                    step=0.1,
                    label="Face Detection Threshold"
                )
        
        # Swap button
        swap_btn = create_action_button("🔄 Swap Faces", variant="primary")
        
        # Output
        with gr.Row():
            output_image = gr.Image(label="Result", type="pil")
            swap_info = gr.HTML()
        
        # Mode switching logic
        def switch_mode(selected_mode):
            if selected_mode == "Upload Both Images":
                return (
                    gr.update(visible=True),   # upload_mode_group
                    gr.update(visible=False),  # generate_mode_group
                    gr.update(visible=False)   # generated_target
                )
            else:
                return (
                    gr.update(visible=False),  # upload_mode_group
                    gr.update(visible=True),   # generate_mode_group
                    gr.update(visible=False)   # generated_target initially hidden
                )
        
        mode.change(
            switch_mode,
            inputs=[mode],
            outputs=[upload_mode_group, generate_mode_group, generated_target]
        )
        
        # Generate target image
        def generate_target_image(prompt, model_name, progress=gr.Progress()):
            """Generate target image for face swap"""
            if not prompt:
                return None, "❌ Please enter a prompt"
            if not model_name:
                return None, "❌ Please select a model"
            
            try:
                progress(0.1, "Generating target image...")
                
                # Submit generation job
                job_id = app.submit_generation_job(
                    job_type="image",
                    params={
                        "prompt": prompt,
                        "model_name": model_name,
                        "width": 512,
                        "height": 512,
                        "num_inference_steps": 30,
                        "guidance_scale": 7.5
                    }
                )
                
                # Wait for completion
                job = app.queue_manager.get_job(job_id)
                
                while job and job.status.value in ["pending", "running"]:
                    time.sleep(0.5)
                    job = app.queue_manager.get_job(job_id)
                    if job:
                        progress(0.1 + job.progress * 0.8, job.progress_message or "Generating...")
                
                if job and job.status.value == "completed" and job.result:
                    logger.info(f"Job result keys: {list(job.result.keys())}")
                    
                    # Try different possible keys for the image path
                    image_path = job.result.get("path") or job.result.get("image_path")
                    
                    # Also check if image is directly in result
                    image = job.result.get("image")
                    
                    if image_path and Path(image_path).exists():
                        image = Image.open(image_path)
                        progress(1.0, "Target image generated!")
                        return gr.update(value=image, visible=True), "✅ Target image generated!"
                    elif image:
                        progress(1.0, "Target image generated!")
                        return gr.update(value=image, visible=True), "✅ Target image generated!"
                    else:
                        logger.error(f"No valid image found in result. Path: {image_path}, Image: {image is not None}")
                
                error_msg = job.error if job else "Job not found"
                logger.error(f"Generation failed: {error_msg}")
                return None, f"❌ Generation failed: {error_msg}"
                
            except Exception as e:
                logger.error(f"Target generation error: {e}")
                return None, f"❌ Error: {str(e)}"
        
        generate_target_btn.click(
            generate_target_image,
            inputs=[gen_prompt, gen_model],
            outputs=[generated_target, swap_info]
        )
        
        # Main face swap function
        def swap_faces(mode, source_img, target_img, source_gen, generated,
                      restore, blend, face_idx, threshold, progress=gr.Progress()):
            """Perform face swap"""
            
            # Determine which images to use based on mode
            if mode == "Upload Both Images":
                if not source_img or not target_img:
                    return None, "❌ Please provide both source and target images"
                source_face = source_img
                target_face = target_img
            else:
                if not source_gen:
                    return None, "❌ Please provide source face image"
                if not generated:
                    return None, "❌ Please generate a target image first"
                source_face = source_gen
                target_face = generated
            
            try:
                progress(0.1, "Detecting faces...")
                
                # Submit face swap job
                job_id = app.submit_generation_job(
                    job_type="face_swap",
                    params={
                        "source_image": source_face,
                        "target_image": target_face,
                        "face_restore": restore,
                        "blend_ratio": blend,
                        "face_index": int(face_idx),
                        "similarity_threshold": threshold
                    }
                )
                
                progress(0.2, f"Job {job_id[:8]} submitted...")
                
                # Wait for completion
                job = app.queue_manager.get_job(job_id)
                
                while job and job.status.value in ["pending", "running"]:
                    time.sleep(0.5)
                    job = app.queue_manager.get_job(job_id)
                    if job:
                        progress(0.2 + job.progress * 0.7, job.progress_message or "Processing...")
                
                if job and job.status.value == "completed" and job.result:
                    logger.info(f"Face swap job completed. Result keys: {list(job.result.keys()) if isinstance(job.result, dict) else 'Not a dict'}")
                    # Face swap returns 'path', not 'output_path'
                    result_path = job.result.get("path") or job.result.get("output_path") or job.result.get("image_path")
                    logger.info(f"Looking for result at path: {result_path}")
                    
                    # Handle Windows paths in WSL environment
                    if result_path:
                        result_path_obj = Path(result_path)
                        if not result_path_obj.exists():
                            # Try to extract just the filename and look in outputs directory
                            filename = result_path_obj.name
                            alt_path = Path("outputs") / filename
                            if alt_path.exists():
                                result_path = str(alt_path)
                                logger.info(f"Using alternative path: {result_path}")
                    
                    if result_path and Path(result_path).exists():
                        result_image = Image.open(result_path)
                        
                        # Get additional info from job result
                        job_info = job.result.get('info', {})
                        
                        info = f"""
                        <div class="success-box">
                            <h4>✅ Face Swap Complete!</h4>
                            <ul>
                                <li><strong>Mode:</strong> {mode}</li>
                                <li><strong>Face Restoration:</strong> {'Enabled' if restore else 'Disabled'}</li>
                                <li><strong>Blend Ratio:</strong> {blend}</li>
                                <li><strong>Source Faces:</strong> {job_info.get('source_faces', 'N/A')}</li>
                                <li><strong>Target Faces:</strong> {job_info.get('target_faces', 'N/A')}</li>
                                <li><strong>Swapped Faces:</strong> {job_info.get('swapped_faces', 'N/A')}</li>
                                <li><strong>Processing Time:</strong> {job_info.get('processing_time', job.result.get('processing_time', 'N/A')):.2f}s</li>
                            </ul>
                        </div>
                        """
                        
                        progress(1.0, "Face swap complete!")
                        return result_image, info
                
                if job and job.error:
                    error = job.error
                elif job and job.status.value == "failed":
                    error = "Face swap failed but no error message provided"
                elif not job:
                    error = "Job not found"
                else:
                    error = f"Job status: {job.status.value if job else 'N/A'}"
                return None, f"❌ Face swap failed: {error}"
                
            except Exception as e:
                logger.error(f"Face swap error: {e}")
                return None, f"❌ Error: {str(e)}"
        
        swap_btn.click(
            swap_faces,
            inputs=[mode, source_image, target_image, source_face_gen, generated_target,
                   face_restore, blend_ratio, face_index, similarity_threshold],
            outputs=[output_image, swap_info]
        )