"""Face swap tab for enhanced UI"""

import gradio as gr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...features.face_swap import FaceRestoreModel, BlendMode, FaceSwapParams


def create_face_swap_tab(app: Any) -> None:
    """Create the face swap tab
    
    Args:
        app: The enhanced application instance
    """
    gr.Markdown("""
    ### 🔄 Advanced Face Swap Studio
    Swap faces in images and videos with professional-grade quality.
    """)
    
    with gr.Tabs():
        # Single Image Face Swap
        with gr.Tab("🖼️ Image Face Swap"):
            with gr.Row():
                with gr.Column():
                    # Input images
                    source_image = gr.Image(
                        label="Source Face",
                        type="pil"
                    )
                    
                    target_image = gr.Image(
                        label="Target Image",
                        type="pil"
                    )
                    
                    # Basic parameters
                    with gr.Accordion("Swap Settings", open=True):
                        source_face_index = gr.Number(
                            value=0,
                            label="Source Face Index"
                        )
                        gr.Markdown("*Which face to use from source (0 = first face)*")
                        
                        target_face_index = gr.Number(
                            value=-1,
                            label="Target Face Index"
                        )
                        gr.Markdown("*Which face to replace (-1 = all faces)*")
                        
                        similarity_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.6,
                            step=0.1,
                            label="Face Similarity Threshold"
                        )
                        gr.Markdown("*Only swap faces above this similarity*")
                        
                        blend_mode = gr.Radio(
                            choices=["seamless", "hard", "soft", "poisson"],
                            value="seamless",
                            label="Blend Mode"
                        )
                        
                    # Enhancement options
                    with gr.Accordion("Enhancement Options", open=True):
                        face_restore = gr.Checkbox(
                            label="Face Restoration",
                            value=True
                        )
                        
                        face_restore_model = gr.Dropdown(
                            choices=["CodeFormer", "GFPGAN", "RestoreFormer"],
                            value="CodeFormer",
                            label="Restoration Model",
                            visible=True
                        )
                        
                        face_restore_fidelity = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Restoration Fidelity",
                            visible=True
                        )
                        gr.Markdown("*0 = Natural, 1 = Enhanced*")
                        
                        background_enhance = gr.Checkbox(
                            label="Enhance Background",
                            value=False
                        )
                        
                        face_upsample = gr.Checkbox(
                            label="Upsample Face",
                            value=True
                        )
                        
                        upscale_factor = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=2,
                            step=1,
                            label="Upscale Factor",
                            visible=True
                        )
                        
                    # Advanced options
                    with gr.Accordion("Advanced Options", open=False):
                        preserve_expression = gr.Checkbox(
                            label="Preserve Original Expression",
                            value=False
                        )
                        
                        expression_weight = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="Expression Weight",
                            visible=False
                        )
                        
                        preserve_lighting = gr.Checkbox(
                            label="Preserve Original Lighting",
                            value=True
                        )
                        
                        lighting_weight = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Lighting Weight",
                            visible=True
                        )
                        
                        preserve_age = gr.Checkbox(
                            label="Preserve Original Age",
                            value=True
                        )
                        
                        age_weight = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Age Weight",
                            visible=True
                        )
                        
                    swap_image_btn = gr.Button(
                        "🔄 Swap Face",
                        variant="primary",
                        size="lg"
                    )
                    
                with gr.Column():
                    # Output
                    swapped_image = gr.Image(
                        label="Result",
                        type="pil"
                    )
                    
                    swap_info = gr.HTML()
                    
                    # Face detection preview
                    with gr.Accordion("Face Detection Preview", open=False):
                        detect_source_btn = gr.Button("Detect Source Faces")
                        detect_target_btn = gr.Button("Detect Target Faces")
                        detection_info = gr.HTML()
                        
        # Video Face Swap
        with gr.Tab("🎬 Video Face Swap"):
            with gr.Row():
                with gr.Column():
                    # Input source and video
                    video_source_image = gr.Image(
                        label="Source Face",
                        type="pil"
                    )
                    
                    target_video = gr.Video(
                        label="Target Video"
                    )
                    
                    # Video parameters
                    with gr.Accordion("Video Settings", open=True):
                        video_source_face_index = gr.Number(
                            value=0,
                            label="Source Face Index"
                        )
                        
                        video_target_face_index = gr.Number(
                            value=-1,
                            label="Target Face Index (-1 = all)"
                        )
                        
                        temporal_smoothing = gr.Checkbox(
                            label="Temporal Smoothing",
                            value=True
                        )
                        gr.Markdown("*Reduce flicker between frames*")
                        
                        smoothing_window = gr.Slider(
                            minimum=3,
                            maximum=15,
                            value=5,
                            step=2,
                            label="Smoothing Window",
                            visible=True
                        )
                        
                    # Use same enhancement options as image
                    video_face_restore = gr.Checkbox(
                        label="Face Restoration",
                        value=True
                    )
                    
                    video_quality = gr.Radio(
                        choices=["fast", "balanced", "quality"],
                        value="balanced",
                        label="Processing Quality"
                    )
                    
                    swap_video_btn = gr.Button(
                        "🎬 Process Video",
                        variant="primary",
                        size="lg"
                    )
                    
                with gr.Column():
                    # Output
                    swapped_video = gr.Video(
                        label="Result"
                    )
                    
                    video_progress = gr.HTML()
                    video_info = gr.HTML()
                    
        # Batch Processing
        with gr.Tab("📦 Batch Processing"):
            gr.Markdown("""
            Process multiple images at once with the same source face.
            """)
            
            with gr.Row():
                with gr.Column():
                    batch_source_image = gr.Image(
                        label="Source Face",
                        type="pil"
                    )
                    
                    batch_target_images = gr.File(
                        label="Target Images",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    
                    # Batch settings
                    batch_params_same = gr.Checkbox(
                        label="Use Same Settings for All",
                        value=True
                    )
                    
                    batch_face_restore = gr.Checkbox(
                        label="Face Restoration",
                        value=True
                    )
                    
                    batch_quality = gr.Radio(
                        choices=["fast", "balanced", "quality"],
                        value="balanced",
                        label="Processing Quality"
                    )
                    
                    batch_process_btn = gr.Button(
                        "📦 Process Batch",
                        variant="primary",
                        size="lg"
                    )
                    
                with gr.Column():
                    batch_progress = gr.HTML()
                    batch_results = gr.Gallery(
                        label="Results",
                        columns=2,
                        height="auto"
                    )
                    batch_download = gr.File(
                        label="Download Results",
                        visible=False
                    )
                    
    # Helper functions
    def toggle_restore_options(face_restore):
        """Toggle face restoration options visibility"""
        return (
            gr.update(visible=face_restore),
            gr.update(visible=face_restore)
        )
        
    face_restore.change(
        toggle_restore_options,
        inputs=[face_restore],
        outputs=[face_restore_model, face_restore_fidelity]
    )
    
    def toggle_expression_options(preserve_expr):
        """Toggle expression preservation options"""
        return gr.update(visible=preserve_expr)
        
    preserve_expression.change(
        toggle_expression_options,
        inputs=[preserve_expression],
        outputs=[expression_weight]
    )
    
    def toggle_lighting_options(preserve_light):
        """Toggle lighting preservation options"""
        return gr.update(visible=preserve_light)
        
    preserve_lighting.change(
        toggle_lighting_options,
        inputs=[preserve_lighting],
        outputs=[lighting_weight]
    )
    
    def toggle_age_options(preserve_age_val):
        """Toggle age preservation options"""
        return gr.update(visible=preserve_age_val)
        
    preserve_age.change(
        toggle_age_options,
        inputs=[preserve_age],
        outputs=[age_weight]
    )
    
    def toggle_upscale_options(face_upsample_val):
        """Toggle upscale options"""
        return gr.update(visible=face_upsample_val)
        
    face_upsample.change(
        toggle_upscale_options,
        inputs=[face_upsample],
        outputs=[upscale_factor]
    )
    
    def toggle_smoothing_options(temporal_smooth):
        """Toggle temporal smoothing options"""
        return gr.update(visible=temporal_smooth)
        
    temporal_smoothing.change(
        toggle_smoothing_options,
        inputs=[temporal_smoothing],
        outputs=[smoothing_window]
    )
    
    # Face detection preview
    def detect_faces(image, image_type):
        """Detect and preview faces in an image"""
        try:
            if not image:
                return "<p>Please upload an image first</p>"
                
            faces = app.face_swap_manager.detect_faces(image)
            
            if not faces:
                return f"<p>No faces detected in {image_type} image</p>"
                
            info_html = f"<h4>Detected {len(faces)} face(s) in {image_type} image:</h4><ul>"
            
            for i, face in enumerate(faces):
                info_html += f"""
                <li>
                    <strong>Face {i}:</strong>
                    <ul>
                        <li>Confidence: {face.det_score:.2%}</li>
                        <li>Bounding Box: {face.bbox.astype(int).tolist()}</li>
                        {f'<li>Age: ~{face.age}</li>' if face.age else ''}
                        {f'<li>Gender: {face.gender}</li>' if face.gender else ''}
                    </ul>
                </li>
                """
                
            info_html += "</ul>"
            return info_html
            
        except Exception as e:
            return f"<p style='color: red;'>Error detecting faces: {str(e)}</p>"
            
    detect_source_btn.click(
        lambda img: detect_faces(img, "source"),
        inputs=[source_image],
        outputs=[detection_info]
    )
    
    detect_target_btn.click(
        lambda img: detect_faces(img, "target"),
        inputs=[target_image],
        outputs=[detection_info]
    )
    
    # Main face swap function
    def swap_face_image(
        source_img, target_img, 
        source_idx, target_idx, similarity, blend,
        restore, restore_model, fidelity, bg_enhance, upsample, upscale,
        preserve_expr, expr_weight, preserve_light, light_weight,
        preserve_age_val, age_weight_val
    ):
        """Perform face swap on images"""
        try:
            if not source_img or not target_img:
                return None, "<p style='color: red;'>Please upload both source and target images</p>"
                
            # Initialize models if needed
            if not app.face_swap_manager.models_loaded:
                success, msg = app.face_swap_manager.initialize_models()
                if not success:
                    return None, f"<p style='color: red;'>Failed to initialize models: {msg}</p>"
                    
            # Create parameters
            params = FaceSwapParams(
                source_face_index=int(source_idx),
                target_face_index=int(target_idx),
                similarity_threshold=similarity,
                blend_mode=BlendMode(blend),
                face_restore=restore,
                face_restore_model=FaceRestoreModel[restore_model.upper()],
                face_restore_fidelity=fidelity,
                background_enhance=bg_enhance,
                face_upsample=upsample,
                upscale_factor=int(upscale),
                preserve_expression=preserve_expr,
                expression_weight=expr_weight,
                preserve_lighting=preserve_light,
                lighting_weight=light_weight,
                preserve_age=preserve_age_val,
                age_weight=age_weight_val
            )
            
            # Perform swap
            result_img, info = app.face_swap_manager.swap_face(
                source_image=source_img,
                target_image=target_img,
                params=params
            )
            
            if result_img:
                info_html = f"""
                <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                    <h4>✅ Face Swap Successful!</h4>
                    <ul>
                        <li><strong>Source Faces:</strong> {info['source_faces']}</li>
                        <li><strong>Target Faces:</strong> {info['target_faces']}</li>
                        <li><strong>Swapped:</strong> {info['swapped_faces']} faces</li>
                        <li><strong>Processing Time:</strong> {info['processing_time']}</li>
                        <li><strong>Blend Mode:</strong> {info['parameters']['blend_mode']}</li>
                        {f"<li><strong>Face Restoration:</strong> {info['parameters']['restore_model']}</li>" if info['parameters']['face_restore'] else ""}
                    </ul>
                </div>
                """
                return result_img, info_html
            else:
                error_msg = info.get("error", "Unknown error")
                return None, f"<p style='color: red;'>Face swap failed: {error_msg}</p>"
                
        except Exception as e:
            import traceback
            return None, f"""
            <div style='padding: 10px; background: #ffebee; border-radius: 5px;'>
                <h4>❌ Error</h4>
                <p>{str(e)}</p>
                <details>
                    <summary>Traceback</summary>
                    <pre>{traceback.format_exc()}</pre>
                </details>
            </div>
            """
            
    swap_image_btn.click(
        swap_face_image,
        inputs=[
            source_image, target_image,
            source_face_index, target_face_index, similarity_threshold, blend_mode,
            face_restore, face_restore_model, face_restore_fidelity,
            background_enhance, face_upsample, upscale_factor,
            preserve_expression, expression_weight,
            preserve_lighting, lighting_weight,
            preserve_age, age_weight
        ],
        outputs=[swapped_image, swap_info]
    )
    
    # Video face swap function
    def swap_face_video(
        source_img, target_vid,
        source_idx, target_idx,
        temporal_smooth, smooth_window,
        restore, quality
    ):
        """Perform face swap on video"""
        try:
            if not source_img or not target_vid:
                return None, "", "<p style='color: red;'>Please upload both source image and target video</p>"
                
            # Initialize models if needed
            if not app.face_swap_manager.models_loaded:
                yield None, "<p>Initializing models...</p>", ""
                success, msg = app.face_swap_manager.initialize_models()
                if not success:
                    yield None, "", f"<p style='color: red;'>Failed to initialize models: {msg}</p>"
                    return
                    
            # Create parameters based on quality
            if quality == "fast":
                params = FaceSwapParams(
                    source_face_index=int(source_idx),
                    target_face_index=int(target_idx),
                    face_restore=False,
                    temporal_smoothing=temporal_smooth,
                    smoothing_window=int(smooth_window)
                )
            elif quality == "quality":
                params = FaceSwapParams(
                    source_face_index=int(source_idx),
                    target_face_index=int(target_idx),
                    face_restore=restore,
                    face_restore_model=FaceRestoreModel.CODEFORMER,
                    face_restore_fidelity=0.7,
                    face_upsample=True,
                    temporal_smoothing=temporal_smooth,
                    smoothing_window=int(smooth_window)
                )
            else:  # balanced
                params = FaceSwapParams(
                    source_face_index=int(source_idx),
                    target_face_index=int(target_idx),
                    face_restore=restore,
                    face_restore_model=FaceRestoreModel.CODEFORMER,
                    face_restore_fidelity=0.5,
                    temporal_smoothing=temporal_smooth,
                    smoothing_window=int(smooth_window)
                )
                
            # Output path
            import uuid
            output_path = Path(app.output_dir) / f"swapped_video_{uuid.uuid4()}.mp4"
            
            # Progress callback
            def progress_callback(progress, message):
                progress_html = f"""
                <div style='padding: 10px; background: #e3f2fd; border-radius: 5px;'>
                    <h4>Processing Video...</h4>
                    <div style='width: 100%; background: #ddd; border-radius: 3px; overflow: hidden;'>
                        <div style='width: {progress*100:.1f}%; background: #2196f3; height: 20px;'></div>
                    </div>
                    <p>{message}</p>
                </div>
                """
                return progress_html
                
            # Process video
            yield None, progress_callback(0, "Starting video processing..."), ""
            
            success, info = app.face_swap_manager.process_video(
                source_image=source_img,
                target_video=target_vid,
                output_path=output_path,
                params=params,
                progress_callback=lambda p, m: None  # Can't yield in callback
            )
            
            if success:
                info_html = f"""
                <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                    <h4>✅ Video Processing Complete!</h4>
                    <ul>
                        <li><strong>Total Frames:</strong> {info['total_frames']}</li>
                        <li><strong>Processed:</strong> {info['processed_frames']} frames</li>
                        <li><strong>FPS:</strong> {info['fps']}</li>
                        <li><strong>Resolution:</strong> {info['resolution']}</li>
                    </ul>
                </div>
                """
                yield str(output_path), "", info_html
            else:
                error_msg = info.get("error", "Unknown error")
                yield None, "", f"<p style='color: red;'>Video processing failed: {error_msg}</p>"
                
        except Exception as e:
            yield None, "", f"<p style='color: red;'>Error: {str(e)}</p>"
            
    swap_video_btn.click(
        swap_face_video,
        inputs=[
            video_source_image, target_video,
            video_source_face_index, video_target_face_index,
            temporal_smoothing, smoothing_window,
            video_face_restore, video_quality
        ],
        outputs=[swapped_video, video_progress, video_info]
    )
    
    # Batch processing function
    def process_batch(source_img, target_imgs, same_settings, restore, quality):
        """Process batch of images"""
        try:
            if not source_img or not target_imgs:
                return "", [], gr.update()
                
            # Initialize models if needed
            if not app.face_swap_manager.models_loaded:
                success, msg = app.face_swap_manager.initialize_models()
                if not success:
                    return f"<p style='color: red;'>Failed to initialize models: {msg}</p>", [], gr.update()
                    
            # Create parameters based on quality
            if quality == "fast":
                params = FaceSwapParams(face_restore=False)
            elif quality == "quality":
                params = FaceSwapParams(
                    face_restore=restore,
                    face_restore_model=FaceRestoreModel.CODEFORMER,
                    face_upsample=True
                )
            else:  # balanced
                params = FaceSwapParams(
                    face_restore=restore,
                    face_restore_model=FaceRestoreModel.CODEFORMER
                )
                
            # Output directory
            import uuid
            batch_id = str(uuid.uuid4())[:8]
            output_dir = Path(app.output_dir) / f"batch_{batch_id}"
            
            # Progress callback
            def progress_callback(progress, message):
                return f"""
                <div style='padding: 10px; background: #e3f2fd; border-radius: 5px;'>
                    <h4>Batch Processing...</h4>
                    <div style='width: 100%; background: #ddd; border-radius: 3px; overflow: hidden;'>
                        <div style='width: {progress*100:.1f}%; background: #2196f3; height: 20px;'></div>
                    </div>
                    <p>{message}</p>
                </div>
                """
                
            # Get target image paths
            target_paths = []
            for img in target_imgs:
                if hasattr(img, 'name'):
                    target_paths.append(img.name)
                else:
                    target_paths.append(img)
                    
            # Process batch
            results = app.face_swap_manager.batch_process(
                source_images=[source_img] * len(target_paths),
                target_images=target_paths,
                output_dir=output_dir,
                params=params,
                progress_callback=lambda p, m: None
            )
            
            # Collect successful results
            success_count = sum(1 for success, _ in results if success)
            result_images = [path for success, path in results if success]
            
            if success_count > 0:
                # Create zip file
                import zipfile
                zip_path = Path(app.output_dir) / f"batch_results_{batch_id}.zip"
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    for img_path in result_images:
                        zf.write(img_path, Path(img_path).name)
                        
                progress_html = f"""
                <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                    <h4>✅ Batch Processing Complete!</h4>
                    <p>Successfully processed {success_count}/{len(target_paths)} images</p>
                </div>
                """
                
                return progress_html, result_images, gr.update(value=str(zip_path), visible=True)
            else:
                return "<p style='color: red;'>All batch processing failed</p>", [], gr.update()
                
        except Exception as e:
            return f"<p style='color: red;'>Error: {str(e)}</p>", [], gr.update()
            
    batch_process_btn.click(
        process_batch,
        inputs=[
            batch_source_image, batch_target_images,
            batch_params_same, batch_face_restore, batch_quality
        ],
        outputs=[batch_progress, batch_results, batch_download]
    )