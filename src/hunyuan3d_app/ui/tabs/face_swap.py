"""Face swap tab for enhanced UI"""

import gradio as gr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...features.face_swap import FaceRestoreModel, BlendMode, FaceSwapParams
from ...features.face_swap.facefusion_adapter import FaceFusionModel


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
                            label="Source Face Index",
                            info="Which face to use from source image"
                        )
                        gr.Markdown("*💡 Check Face Detection Preview to see face indices*")
                        
                        target_face_index = gr.Number(
                            value=-1,
                            label="Target Face Index",
                            info="Which face to replace (-1 = all faces)"
                        )
                        gr.Markdown("*💡 Use specific index for better control*")
                        
                        similarity_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                            label="Face Similarity Threshold"
                        )
                        gr.Markdown("*Only swap if faces are similar (0=swap all, 1=identical only)*")
                        
                        blend_mode = gr.Radio(
                            choices=["seamless", "hard", "soft", "poisson"],
                            value="seamless",
                            label="Blend Mode"
                        )
                    
                    # FaceFusion 2025 Options
                    with gr.Accordion("🚀 FaceFusion 3.2.0 Settings", open=True):
                        use_facefusion = gr.Checkbox(
                            label="Enable FaceFusion 3.2.0",
                            value=True,
                            visible=False  # Always enabled, hide the checkbox
                        )
                        
                        facefusion_model = gr.Dropdown(
                            choices=[
                                "inswapper_128",
                                "hyperswap_1a_256", 
                                "hyperswap_1b_256",
                                "hyperswap_1c_256",
                                "ghost_1_256",
                                "ghost_2_256", 
                                "ghost_3_256",
                                "simswap_256",
                                "simswap_unofficial_512",
                                "blendswap_256",
                                "uniface_256",
                                "hififace_unofficial_256"
                            ],
                            value="inswapper_128",
                            label="FaceFusion Model",
                            info="Choose face swapping model (higher numbers = better quality)",
                            visible=True
                        )
                        
                        pixel_boost = gr.Dropdown(
                            choices=["128x128", "256x256", "512x512"],
                            value="256x256",
                            label="Pixel Boost (2025)",
                            info="Enhanced resolution processing for sharper results",
                            visible=True
                        )
                        
                        live_portrait = gr.Checkbox(
                            label="Live Portrait Mode (2025)",
                            value=False,
                            info="Advanced expression preservation technology",
                            visible=True
                        )
                        
                        face_detector_score = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Face Detection Confidence",
                            info="Minimum confidence for face detection",
                            visible=True
                        )
                        
                        gr.Markdown("*🎯 FaceFusion 3.2.0 provides superior quality and accuracy*")
                        
                        # FaceFusion is always enabled - no visibility toggle needed
                        
                    # Legacy options - set defaults but hide from user
                    face_restore = gr.Checkbox(value=False, visible=False)
                    face_restore_model = gr.Dropdown(choices=["CodeFormer"], value="CodeFormer", visible=False)
                    face_restore_fidelity = gr.Slider(value=0.5, visible=False)
                    background_enhance = gr.Checkbox(value=False, visible=False)
                    face_upsample = gr.Checkbox(value=False, visible=False)
                    upscale_factor = gr.Slider(value=1, visible=False)
                        
                    # Advanced options - hidden, handled by FaceFusion
                    preserve_expression = gr.Checkbox(value=False, visible=False)
                    expression_weight = gr.Slider(value=0.3, visible=False)
                    preserve_lighting = gr.Checkbox(value=True, visible=False)
                    lighting_weight = gr.Slider(value=0.5, visible=False)
                    preserve_age = gr.Checkbox(value=True, visible=False)
                    age_weight = gr.Slider(value=0.7, visible=False)
                        
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
                        with gr.Row():
                            detect_source_btn = gr.Button("🔍 Detect Source Faces")
                            detect_target_btn = gr.Button("🔍 Detect Target Faces")
                        
                        with gr.Row():
                            source_preview = gr.Image(
                                label="Source Face Preview",
                                type="pil",
                                interactive=False
                            )
                            target_preview = gr.Image(
                                label="Target Face Preview", 
                                type="pil",
                                interactive=False
                            )
                        
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
    
    # Face detection preview with visualization
    def detect_and_visualize_faces(image, image_type):
        """Detect faces and return visualization with numbered boxes"""
        try:
            if not image:
                return None, "<p>Please upload an image first</p>"
            
            # Initialize models if needed
            if not app.face_swap_manager.facefusion_loaded:
                success, msg = app.face_swap_manager.initialize_models()
                if not success:
                    return None, f"""
                    <div style='padding: 10px; background: #ffebee; border-radius: 5px;'>
                        <h4>Face Detection Models Not Found</h4>
                        <p>Run <code>python download_face_swap_models.py</code> to download the required models.</p>
                    </div>
                    """
                
            faces = app.face_swap_manager.detect_faces(image)
            
            if not faces:
                return image, f"<p>No faces detected in {image_type} image</p>"
            
            # Create visualization
            import numpy as np
            from PIL import ImageDraw, ImageFont
            
            # Create a copy to draw on
            vis_img = image.copy()
            draw = ImageDraw.Draw(vis_img)
            
            # Colors for different confidence levels
            def get_color(score):
                if score >= 0.8:
                    return 'green'
                elif score >= 0.6:
                    return 'yellow'
                elif score >= 0.5:
                    return 'orange'
                else:
                    return 'red'
            
            # Sort faces by confidence for display
            faces_with_idx = [(i, face) for i, face in enumerate(faces)]
            faces_with_idx.sort(key=lambda x: x[1].det_score, reverse=True)
            
            info_html = f"<h4>Detected {len(faces)} face(s) in {image_type} image:</h4>"
            info_html += "<table style='width: 100%; border-collapse: collapse;'>"
            info_html += "<tr><th>Index</th><th>Confidence</th><th>Size</th><th>Quality</th></tr>"
            
            for idx, face in faces_with_idx:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                # Determine face quality
                aspect_ratio = width / height if height > 0 else 0
                if width < 30 or height < 30:
                    quality = "Too Small"
                    quality_color = "#ff6b6b"
                elif aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    quality = "Poor Aspect"
                    quality_color = "#ffa94d"
                elif face.det_score < 0.5:
                    quality = "Low Confidence"
                    quality_color = "#ffa94d"
                else:
                    quality = "Good"
                    quality_color = "#51cf66"
                
                # Get color based on confidence
                color = get_color(face.det_score)
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw index number
                text = f"{idx}"
                # Try to use a font
                try:
                    font = ImageFont.truetype("arial.ttf", max(20, min(40, height//5)))
                except:
                    font = None
                
                # Draw text background
                text_bbox = draw.textbbox((x1+5, y1+5), text, font=font) if font else [x1+5, y1+5, x1+35, y1+35]
                draw.rectangle(text_bbox, fill=color)
                draw.text((x1+5, y1+5), text, fill='white', font=font)
                
                # Add to info table
                info_html += f"""
                <tr>
                    <td style='text-align: center; font-weight: bold;'>{idx}</td>
                    <td style='text-align: center; color: {color};'>{face.det_score:.1%}</td>
                    <td style='text-align: center;'>{width}×{height}</td>
                    <td style='text-align: center; color: {quality_color};'>{quality}</td>
                </tr>
                """
            
            info_html += "</table>"
            info_html += "<p><small>💡 Use the index number to select which face to swap</small></p>"
            
            return vis_img, info_html
            
        except Exception as e:
            import traceback
            return image, f"<p style='color: red;'>Error detecting faces: {str(e)}</p><pre>{traceback.format_exc()}</pre>"
            
    detect_source_btn.click(
        lambda img: detect_and_visualize_faces(img, "source"),
        inputs=[source_image],
        outputs=[source_preview, detection_info]
    )
    
    detect_target_btn.click(
        lambda img: detect_and_visualize_faces(img, "target"),
        inputs=[target_image],
        outputs=[target_preview, detection_info]
    )
    
    # Main face swap function
    def swap_face_image(
        source_img, target_img, 
        source_idx, target_idx, similarity, blend,
        use_ff, ff_model, pixel_boost_val, live_portrait_val, detector_score,
        restore, restore_model, fidelity, bg_enhance, upsample, upscale,
        preserve_expr, expr_weight, preserve_light, light_weight,
        preserve_age_val, age_weight_val
    ):
        """Perform face swap on images"""
        try:
            if not source_img or not target_img:
                return None, "<p style='color: red;'>Please upload both source and target images</p>"
                
            # Initialize models if needed
            if not app.face_swap_manager.facefusion_loaded:
                success, msg = app.face_swap_manager.initialize_models()
                if not success:
                    error_html = f"""
                    <div style='padding: 10px; background: #ffebee; border-radius: 5px;'>
                        <h4>🚀 FaceFusion 3.2.0 Initialization Failed</h4>
                        <p><strong>Error:</strong> {msg}</p>
                        <h5>🔧 Troubleshooting:</h5>
                        <ul>
                            <li>Ensure FaceFusion is installed at: <code>models/facefusion/</code></li>
                            <li>Check that required dependencies are installed:</li>
                            <ul>
                                <li><code>pip install onnx onnxruntime</code></li>
                                <li><code>pip install opencv-python psutil</code></li>
                            </ul>
                            <li>FaceFusion models will download automatically on first use</li>
                        </ul>
                        <p><strong>Note:</strong> This app uses FaceFusion 3.2.0 for state-of-the-art face swapping.</p>
                    </div>
                    """
                    return None, error_html
                    
            # Create parameters
            params = FaceSwapParams(
                source_face_index=int(source_idx),
                target_face_index=int(target_idx),
                similarity_threshold=similarity,
                blend_mode=BlendMode(blend),
                
                # FaceFusion 2025 options
                use_facefusion=use_ff,
                facefusion_model=FaceFusionModel(ff_model) if use_ff else FaceFusionModel.INSWAPPER_128,
                pixel_boost=pixel_boost_val,
                live_portrait=live_portrait_val,
                face_detector_score=detector_score,
                
                # Legacy options
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
            
            # Debug logging
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Face swap UI received: result_img={type(result_img)}, info={type(info)}")
            if info:
                logger.info(f"Info contents: {info}")
            
            if result_img is not None:
                try:
                    method = info.get('method', 'Unknown')
                    is_facefusion = 'FaceFusion' in method
                    
                    method_emoji = "🚀" if is_facefusion else "🔧"
                    method_bg = "#e3f2fd" if is_facefusion else "#e8f5e9"
                    
                    info_html = f"""
                    <div style='padding: 10px; background: {method_bg}; border-radius: 5px;'>
                        <h4>{method_emoji} Face Swap Successful! ({method})</h4>
                        <ul>"""
                    
                    if is_facefusion:
                        info_html += f"""
                            <li><strong>Model:</strong> {info.get('model', 'N/A')}</li>
                            <li><strong>Pixel Boost:</strong> {info.get('pixel_boost', 'N/A')}</li>
                            <li><strong>Live Portrait:</strong> {'Yes' if info.get('live_portrait', False) else 'No'}</li>
                            <li><strong>Detector:</strong> {info.get('detector', 'N/A')}</li>
                            <li><strong>Processing Time:</strong> {info.get('processing_time', 0):.2f}s</li>
                        """
                    else:
                        info_html += f"""
                            <li><strong>Source Faces:</strong> {info.get('source_faces', 'N/A')}</li>
                            <li><strong>Target Faces:</strong> {info.get('target_faces', 'N/A')}</li>
                            <li><strong>Swapped:</strong> {info.get('swapped_faces', 'N/A')} faces</li>
                            <li><strong>Processing Time:</strong> {info.get('processing_time', 0):.2f}s</li>
                            <li><strong>Blend Mode:</strong> {info.get('parameters', {}).get('blend_mode', 'N/A')}</li>
                            {f"<li><strong>Face Restoration:</strong> Enabled</li>" if info.get('parameters', {}).get('face_restore', False) else ""}
                        """
                    
                    info_html += """
                        </ul>
                    </div>
                    """
                    return result_img, info_html
                except Exception as e:
                    logger.error(f"Error formatting success info: {e}")
                    # Still return the image even if info formatting fails
                    return result_img, "<div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'><h4>Face Swap Successful!</h4></div>"
            else:
                if info is None:
                    error_msg = "No information returned from face swap"
                elif isinstance(info, dict):
                    error_msg = info.get("error", "Unknown error occurred")
                else:
                    error_msg = str(info)
                    
                return None, f"<p style='color: red;'>Face swap failed: {error_msg}</p>"
                
        except Exception as e:
            import traceback
            return None, f"""
            <div style='padding: 10px; background: #ffebee; border-radius: 5px;'>
                <h4>Error</h4>
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
            use_facefusion, facefusion_model, pixel_boost, live_portrait, face_detector_score,
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
            if not app.face_swap_manager.facefusion_loaded:
                yield None, "<p>Initializing models...</p>", ""
                success, msg = app.face_swap_manager.initialize_models()
                if not success:
                    error_html = f"""
                    <div style='padding: 10px; background: #ffebee; border-radius: 5px;'>
                        <h4>🚀 FaceFusion 3.2.0 Not Ready</h4>
                        <p>{msg}</p>
                        <p>Run <code>python download_face_swap_models.py</code> to download the required models.</p>
                    </div>
                    """
                    yield None, "", error_html
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
                    <h4>Video Processing Complete!</h4>
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
            if not app.face_swap_manager.facefusion_loaded:
                success, msg = app.face_swap_manager.initialize_models()
                if not success:
                    error_html = f"""
                    <div style='padding: 10px; background: #ffebee; border-radius: 5px;'>
                        <h4>🚀 FaceFusion 3.2.0 Not Ready</h4>
                        <p>{msg}</p>
                        <p>Run <code>python download_face_swap_models.py</code> to download the required models.</p>
                    </div>
                    """
                    return error_html, [], gr.update()
                    
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
                    <h4>Batch Processing Complete!</h4>
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