"""Video generation tab for enhanced UI"""

import gradio as gr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...config import VIDEO_MODELS
from ...generation.video import VideoModel, VideoGenerationParams


def create_video_generation_tab(app: Any) -> None:
    """Create the video generation tab
    
    Args:
        app: The enhanced application instance
    """
    gr.Markdown("""
    ### 🎬 AI Video Generation
    Create videos from text prompts using state-of-the-art models.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Prompt inputs
            prompt = gr.Textbox(
                label="Video Prompt",
                placeholder="A majestic eagle soaring through mountain clouds...",
                lines=3
            )
            
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="blurry, low quality, distorted",
                lines=2
            )
            
            # Model selection
            with gr.Row():
                video_model = gr.Dropdown(
                    choices=list(VIDEO_MODELS.keys()),
                    value="ltxvideo",
                    label="Video Model",
                    interactive=True
                )
                
                quality_preset = gr.Radio(
                    choices=["fast", "balanced", "quality"],
                    value="balanced",
                    label="Quality Preset"
                )
                
            # Video parameters
            with gr.Accordion("Video Settings", open=True):
                with gr.Row():
                    duration = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=5.0,
                        step=0.5,
                        label="Duration (seconds)"
                    )
                    
                    fps = gr.Slider(
                        minimum=8,
                        maximum=60,
                        value=24,
                        step=1,
                        label="FPS"
                    )
                    
                with gr.Row():
                    width = gr.Slider(
                        minimum=256,
                        maximum=1920,
                        value=768,
                        step=64,
                        label="Width"
                    )
                    
                    height = gr.Slider(
                        minimum=256,
                        maximum=1080,
                        value=512,
                        step=64,
                        label="Height"
                    )
                    
                with gr.Row():
                    motion_strength = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Motion Strength"
                    )
                    
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale"
                    )
                    
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=5,
                        label="Inference Steps"
                    )
                    
                    seed = gr.Number(
                        value=-1,
                        label="Seed (-1 for random)"
                    )
                    
            # Character consistency
            with gr.Accordion("Character Consistency (Optional)", open=False):
                use_character = gr.Checkbox(
                    label="Use Character Profile",
                    value=False
                )
                
                character_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Character",
                    visible=False
                )
                
                consistency_strength = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.8,
                    step=0.1,
                    label="Consistency Strength",
                    visible=False
                )
                
            # Generation controls
            with gr.Row():
                generate_btn = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    size="lg"
                )
                
                stop_btn = gr.Button(
                    "⏹️ Stop",
                    variant="stop",
                    size="lg"
                )
                
                estimate_btn = gr.Button(
                    "⏱️ Estimate Time",
                    variant="secondary"
                )
                
        with gr.Column(scale=2):
            # Output displays
            output_video = gr.Video(
                label="Generated Video",
                autoplay=True
            )
            
            preview_gif = gr.Image(
                label="Preview GIF",
                type="filepath"
            )
            
            generation_info = gr.HTML()
            
            # Model info display
            model_info_display = gr.Markdown()
            
    # Update model info when selection changes
    def update_model_info(model_key):
        if model_key in VIDEO_MODELS:
            model = VIDEO_MODELS[model_key]
            info = f"""
            #### {model['name']}
            - **Size**: {model['size']}
            - **VRAM**: {model['vram_required']}
            - **Max Duration**: {model['max_duration']}s
            - **Optimal FPS**: {model['optimal_fps']}
            - **Optimal Resolution**: {model['optimal_resolution'][0]}x{model['optimal_resolution'][1]}
            
            {model['description']}
            
            **Capabilities**: {', '.join(model['capabilities'])}
            """
            
            # Update parameter limits
            return (
                info,
                gr.update(maximum=model['max_duration']),  # duration
                gr.update(value=model['optimal_fps']),      # fps
                gr.update(value=model['optimal_resolution'][0]),  # width
                gr.update(value=model['optimal_resolution'][1])   # height
            )
        return "", gr.update(), gr.update(), gr.update(), gr.update()
        
    video_model.change(
        update_model_info,
        inputs=[video_model],
        outputs=[model_info_display, duration, fps, width, height]
    )
    
    # Handle character consistency toggle
    def toggle_character_options(use_char):
        if use_char:
            # Get available characters
            characters = app.character_consistency_manager.list_characters()
            char_choices = [(f"{c.name} ({c.id[:8]}...)", c.id) for c in characters]
            
            return (
                gr.update(visible=True, choices=char_choices),
                gr.update(visible=True)
            )
        return (
            gr.update(visible=False),
            gr.update(visible=False)
        )
        
    use_character.change(
        toggle_character_options,
        inputs=[use_character],
        outputs=[character_dropdown, consistency_strength]
    )
    
    # Estimate generation time
    def estimate_time(model_key, duration, fps, width, height, steps):
        try:
            from ...generation.video import VideoModel
            
            model_enum = {
                "ltxvideo": VideoModel.LTXVIDEO,
                "wan21": VideoModel.WAN_2_1,
                "skyreels": VideoModel.SKYREELS
            }.get(model_key)
            
            if not model_enum:
                return "Unknown model"
                
            params = VideoGenerationParams(
                prompt="",
                duration_seconds=duration,
                fps=fps,
                width=width,
                height=height,
                num_inference_steps=steps
            )
            
            estimated = app.video_generator.estimate_generation_time(model_enum, params)
            
            return f"""
            <div style='padding: 10px; background: #f0f0f0; border-radius: 5px;'>
                <h4>⏱️ Estimated Generation Time</h4>
                <p><strong>{estimated:.1f} seconds</strong> ({estimated/60:.1f} minutes)</p>
                <p style='font-size: 0.9em; color: #666;'>
                    This is an estimate based on model benchmarks and may vary based on GPU load.
                </p>
            </div>
            """
        except Exception as e:
            return f"Error estimating time: {str(e)}"
            
    estimate_btn.click(
        estimate_time,
        inputs=[video_model, duration, fps, width, height, num_inference_steps],
        outputs=[generation_info]
    )
    
    # Main generation function
    def generate_video(
        prompt, negative_prompt, model_key, quality_preset,
        duration, fps, width, height, motion_strength,
        guidance_scale, steps, seed, use_char, char_id, consistency
    ):
        try:
            # Map model key to enum
            model_enum = {
                "ltxvideo": VideoModel.LTXVIDEO,
                "wan21": VideoModel.WAN_2_1,
                "skyreels": VideoModel.SKYREELS
            }.get(model_key)
            
            if not model_enum:
                yield None, None, "<p style='color: red;'>Invalid model selected</p>"
                return
                
            # Create parameters
            params = VideoGenerationParams(
                prompt=prompt,
                negative_prompt=negative_prompt,
                duration_seconds=duration,
                fps=fps,
                width=width,
                height=height,
                motion_strength=motion_strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                seed=seed
            )
            
            # Add character embeddings if selected
            if use_char and char_id:
                character = app.character_consistency_manager.get_character(char_id)
                if character and character.full_embeddings is not None:
                    params.character_embeddings = character.full_embeddings
                    params.consistency_strength = consistency
                    
            # Adjust parameters based on quality preset
            if quality_preset == "fast":
                params.num_inference_steps = max(10, steps // 2)
                params.width = min(width, 512)
                params.height = min(height, 384)
            elif quality_preset == "quality":
                params.num_inference_steps = int(steps * 1.5)
                
            # Submit to queue
            import uuid
            job_id = str(uuid.uuid4())
            
            # Create progress callback
            progress_callback = app.progress_manager.create_progress_callback(
                job_id, "video_generation"
            )
            
            # Load model if needed
            yield None, None, "<p>Loading video model...</p>"
            
            success, msg = app.video_generator.load_model(
                model_enum,
                progress_callback=progress_callback
            )
            
            if not success:
                yield None, None, f"<p style='color: red;'>Failed to load model: {msg}</p>"
                return
                
            # Generate video
            yield None, None, "<p>Generating video frames...</p>"
            
            frames, info = app.video_generator.generate_video(
                params,
                progress_callback=progress_callback
            )
            
            if not frames:
                error_msg = info.get("error", "Unknown error")
                yield None, None, f"<p style='color: red;'>Generation failed: {error_msg}</p>"
                return
                
            # Save video
            yield None, None, "<p>Encoding video...</p>"
            
            from pathlib import Path
            output_path = Path(app.output_dir) / f"video_{job_id}.mp4"
            gif_path = Path(app.output_dir) / f"preview_{job_id}.gif"
            
            # Save as MP4
            success = app.video_generator.save_video(
                frames, output_path, fps=params.fps
            )
            
            if not success:
                yield None, None, "<p style='color: red;'>Failed to save video</p>"
                return
                
            # Create preview GIF
            app.video_generator.create_preview_gif(
                frames, gif_path, fps=min(10, params.fps)
            )
            
            # Format info
            info_html = f"""
            <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                <h4>✅ Video Generated Successfully!</h4>
                <ul>
                    <li><strong>Model:</strong> {info['model']}</li>
                    <li><strong>Duration:</strong> {info['duration']}s</li>
                    <li><strong>Resolution:</strong> {info['resolution']}</li>
                    <li><strong>FPS:</strong> {info['fps']}</li>
                    <li><strong>Frames:</strong> {info['frames']}</li>
                    <li><strong>Generation Time:</strong> {info['generation_time']}</li>
                    <li><strong>Seed:</strong> {info['seed']}</li>
                </ul>
            </div>
            """
            
            # Mark job complete
            app.progress_manager.complete_task(job_id, {"output": str(output_path)})
            
            yield str(output_path), str(gif_path), info_html
            
        except Exception as e:
            import traceback
            error_msg = f"""
            <div style='padding: 10px; background: #ffebee; border-radius: 5px;'>
                <h4>❌ Error</h4>
                <p>{str(e)}</p>
                <details>
                    <summary>Full traceback</summary>
                    <pre>{traceback.format_exc()}</pre>
                </details>
            </div>
            """
            yield None, None, error_msg
            
    # Wire up generation
    generate_btn.click(
        generate_video,
        inputs=[
            prompt, negative_prompt, video_model, quality_preset,
            duration, fps, width, height, motion_strength,
            guidance_scale, num_inference_steps, seed,
            use_character, character_dropdown, consistency_strength
        ],
        outputs=[output_video, preview_gif, generation_info]
    )
    
    # Initialize model info
    video_model.change(
        update_model_info,
        inputs=[video_model],
        outputs=[model_info_display, duration, fps, width, height]
    )