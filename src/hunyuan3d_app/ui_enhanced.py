"""Enhanced UI with all new features integrated"""

import gradio as gr
from pathlib import Path

from .ui import create_interface as create_base_interface
from .ui_modern import ModernUI, load_modern_css
from .config import ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, QUALITY_PRESETS


def create_enhanced_interface(app):
    """Create enhanced interface with all new features"""
    
    # Load custom CSS
    custom_css = load_modern_css()
    
    with gr.Blocks(
        title="Hunyuan3D Studio Enhanced",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate"
        ),
        css=custom_css
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎨 Hunyuan3D Studio Enhanced</h1>
            <p>Professional Text → Image → 3D Pipeline with Advanced Features</p>
        </div>
        """)
        
        # Dashboard stats
        with gr.Row():
            stats = app.get_system_stats()
            with gr.Column(scale=1):
                ModernUI.create_stat_card(
                    str(stats.get('total_generations', 0)),
                    "Total Generations",
                    "🎨"
                )
            with gr.Column(scale=1):
                ModernUI.create_stat_card(
                    str(stats.get('models_loaded', 0)),
                    "Models Loaded",
                    "🤖"
                )
            with gr.Column(scale=1):
                ModernUI.create_stat_card(
                    f"{stats.get('vram_used', 0):.1f}GB",
                    "VRAM Used",
                    "💾"
                )
            with gr.Column(scale=1):
                ModernUI.create_stat_card(
                    str(stats.get('queue_pending', 0)),
                    "Queue Pending",
                    "⏳"
                )
        
        # Main tabs
        with gr.Tabs() as main_tabs:
            # Quick Generate Tab (existing functionality)
            with gr.Tab("🚀 Quick Generate"):
                create_quick_generate_tab(app)
                
            # Advanced Pipeline Tab
            with gr.Tab("🔧 Advanced Pipeline"):
                create_advanced_pipeline_tab(app)
                
            # Model Hub Tab
            with gr.Tab("🌐 Model Hub"):
                create_model_hub_tab(app)
                
            # Queue & History Tab
            with gr.Tab("📋 Queue & History"):
                create_queue_history_tab(app)
                
            # Model Comparison Tab
            with gr.Tab("📊 Benchmarks"):
                app.model_comparison.create_ui_component()
                
            # Settings Tab
            with gr.Tab("⚙️ Settings"):
                create_settings_tab(app)
                
        # Auto-refresh for stats
        def refresh_stats():
            stats = app.get_system_stats()
            # Would need to return updated HTML for stat cards
            pass
            
    return interface


def create_quick_generate_tab(app):
    """Create the quick generate tab with existing functionality"""
    gr.Markdown("### Complete Pipeline: Text → Image → 3D")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Text prompt
            prompt = gr.Textbox(
                label="What would you like to create?",
                placeholder="A majestic golden crown with intricate details...",
                lines=3
            )
            negative_prompt = gr.Textbox(
                label="What to avoid",
                value="blurry, low quality, multiple objects",
                lines=2
            )
            
            # Model selection
            with gr.Row():
                image_model = gr.Dropdown(
                    choices=[m for m in ALL_IMAGE_MODELS.keys()],
                    value=list(ALL_IMAGE_MODELS.keys())[0],
                    label="Image Model"
                )
                hunyuan_model = gr.Dropdown(
                    choices=[m for m in HUNYUAN3D_MODELS.keys()],
                    value=list(HUNYUAN3D_MODELS.keys())[0],
                    label="3D Model"
                )
                
            # Generation settings
            with gr.Accordion("Generation Settings", open=False):
                with gr.Row():
                    width = gr.Slider(512, 2048, 1024, step=64, label="Width")
                    height = gr.Slider(512, 2048, 1024, step=64, label="Height")
                with gr.Row():
                    steps = gr.Slider(10, 100, 30, step=5, label="Steps")
                    guidance = gr.Slider(1, 20, 7.5, step=0.5, label="Guidance")
                    seed = gr.Number(-1, label="Seed (-1 for random)")
                    
            # Generate button
            generate_btn = gr.Button("🎨 Generate 3D Asset", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            # Results
            output_image = gr.Image(label="Generated Image", type="pil")
            output_3d = gr.Model3D(label="3D Model")
            output_info = gr.HTML()
            
    # Wire up the generation
    def generate_full_pipeline(prompt, negative_prompt, image_model, hunyuan_model, 
                             width, height, steps, guidance, seed):
        # Submit job to queue
        job_id = app.submit_generation_job(
            job_type="full_pipeline",
            params={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model_name": image_model,
                "hunyuan3d_model_name": hunyuan_model,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance,
                "seed": seed
            }
        )
        
        return f"Job submitted with ID: {job_id[:8]}... Check Queue tab for progress."
        
    generate_btn.click(
        generate_full_pipeline,
        inputs=[prompt, negative_prompt, image_model, hunyuan_model,
                width, height, steps, guidance, seed],
        outputs=[output_info]
    )


def create_advanced_pipeline_tab(app):
    """Create advanced pipeline with LoRA support"""
    gr.Markdown("### Advanced Generation with LoRA & Custom Settings")
    
    with gr.Row():
        with gr.Column():
            # Model and LoRA selection
            with gr.Group():
                gr.Markdown("#### Model Configuration")
                
                base_model = gr.Dropdown(
                    choices=[m for m in ALL_IMAGE_MODELS.keys()],
                    label="Base Model"
                )
                
                # LoRA selection
                available_loras = app.get_available_loras()
                lora_choices = [(l["name"], l["name"]) for l in available_loras]
                
                with gr.Group():
                    gr.Markdown("#### LoRA Configuration")
                    lora_1 = gr.Dropdown(choices=lora_choices, label="LoRA 1", value=None)
                    lora_1_weight = gr.Slider(0, 2, 1, step=0.1, label="LoRA 1 Weight")
                    
                    lora_2 = gr.Dropdown(choices=lora_choices, label="LoRA 2", value=None)
                    lora_2_weight = gr.Slider(0, 2, 1, step=0.1, label="LoRA 2 Weight")
                    
            # Prompt
            prompt = gr.Textbox(label="Prompt", lines=3)
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=2)
            
        with gr.Column():
            # Output
            output = gr.Image(label="Result", type="pil")
            info = gr.HTML()
            
    # Generation button
    generate_btn = gr.Button("Generate with LoRAs", variant="primary")
    
    def generate_with_loras(base_model, lora_1, lora_1_weight, lora_2, lora_2_weight,
                           prompt, negative_prompt):
        # Build LoRA configs
        lora_configs = []
        if lora_1:
            lora_info = next((l for l in available_loras if l["name"] == lora_1), None)
            if lora_info:
                # Need to create LoRAInfo object
                from .lora_manager import LoRAInfo
                lora_obj = LoRAInfo(
                    name=lora_info["name"],
                    path=Path(lora_info["path"]),
                    base_model=lora_info["base_model"],
                    trigger_words=lora_info["trigger_words"]
                )
                lora_configs.append((lora_obj, lora_1_weight))
                
        if lora_2:
            lora_info = next((l for l in available_loras if l["name"] == lora_2), None)
            if lora_info:
                from .lora_manager import LoRAInfo
                lora_obj = LoRAInfo(
                    name=lora_info["name"],
                    path=Path(lora_info["path"]),
                    base_model=lora_info["base_model"],
                    trigger_words=lora_info["trigger_words"]
                )
                lora_configs.append((lora_obj, lora_2_weight))
                
        # Submit job
        job_id = app.submit_generation_job(
            job_type="image",
            params={
                "model_name": base_model,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "lora_configs": lora_configs,
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "guidance_scale": 7.5,
                "seed": -1
            }
        )
        
        return None, f"Job submitted: {job_id[:8]}..."
        
    generate_btn.click(
        generate_with_loras,
        inputs=[base_model, lora_1, lora_1_weight, lora_2, lora_2_weight,
                prompt, negative_prompt],
        outputs=[output, info]
    )


def create_model_hub_tab(app):
    """Create model hub with Civitai integration"""
    with gr.Tabs():
        # Civitai search
        with gr.Tab("🔍 Search Civitai"):
            with gr.Row():
                search_query = gr.Textbox(
                    label="Search Query",
                    placeholder="anime, realistic, fantasy..."
                )
                model_type = gr.Dropdown(
                    choices=["LORA", "Checkpoint", "TextualInversion"],
                    value="LORA",
                    label="Model Type"
                )
                base_model = gr.Dropdown(
                    choices=["FLUX.1", "SDXL", "SD 1.5"],
                    value="FLUX.1",
                    label="Base Model"
                )
                search_btn = gr.Button("Search", variant="primary")
                
            # Results
            search_results = gr.HTML()
            
            def search_civitai(query, model_type, base_model):
                results = app.search_civitai_models(query, model_type, base_model)
                
                html = "<div class='bento-container bento-grid-2'>"
                for model in results[:10]:
                    html += f"""
                    <div class='bento-card'>
                        <h4>{model['name']}</h4>
                        <p>{model['description'][:100]}...</p>
                        <div style='margin-top: 0.5rem;'>
                            <span>📥 {model['downloads']:,}</span>
                            <span style='margin-left: 1rem;'>❤️ {model['likes']:,}</span>
                        </div>
                        <button class='quick-action-btn' onclick='downloadModel({model['id']})'>
                            Download
                        </button>
                    </div>
                    """
                html += "</div>"
                
                return html
                
            search_btn.click(
                search_civitai,
                inputs=[search_query, model_type, base_model],
                outputs=[search_results]
            )
            
        # Downloaded models
        with gr.Tab("📦 Downloaded Models"):
            gr.Markdown("### Your Downloaded Models")
            
            # LoRA list
            lora_gallery = gr.Gallery(
                label="LoRAs",
                columns=4,
                height="auto"
            )
            
            def load_lora_gallery():
                loras = app.get_available_loras()
                # Would need to create thumbnails/previews
                return []
                
            # Refresh on tab load
            lora_gallery.load(load_lora_gallery, outputs=[lora_gallery])


def create_queue_history_tab(app):
    """Create queue and history management tab"""
    with gr.Tabs():
        # Queue management
        with gr.Tab("⏳ Queue"):
            app.queue_manager.create_ui_component()
            
        # Generation history
        with gr.Tab("📚 History"):
            app.history_manager.create_ui_component()


def create_settings_tab(app):
    """Create settings tab with credential management"""
    with gr.Tabs():
        # API Credentials
        with gr.Tab("🔐 API Credentials"):
            app.credential_manager.create_ui_component()
            
        # Performance Settings
        with gr.Tab("⚡ Performance"):
            gr.Markdown("### Performance Settings")
            
            with gr.Group():
                gr.Markdown("#### Memory Optimization")
                
                enable_xformers = gr.Checkbox(
                    label="Enable xFormers (memory efficient attention)",
                    value=True
                )
                
                enable_cpu_offload = gr.Checkbox(
                    label="Enable CPU Offload (for low VRAM)",
                    value=False
                )
                
                batch_size = gr.Slider(
                    1, 8, 1, step=1,
                    label="Batch Size"
                )
                
            with gr.Group():
                gr.Markdown("#### Queue Settings")
                
                max_workers = gr.Slider(
                    1, 4, 2, step=1,
                    label="Max Concurrent Jobs"
                )
                
                auto_cleanup_days = gr.Slider(
                    7, 90, 30, step=1,
                    label="Auto-cleanup after (days)"
                )
                
            save_settings_btn = gr.Button("💾 Save Settings", variant="primary")
            
        # System Info
        with gr.Tab("💻 System Info"):
            system_info = gr.HTML(app.check_system_requirements())
            
            refresh_btn = gr.Button("🔄 Refresh System Info")
            refresh_btn.click(
                app.check_system_requirements,
                outputs=[system_info]
            )