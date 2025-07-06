"""Consolidated Settings Tab"""

import gradio as gr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from ..components.common import create_action_button

logger = logging.getLogger(__name__)


def create_settings_tab(app: Any) -> None:
    """Create the consolidated settings tab with all configuration options"""
    
    gr.Markdown("## ⚙️ Settings & Configuration")
    gr.Markdown("Manage models, credentials, performance settings, and system configuration")
    
    with gr.Tabs() as settings_tabs:
        # 1. Model Management (from Downloads Manager)
        with gr.Tab("Model Management", elem_id="model_management_tab"):
            create_model_management_subtab(app)
        
        # 2. API Credentials
        with gr.Tab("API Credentials", elem_id="credentials_tab"):
            create_credentials_subtab(app)
        
        # 3. Performance Settings
        with gr.Tab("Performance", elem_id="performance_tab"):
            create_performance_subtab(app)
        
        # 4. Queue Management
        with gr.Tab("Queue Management", elem_id="queue_tab"):
            create_queue_management_subtab(app)
        
        # 5. System Info & Testing
        with gr.Tab("System Info", elem_id="system_tab"):
            create_system_info_subtab(app)
        
        # 6. Preferences
        with gr.Tab("Preferences", elem_id="preferences_tab"):
            create_preferences_subtab(app)


def create_model_management_subtab(app: Any) -> None:
    """Model download and management interface"""
    
    with gr.Tabs() as model_tabs:
        # Download new models
        with gr.Tab("Download Models"):
            gr.Markdown("### Download AI Models")
            
            # Model search and filters
            with gr.Row():
                model_search = gr.Textbox(
                    label="Search Models",
                    placeholder="Search by name or type..."
                )
                
                model_type_filter = gr.Dropdown(
                    choices=["All", "Image", "3D", "Video", "LoRA"],
                    value="All",
                    label="Model Type"
                )
                
                search_btn = create_action_button("🔍 Search", size="sm")
            
            # Available models list
            available_models = gr.DataFrame(
                headers=["Name", "Type", "Size", "VRAM", "Description", "Status"],
                datatype=["str", "str", "str", "str", "str", "str"],
                col_count=6
            )
            
            # Download controls
            with gr.Row():
                selected_model = gr.Textbox(
                    label="Selected Model",
                    interactive=False
                )
                
                download_btn = create_action_button("📥 Download", variant="primary")
                cancel_btn = create_action_button("❌ Cancel", variant="stop", size="sm")
            
            # Download progress
            download_progress = gr.Progress()
            download_status = gr.HTML()
            
            # Load available models
            def load_available_models(search_query="", type_filter="All"):
                """Load list of available models"""
                try:
                    from ...config import ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, VIDEO_MODELS
                    
                    models_data = []
                    
                    # Add image models
                    if type_filter in ["All", "Image"]:
                        for key, config in ALL_IMAGE_MODELS.items():
                            if search_query.lower() in config.name.lower():
                                try:
                                    downloaded_models = app.model_manager.get_downloaded_models("image")
                                    status = "Downloaded" if str(key) in downloaded_models else "Available"
                                except:
                                    status = "Available"
                                models_data.append([
                                    config.name,
                                    "Image",
                                    config.size,
                                    config.vram_required,
                                    config.description,
                                    status
                                ])
                    
                    # Add 3D models
                    if type_filter in ["All", "3D"]:
                        for key, name in HUNYUAN3D_MODELS.items():
                            if search_query.lower() in name.lower():
                                try:
                                    downloaded_models = app.model_manager.get_downloaded_models("3d")
                                    status = "Downloaded" if str(key) in downloaded_models else "Available"
                                except:
                                    status = "Available"
                                models_data.append([
                                    name,
                                    "3D",
                                    "~7-15GB",
                                    "16GB+",
                                    "Hunyuan3D model for 3D generation",
                                    status
                                ])
                    
                    # Add video models
                    if type_filter in ["All", "Video"]:
                        for key, config in VIDEO_MODELS.items():
                            if search_query.lower() in config["name"].lower():
                                models_data.append([
                                    config["name"],
                                    "Video",
                                    config["size"],
                                    config["vram_required"],
                                    config["description"],
                                    "Available"
                                ])
                    
                    return models_data
                    
                except Exception as e:
                    logger.error(f"Error loading models: {e}")
                    return []
            
            # Wire up search
            search_btn.click(
                load_available_models,
                inputs=[model_search, model_type_filter],
                outputs=[available_models]
            )
            
            # Don't load models synchronously - it blocks the UI
            # available_models.value = load_available_models()
        
        # Manage downloaded models
        with gr.Tab("Installed Models"):
            gr.Markdown("### Manage Installed Models")
            
            # Model type tabs
            with gr.Tabs():
                # Image models
                with gr.Tab("Image Models"):
                    image_models_list = gr.DataFrame(
                        headers=["Name", "Path", "Size", "Last Used"],
                        datatype=["str", "str", "str", "str"],
                    )
                    
                    with gr.Row():
                        refresh_image_btn = create_action_button("🔄 Refresh", size="sm")
                        delete_image_btn = create_action_button("🗑️ Delete Selected", variant="stop", size="sm")
                
                # 3D models
                with gr.Tab("3D Models"):
                    threed_models_list = gr.DataFrame(
                        headers=["Name", "Path", "Size", "Last Used"],
                        datatype=["str", "str", "str", "str"],
                    )
                    
                    with gr.Row():
                        refresh_3d_btn = create_action_button("🔄 Refresh", size="sm")
                        delete_3d_btn = create_action_button("🗑️ Delete Selected", variant="stop", size="sm")
                
                # LoRA models
                with gr.Tab("LoRA Models"):
                    lora_models_list = gr.DataFrame(
                        headers=["Name", "Base Model", "Trigger Words", "Size"],
                        datatype=["str", "str", "str", "str"],
                    )
                    
                    with gr.Row():
                        refresh_lora_btn = create_action_button("🔄 Refresh", size="sm")
                        delete_lora_btn = create_action_button("🗑️ Delete Selected", variant="stop", size="sm")
        
        # Model conversion tools
        with gr.Tab("Model Tools"):
            gr.Markdown("### Model Conversion & Optimization")
            
            # GGUF conversion
            with gr.Group():
                gr.Markdown("#### Convert to GGUF Format")
                
                source_model = gr.Dropdown(
                    choices=[],
                    label="Source Model"
                )
                
                quantization = gr.Radio(
                    choices=["Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
                    value="Q8_0",
                    label="Quantization Level"
                )
                
                convert_btn = create_action_button("🔄 Convert to GGUF", variant="primary")
                conversion_status = gr.HTML()


def create_credentials_subtab(app: Any) -> None:
    """API credentials management"""
    
    if hasattr(app, 'credential_manager'):
        app.credential_manager.create_ui_component()
    else:
        gr.Markdown("### API Credentials")
        
        # Manual credential inputs
        with gr.Group():
            gr.Markdown("#### Hugging Face")
            hf_token = gr.Textbox(
                label="HF Token",
                type="password",
                placeholder="hf_..."
            )
            
        with gr.Group():
            gr.Markdown("#### Civitai")
            civitai_token = gr.Textbox(
                label="API Key",
                type="password",
                placeholder="Your Civitai API key"
            )
        
        save_creds_btn = create_action_button("💾 Save Credentials", variant="primary")
        creds_status = gr.HTML()


def create_performance_subtab(app: Any) -> None:
    """Performance and optimization settings"""
    
    gr.Markdown("### Performance Settings")
    
    # Memory optimization
    with gr.Group():
        gr.Markdown("#### Memory Optimization")
        
        enable_xformers = gr.Checkbox(
            label="Enable xFormers (memory efficient attention)",
            value=True,
            info="Reduces VRAM usage with minimal quality impact"
        )
        
        enable_cpu_offload = gr.Checkbox(
            label="Enable CPU Offload",
            value=False,
            info="Move model parts to CPU when not in use (slower but saves VRAM)"
        )
        
        enable_sequential_cpu_offload = gr.Checkbox(
            label="Enable Sequential CPU Offload",
            value=False,
            info="More aggressive CPU offloading (very slow but minimal VRAM)"
        )
        
        attention_slicing = gr.Dropdown(
            choices=["Disabled", "Auto", "Max"],
            value="Auto",
            label="Attention Slicing",
            info="Process attention in chunks to save memory"
        )
    
    # Generation settings
    with gr.Group():
        gr.Markdown("#### Generation Settings")
        
        with gr.Row():
            default_batch_size = gr.Slider(
                1, 8, 1,
                step=1,
                label="Default Batch Size"
            )
            
            vae_tiling = gr.Checkbox(
                label="Enable VAE Tiling",
                value=False,
                info="Process large images in tiles"
            )
        
        with gr.Row():
            torch_compile = gr.Checkbox(
                label="Enable Torch Compile",
                value=False,
                info="Compile models for faster inference (initial slowdown)"
            )
            
            channels_last = gr.Checkbox(
                label="Channels Last Memory Format",
                value=True,
                info="Optimize memory layout for better performance"
            )
    
    # Cache settings
    with gr.Group():
        gr.Markdown("#### Cache Settings")
        
        with gr.Row():
            model_cache_size = gr.Slider(
                1, 10, 3,
                step=1,
                label="Max Models in Memory",
                info="Number of models to keep loaded"
            )
            
            clear_cache_btn = create_action_button("🧹 Clear Cache", size="sm")
        
        cache_info = gr.HTML()
    
    # Apply settings
    apply_perf_btn = create_action_button("💾 Apply Settings", variant="primary")
    perf_status = gr.HTML()


def create_queue_management_subtab(app: Any) -> None:
    """Queue and job management"""
    
    gr.Markdown("### Queue Management")
    
    # Queue status
    with gr.Row():
        queue_status = gr.HTML()
        refresh_queue_btn = create_action_button("🔄 Refresh", size="sm")
    
    # Active jobs
    with gr.Group():
        gr.Markdown("#### Active Jobs")
        
        active_jobs = gr.DataFrame(
            headers=["ID", "Type", "Status", "Progress", "Started", "Actions"],
            datatype=["str", "str", "str", "number", "str", "str"]
        )
        
        with gr.Row():
            pause_queue_btn = create_action_button("⏸️ Pause Queue", size="sm")
            resume_queue_btn = create_action_button("▶️ Resume Queue", size="sm")
            clear_completed_btn = create_action_button("🧹 Clear Completed", size="sm")
    
    # Queue settings
    with gr.Group():
        gr.Markdown("#### Queue Settings")
        
        with gr.Row():
            max_workers = gr.Slider(
                1, 8, 2,
                step=1,
                label="Max Concurrent Jobs",
                info="Number of jobs to process simultaneously"
            )
            
            job_timeout = gr.Slider(
                60, 1800, 600,
                step=60,
                label="Job Timeout (seconds)",
                info="Maximum time for a single job"
            )
        
        auto_cleanup = gr.Checkbox(
            label="Auto-cleanup completed jobs",
            value=True,
            info="Remove completed jobs after 24 hours"
        )
    
    # Job history
    with gr.Group():
        gr.Markdown("#### Job History")
        
        job_history = gr.DataFrame(
            headers=["ID", "Type", "Status", "Duration", "Completed", "Size"],
            datatype=["str", "str", "str", "str", "str", "str"]
        )
        
        with gr.Row():
            export_history_btn = create_action_button("📤 Export History", size="sm")
            clear_history_btn = create_action_button("🗑️ Clear History", variant="stop", size="sm")
    
    # Wire up queue display
    if hasattr(app, 'queue_manager'):
        app.queue_manager.create_ui_component()


def create_system_info_subtab(app: Any) -> None:
    """System information and diagnostics"""
    
    gr.Markdown("### System Information")
    
    # System requirements check
    with gr.Group():
        gr.Markdown("#### System Requirements")
        
        system_check = gr.HTML(
            value=app.check_system_requirements()["html"] if hasattr(app, 'check_system_requirements') else ""
        )
        
        check_system_btn = create_action_button("🔍 Check System", variant="primary")
    
    # Performance metrics
    with gr.Group():
        gr.Markdown("#### Performance Metrics")
        
        perf_metrics = gr.HTML()
        
        with gr.Row():
            benchmark_btn = create_action_button("📊 Run Benchmark", variant="primary")
            export_metrics_btn = create_action_button("📤 Export Metrics", size="sm")
    
    # GPU information
    with gr.Group():
        gr.Markdown("#### GPU Information")
        
        gpu_info = gr.HTML()
        monitor_gpu = gr.Checkbox(
            label="Enable GPU Monitoring",
            value=True
        )
    
    # Diagnostics
    with gr.Group():
        gr.Markdown("#### Diagnostics")
        
        with gr.Row():
            test_image_gen = create_action_button("🖼️ Test Image Generation", size="sm")
            test_3d_gen = create_action_button("🎲 Test 3D Generation", size="sm")
            test_video_gen = create_action_button("🎬 Test Video Generation", size="sm")
        
        diagnostic_output = gr.Textbox(
            label="Diagnostic Output",
            lines=10,
            max_lines=20
        )
    
    # Logs
    with gr.Group():
        gr.Markdown("#### Application Logs")
        
        log_level = gr.Dropdown(
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            value="INFO",
            label="Log Level"
        )
        
        logs_output = gr.Textbox(
            label="Recent Logs",
            lines=15,
            max_lines=30
        )
        
        with gr.Row():
            refresh_logs_btn = create_action_button("🔄 Refresh Logs", size="sm")
            export_logs_btn = create_action_button("📤 Export Logs", size="sm")
            clear_logs_btn = create_action_button("🗑️ Clear Logs", size="sm")
    
    # Wire up system check
    if hasattr(app, 'check_system_requirements'):
        check_system_btn.click(
            lambda: app.check_system_requirements()["html"],
            outputs=[system_check]
        )


def create_preferences_subtab(app: Any) -> None:
    """User preferences and UI settings"""
    
    gr.Markdown("### Preferences")
    
    # UI preferences
    with gr.Group():
        gr.Markdown("#### User Interface")
        
        with gr.Row():
            ui_theme = gr.Dropdown(
                choices=["Light", "Dark", "Auto"],
                value="Auto",
                label="Theme"
            )
            
            ui_scale = gr.Slider(
                0.8, 1.2, 1.0,
                step=0.05,
                label="UI Scale"
            )
        
        show_tips = gr.Checkbox(
            label="Show helpful tips",
            value=True
        )
        
        auto_save = gr.Checkbox(
            label="Auto-save preferences",
            value=True
        )
    
    # Default generation settings
    with gr.Group():
        gr.Markdown("#### Default Generation Settings")
        
        with gr.Row():
            default_image_size = gr.Dropdown(
                choices=["512x512", "768x768", "1024x1024", "1536x1536"],
                value="1024x1024",
                label="Default Image Size"
            )
            
            default_steps = gr.Slider(
                10, 100, 30,
                step=5,
                label="Default Steps"
            )
        
        with gr.Row():
            default_guidance = gr.Slider(
                1, 20, 7.5,
                step=0.5,
                label="Default Guidance Scale"
            )
            
            default_sampler = gr.Dropdown(
                choices=["Euler", "Euler a", "DPM++ 2M", "DPM++ SDE"],
                value="Euler a",
                label="Default Sampler"
            )
    
    # File handling
    with gr.Group():
        gr.Markdown("#### File Handling")
        
        output_format = gr.Dropdown(
            choices=["PNG", "JPEG", "WEBP"],
            value="PNG",
            label="Default Image Format"
        )
        
        with gr.Row():
            auto_download = gr.Checkbox(
                label="Auto-download generated files",
                value=False
            )
            
            organize_by_date = gr.Checkbox(
                label="Organize outputs by date",
                value=True
            )
        
        output_directory = gr.Textbox(
            label="Output Directory",
            value=str(app.output_dir) if hasattr(app, 'output_dir') else "outputs"
        )
    
    # Privacy settings
    with gr.Group():
        gr.Markdown("#### Privacy & Data")
        
        analytics = gr.Checkbox(
            label="Share anonymous usage statistics",
            value=False
        )
        
        history_retention = gr.Slider(
            7, 365, 30,
            step=1,
            label="History retention (days)"
        )
        
        clear_on_exit = gr.Checkbox(
            label="Clear temporary files on exit",
            value=True
        )
    
    # Save preferences
    with gr.Row():
        save_prefs_btn = create_action_button("💾 Save Preferences", variant="primary")
        reset_prefs_btn = create_action_button("🔄 Reset to Defaults", size="sm")
        export_prefs_btn = create_action_button("📤 Export Settings", size="sm")
    
    prefs_status = gr.HTML()