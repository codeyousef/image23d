"""Model Management UI Component

Handles model downloads, installed model management, and texture components.
"""

import gradio as gr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import shutil

from ...components.common import create_action_button

logger = logging.getLogger(__name__)


def create_model_management_subtab(app: Any) -> None:
    """Model download and management interface"""
    
    # Store mapping of display names to model keys
    model_name_to_key = {}
    
    # Check if texture components are installed
    from .texture_components import check_texture_components_status
    realesrgan_installed, xatlas_installed = check_texture_components_status()
    
    # Show warning if texture components are missing
    if not realesrgan_installed or not xatlas_installed:
        gr.Markdown("""
        <div style="background: #fef3c7; border: 2px solid #f59e0b; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
            <h4 style="margin-top: 0; color: #92400e;">⚠️ Texture Components Missing!</h4>
            <p style="margin-bottom: 8px; color: #92400e;">Essential texture pipeline components are not installed. Your 3D models will generate without textures.</p>
            <p style="margin-bottom: 0; color: #92400e;"><strong>Quick Fix:</strong> Go to the <strong>"Texture Components"</strong> tab below to install them.</p>
        </div>
        """)
    
    with gr.Tabs() as model_tabs:
        # Download new models
        with gr.Tab("Download Models") as download_tab:
            create_download_models_tab(app, model_name_to_key)
        
        # Manage downloaded models
        with gr.Tab("Installed Models"):
            create_installed_models_tab(app)
        
        # Texture Components
        with gr.Tab("Texture Components"):
            from .texture_components import create_texture_components_tab
            create_texture_components_tab(app)
        
        # Model conversion tools
        with gr.Tab("Model Tools"):
            create_model_tools_tab(app)


def load_available_models(app: Any, search_query: str = "", type_filter: str = "All", model_name_to_key: Dict = None) -> List:
    """Load list of available models"""
    try:
        from ....config import ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, VIDEO_MODELS
        
        models_data = []
        
        # Add image models
        if type_filter in ["All", "Image"]:
            for key, config in ALL_IMAGE_MODELS.items():
                if search_query.lower() in config.name.lower():
                    try:
                        downloaded_models = app.model_manager.get_downloaded_models("image")
                        status = "✅ Downloaded" if str(key) in downloaded_models else "📥 Available"
                    except:
                        status = "📥 Available"
                    if model_name_to_key is not None:
                        model_name_to_key[config.name] = (key, "image")
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
            for key, config in HUNYUAN3D_MODELS.items():
                if search_query.lower() in config["name"].lower():
                    try:
                        downloaded_models = app.model_manager.get_downloaded_models("3d")
                        status = "✅ Downloaded" if str(key) in downloaded_models else "📥 Available"
                    except:
                        status = "📥 Available"
                    if model_name_to_key is not None:
                        model_name_to_key[config["name"]] = (key, "3d")
                    models_data.append([
                        config["name"],
                        "3D",
                        config["size"],
                        config["vram_required"],
                        config["description"],
                        status
                    ])
        
        # Add video models
        if type_filter in ["All", "Video"]:
            for key, config in VIDEO_MODELS.items():
                if search_query.lower() in config["name"].lower():
                    if model_name_to_key is not None:
                        model_name_to_key[config["name"]] = (key, "video")
                    models_data.append([
                        config["name"],
                        "Video",
                        config["size"],
                        config["vram_required"],
                        config["description"],
                        "📥 Available"
                    ])
        
        return models_data
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return []


def create_download_models_tab(app: Any, model_name_to_key: Dict) -> None:
    """Create the download models UI"""
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
        refresh_list_btn = create_action_button("🔄 Refresh List", size="sm")
    
    # Available models list
    available_models = gr.DataFrame(
        value=load_available_models(app, "", "All", model_name_to_key), 
        headers=["Name", "Type", "Size", "VRAM", "Description", "Status"],
        datatype=["str", "str", "str", "str", "str", "str"],
        col_count=6,
        interactive=False,
        elem_classes=["model-list-table"],
        wrap=True
    )
    
    # Download controls
    with gr.Row():
        selected_model = gr.Textbox(
            label="Selected Model",
            interactive=True,
            placeholder="Type or paste a model name here (e.g., FLUX.1-dev)"
        )
        
        download_btn = create_action_button("📥 Download", variant="primary")
        force_download_btn = create_action_button("🔄 Force Re-download", variant="secondary", size="sm")
        cancel_btn = create_action_button("❌ Cancel", variant="stop", size="sm")
    
    # Download progress
    download_progress = gr.Progress()
    download_status = gr.HTML()
    
    # Real-time download progress
    create_download_progress_ui(app)
    
    # Wire up controls
    wire_download_controls(
        app, model_name_to_key, available_models, model_search, model_type_filter,
        search_btn, refresh_list_btn, selected_model, download_btn, 
        force_download_btn, cancel_btn, download_status
    )


def create_download_progress_ui(app: Any) -> None:
    """Create download progress UI with WebSocket or polling fallback"""
    with gr.Group():
        gr.Markdown("#### Download Progress")
        current_download_info = gr.HTML(value="<p>No active downloads</p>")
        
        # Try to use WebSocket component
        try:
            from ...components.websocket_progress import create_websocket_progress
            
            ws_progress = create_websocket_progress(
                host="localhost",
                port=8765,
                task_filter="download"
            )
            current_download_info = ws_progress.create_component()
            
            websocket_status = gr.HTML(
                value="<p style='color: green; font-size: 0.9em;'>🟢 Real-time updates via WebSocket</p>"
            )
            
        except ImportError:
            logger.warning("WebSocket progress component not available, falling back to polling")
            
            # Use polling fallback
            with gr.Row():
                refresh_btn = create_action_button("🔄 Refresh Progress", size="sm")
                auto_refresh = gr.Checkbox(label="Auto-refresh (2s)", value=True, scale=1)
            
            timer = gr.Timer(value=2.0, active=True)
            
            def check_download_progress(app):
                """Check current download progress"""
                try:
                    progress = app.model_manager.get_download_progress()
                    
                    if progress.get("status") == "downloading":
                        return format_download_progress(progress)
                    elif progress.get("is_complete"):
                        return "<p style='color: green;'>✅ Download completed!</p>"
                    else:
                        return f"<p>No active downloads (status: {progress.get('status', 'unknown')})</p>"
                except Exception as e:
                    logger.error(f"Error checking progress: {e}", exc_info=True)
                    return f"<p>Error checking progress: {str(e)}</p>"
            
            def update_if_auto_refresh(auto_refresh_enabled):
                if auto_refresh_enabled:
                    return check_download_progress(app)
                return gr.update()
            
            timer.tick(
                fn=update_if_auto_refresh,
                inputs=[auto_refresh],
                outputs=[current_download_info]
            )
            
            refresh_btn.click(
                fn=lambda: check_download_progress(app),
                outputs=[current_download_info]
            )


def format_download_progress(progress: Dict) -> str:
    """Format download progress information as HTML"""
    model = progress.get("model", "Unknown")
    percentage = progress.get("percentage", 0)
    downloaded_gb = progress.get("downloaded_gb", 0)
    total_gb = progress.get("total_gb", 0)
    current_file = progress.get("current_file", "")
    speed_mbps = progress.get("speed_mbps", 0)
    eta_minutes = progress.get("eta_minutes", 0)
    
    # File progress
    completed_files = progress.get("completed_files", 0)
    total_files = progress.get("total_files", 0)
    files_percentage = progress.get("files_percentage", 0)
    
    # Current file progress
    current_file_percentage = progress.get("current_file_percentage", 0)
    current_file_size_mb = progress.get("current_file_size_mb", 0)
    current_file_downloaded_mb = progress.get("current_file_downloaded_mb", 0)
    
    html = f"""
    <div style='padding: 1rem; background: #f0f7ff; border-radius: 8px;'>
        <h4 style='margin: 0 0 0.5rem 0;'>📥 Downloading: {model}</h4>
        
        <!-- Overall Progress -->
        <div style='margin-bottom: 1rem;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.25rem;'>
                <span>Overall Progress</span>
                <span>{percentage:.1f}% ({downloaded_gb:.2f} / {total_gb:.2f} GB)</span>
            </div>
            <div style='background: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden;'>
                <div style='background: #2196F3; height: 100%; width: {percentage:.1f}%; transition: width 0.3s;'></div>
            </div>
        </div>
        
        <!-- File Progress -->
        <div style='margin-bottom: 1rem;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.25rem;'>
                <span>Files</span>
                <span>{completed_files} / {total_files} ({files_percentage:.1f}%)</span>
            </div>
            <div style='background: #e0e0e0; height: 10px; border-radius: 5px; overflow: hidden;'>
                <div style='background: #4CAF50; height: 100%; width: {files_percentage:.1f}%; transition: width 0.3s;'></div>
            </div>
        </div>
        
        <!-- Current File -->
        {f'''
        <div style='margin-bottom: 1rem; padding: 0.5rem; background: #e8f4ff; border-radius: 4px;'>
            <div style='font-size: 0.9em; margin-bottom: 0.25rem;'>
                Current: {current_file.split('/')[-1] if current_file else 'Starting...'}
            </div>
            <div style='display: flex; justify-content: space-between; font-size: 0.85em; color: #666;'>
                <span>{current_file_downloaded_mb:.1f} / {current_file_size_mb:.1f} MB</span>
                <span>{current_file_percentage:.1f}%</span>
            </div>
            <div style='background: #d0d0d0; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 0.25rem;'>
                <div style='background: #FF9800; height: 100%; width: {current_file_percentage:.1f}%; transition: width 0.3s;'></div>
            </div>
        </div>
        ''' if current_file else ''}
        
        <!-- Stats -->
        <div style='display: flex; justify-content: space-between; font-size: 0.9em; color: #666;'>
            <span>⚡ Speed: {speed_mbps:.1f} MB/s</span>
            <span>⏱️ ETA: {eta_minutes:.1f} min</span>
        </div>
    </div>
    """
    return html


def wire_download_controls(
    app: Any, 
    model_name_to_key: Dict,
    available_models: gr.DataFrame,
    model_search: gr.Textbox,
    model_type_filter: gr.Dropdown,
    search_btn: gr.Button,
    refresh_list_btn: gr.Button,
    selected_model: gr.Textbox,
    download_btn: gr.Button,
    force_download_btn: gr.Button,
    cancel_btn: gr.Button,
    download_status: gr.HTML
) -> None:
    """Wire up all download-related controls"""
    
    # Search functionality
    def search_models(search_query, type_filter):
        return load_available_models(app, search_query, type_filter, model_name_to_key)
    
    search_btn.click(
        search_models,
        inputs=[model_search, model_type_filter],
        outputs=[available_models]
    )
    
    model_type_filter.change(
        search_models,
        inputs=[model_search, model_type_filter],
        outputs=[available_models]
    )
    
    refresh_list_btn.click(
        search_models,
        inputs=[model_search, model_type_filter],
        outputs=[available_models]
    )
    
    # Model selection
    def select_model(evt: gr.SelectData, dataframe_data):
        """Handle model selection from the DataFrame"""
        try:
            if evt.index is not None:
                if isinstance(evt.index, list) and len(evt.index) >= 1:
                    row_idx = evt.index[0]
                elif isinstance(evt.index, tuple):
                    row_idx = evt.index[0]
                else:
                    row_idx = evt.index
                
                if isinstance(dataframe_data, list) and row_idx < len(dataframe_data):
                    model_name = dataframe_data[row_idx][0]
                    logger.info(f"Selected model: {model_name}")
                    return model_name
                elif hasattr(dataframe_data, 'iloc') and not dataframe_data.empty and row_idx < len(dataframe_data):
                    model_name = dataframe_data.iloc[row_idx, 0]
                    logger.info(f"Selected model: {model_name}")
                    return model_name
        except Exception as e:
            logger.error(f"Error selecting model: {e}")
        return ""
    
    available_models.select(
        fn=select_model,
        inputs=[available_models],
        outputs=[selected_model],
        queue=False
    )
    
    # Download functionality
    def download_selected_model(model_name, force=False, progress=gr.Progress()):
        """Download the selected model"""
        if not model_name:
            return "<div style='color: red;'>Please select a model first.</div>"
        
        try:
            from ....config import ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, VIDEO_MODELS
            
            # Get model key and type from mapping
            if model_name not in model_name_to_key:
                return f"<div style='color: red;'>Model '{model_name}' not found. Please select from the list.</div>"
            
            model_key, model_type = model_name_to_key[model_name]
            
            # Get model config
            if model_type == "image":
                model_config = ALL_IMAGE_MODELS.get(model_key)
            elif model_type == "3d":
                model_config = HUNYUAN3D_MODELS.get(model_key)
            elif model_type == "video":
                model_config = VIDEO_MODELS.get(model_key)
            else:
                return f"<div style='color: red;'>Invalid model type: {model_type}</div>"
            
            if not model_config:
                return f"<div style='color: red;'>Model configuration not found for {model_name}</div>"
            
            # Start download
            progress(0, desc=f"Starting download of {model_name}...")
            logger.info(f"Attempting to download: {model_name} (key: {model_key}, type: {model_type})")
            
            # If force download, delete existing incomplete model first
            if force:
                if model_type == "image":
                    model_path = app.model_manager.models_dir / "image" / model_key
                elif model_type == "3d":
                    model_path = app.model_manager.models_dir / "3d" / model_key
                else:
                    model_path = app.model_manager.models_dir / model_type / model_key
                    
                if model_path.exists():
                    logger.info(f"Force download: Removing existing model at {model_path}")
                    shutil.rmtree(model_path)
            
            # Use the model manager to download
            success, message = app.model_manager.download_model(
                model_name=model_key,
                model_type=model_type,
                progress_callback=None
            )
            
            logger.info(f"Download result: success={success}, message={message}")
            
            if success:
                return f"<div style='color: blue;'>⏳ Download started for {model_name}. Check progress below...</div>"
            else:
                return f"<div style='color: red;'>❌ {message}</div>"
                
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return f"<div style='color: red;'>❌ Error downloading model: {str(e)}</div>"
    
    # Wire up download buttons
    download_btn.click(
        fn=lambda model: download_selected_model(model, force=False),
        inputs=[selected_model],
        outputs=[download_status]
    ).then(
        fn=lambda: load_available_models(app, "", "All", model_name_to_key),
        outputs=[available_models]
    )
    
    force_download_btn.click(
        fn=lambda model: download_selected_model(model, force=True),
        inputs=[selected_model],
        outputs=[download_status]
    ).then(
        fn=lambda: load_available_models(app, "", "All", model_name_to_key),
        outputs=[available_models]
    )
    
    # Cancel download
    def cancel_download():
        """Cancel the current download"""
        try:
            message = app.model_manager.stop_download()
            return f"<div style='color: orange;'>⚠️ {message}</div>"
        except Exception as e:
            return f"<div style='color: red;'>❌ Error cancelling download: {str(e)}</div>"
    
    cancel_btn.click(
        fn=cancel_download,
        outputs=[download_status]
    )


def create_installed_models_tab(app: Any) -> None:
    """Create UI for managing installed models"""
    gr.Markdown("### Manage Installed Models")
    
    with gr.Tabs():
        # Image models
        with gr.Tab("Image Models"):
            create_model_type_tab(app, "image")
        
        # 3D models
        with gr.Tab("3D Models"):
            create_model_type_tab(app, "3d")
        
        # LoRA models
        with gr.Tab("LoRA Models"):
            lora_models_list = gr.DataFrame(
                headers=["Name", "Base Model", "Trigger Words", "Size"],
                datatype=["str", "str", "str", "str"],
            )
            
            with gr.Row():
                refresh_lora_btn = create_action_button("🔄 Refresh", size="sm")
                delete_lora_btn = create_action_button("🗑️ Delete Selected", variant="stop", size="sm")


def create_model_type_tab(app: Any, model_type: str) -> None:
    """Create tab for a specific model type"""
    def load_models():
        """Load installed models of this type"""
        try:
            downloaded = app.model_manager.get_downloaded_models(model_type)
            models_data = []
            
            for model_name in downloaded:
                model_path = app.model_manager.models_dir / model_type / model_name
                if not model_path.exists() and model_type == "image":
                    model_path = app.model_manager.models_dir / "gguf" / model_name
                
                # Get size
                size = 0
                if model_path.exists():
                    for file in model_path.rglob("*"):
                        if file.is_file():
                            size += file.stat().st_size
                    size_str = f"{size / (1024**3):.1f} GB" if size > 0 else "Unknown"
                else:
                    size_str = "Unknown"
                
                models_data.append([
                    model_name,
                    str(model_path),
                    size_str,
                    "Never"  # TODO: Track last used
                ])
            
            return models_data
        except Exception as e:
            logger.error(f"Error loading {model_type} models: {e}")
            return []
    
    models_list = gr.DataFrame(
        value=load_models(),
        headers=["Name", "Path", "Size", "Last Used"],
        datatype=["str", "str", "str", "str"],
    )
    
    with gr.Row():
        refresh_btn = create_action_button("🔄 Refresh", size="sm")
        delete_btn = create_action_button("🗑️ Delete Selected", variant="stop", size="sm")
    
    refresh_btn.click(
        load_models,
        outputs=[models_list]
    )


def create_model_tools_tab(app: Any) -> None:
    """Create model conversion and optimization tools"""
    gr.Markdown("### Model Conversion & Optimization")
    
    # GGUF conversion
    with gr.Group():
        gr.Markdown("#### Convert to GGUF Format")
        
        def get_convertible_models():
            """Get list of downloaded models that can be converted to GGUF"""
            try:
                choices = []
                # Get downloaded image models
                image_models = app.model_manager.get_downloaded_models("image")
                for model in image_models:
                    if not model.lower().endswith(('gguf', 'q4', 'q5', 'q6', 'q8')):
                        choices.append(f"Image: {model}")
                
                # Get downloaded 3D models
                threed_models = app.model_manager.get_downloaded_models("3d")
                for model in threed_models:
                    choices.append(f"3D: {model}")
                
                return choices if choices else ["No convertible models found"]
            except Exception as e:
                logger.error(f"Error getting convertible models: {e}")
                return ["Error loading models"]
        
        with gr.Row():
            source_model = gr.Dropdown(
                choices=get_convertible_models(),
                label="Source Model",
                info="Select a model to convert to GGUF format",
                scale=4
            )
            refresh_models_btn = create_action_button("🔄", size="sm")
        
        refresh_models_btn.click(
            get_convertible_models,
            outputs=[source_model]
        )
        
        quantization = gr.Radio(
            choices=["Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
            value="Q8_0",
            label="Quantization Level",
            info="Higher = Better quality, larger size | Lower = Smaller size, slight quality loss"
        )
        
        gr.Markdown("""
        **Quantization Levels:**
        - **Q8_0**: Best quality (98% of original), ~50% size reduction
        - **Q6_K**: Great balance (96% quality), ~60% size reduction
        - **Q5_K_M**: Good quality (95% quality), ~65% size reduction
        - **Q4_K_S**: Memory efficient (90% quality), ~70% size reduction
        """)
        
        convert_btn = create_action_button("🔄 Convert to GGUF", variant="primary")
        conversion_status = gr.HTML()