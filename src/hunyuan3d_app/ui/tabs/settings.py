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
    
    # Store mapping of display names to model keys
    model_name_to_key = {}
    
    # Define load_available_models function BEFORE using it
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
                            status = "✅ Downloaded" if str(key) in downloaded_models else "📥 Available"
                        except:
                            status = "📥 Available"
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
    
    # Check if texture components are installed
    from pathlib import Path
    realesrgan_installed = (Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth").exists()
    try:
        import xatlas
        xatlas_installed = True
    except ImportError:
        xatlas_installed = False
    
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
            
            # Available models list - Now properly loads on creation
            available_models = gr.DataFrame(
                value=load_available_models("", "All"), 
                headers=["Name", "Type", "Size", "VRAM", "Description", "Status"],
                datatype=["str", "str", "str", "str", "str", "str"],
                col_count=6,
                interactive=False,  # Set to False for row selection
                elem_classes=["model-list-table"],
                wrap=True
            )
            
            # Download controls
            with gr.Row():
                selected_model = gr.Textbox(
                    label="Selected Model",
                    interactive=True,  # Make it editable as a workaround
                    placeholder="Type or paste a model name here (e.g., FLUX.1-dev)"
                )
                
                download_btn = create_action_button("📥 Download", variant="primary")
                force_download_btn = create_action_button("🔄 Force Re-download", variant="secondary", size="sm")
                cancel_btn = create_action_button("❌ Cancel", variant="stop", size="sm")
            
            # Download progress
            download_progress = gr.Progress()
            download_status = gr.HTML()
            
            # Real-time download progress with WebSocket
            with gr.Group():
                gr.Markdown("#### Download Progress")
                current_download_info = gr.HTML(value="<p>No active downloads</p>")
                
                # Import WebSocket component
                try:
                    from ..components.websocket_progress import create_websocket_progress
                    
                    # Create WebSocket progress component
                    ws_progress = create_websocket_progress(
                        host="localhost",
                        port=8765,
                        task_filter="download"
                    )
                    ws_progress_component = ws_progress.create_component()
                    
                    # Replace the HTML component with WebSocket component
                    current_download_info = ws_progress_component
                    
                except ImportError:
                    logger.warning("WebSocket progress component not available, falling back to polling")
                
                # Keep the check_download_progress function for fallback
                def check_download_progress():
                    """Check current download progress (fallback for non-WebSocket)"""
                    try:
                        progress = app.model_manager.get_download_progress()
                        # Only log when there's actual download activity
                        if progress.get('status') == 'downloading' and progress.get('percentage', 0) > 0:
                            logger.debug(f"Download progress - {progress.get('model', 'Unknown')}: {progress.get('percentage', 0):.1f}%")
                        elif progress.get('status') not in ['idle', None]:
                            logger.debug(f"Download status: {progress.get('status')}")
                        
                        if progress.get("status") == "downloading":
                            model = progress.get("model", "Unknown")
                            
                            # Get detailed progress info
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
                            # Format the display
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
                        elif progress.get("is_complete"):
                            return "<p style='color: green;'>✅ Download completed!</p>"
                        else:
                            return f"<p>No active downloads (status: {progress.get('status', 'unknown')})</p>"
                    except Exception as e:
                        logger.error(f"Error checking progress: {e}", exc_info=True)
                        return f"<p>Error checking progress: {str(e)}</p>"
                
                # Only add polling controls if WebSocket is not available
                try:
                    # If WebSocket is available, just add a status indicator
                    websocket_status = gr.HTML(
                        value="<p style='color: green; font-size: 0.9em;'>🟢 Real-time updates via WebSocket</p>"
                    )
                except NameError:
                    # WebSocket not available, use polling
                    with gr.Row():
                        refresh_btn = create_action_button("🔄 Refresh Progress", size="sm")
                        auto_refresh = gr.Checkbox(label="Auto-refresh (2s)", value=True, scale=1)
                    
                    # Create a timer that updates the progress
                    timer = gr.Timer(value=2.0, active=True)
                    
                    # Update progress when timer fires (only if auto-refresh is on)
                    def update_if_auto_refresh(auto_refresh_enabled):
                        if auto_refresh_enabled:
                            return check_download_progress()
                        return gr.update()
                    
                    timer.tick(
                        fn=update_if_auto_refresh,
                        inputs=[auto_refresh],
                        outputs=[current_download_info]
                    )
                    
                    refresh_btn.click(
                        fn=check_download_progress,
                        outputs=[current_download_info]
                    )
                
            # Cancel download handler
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
            
            # Wire up search
            search_btn.click(
                load_available_models,
                inputs=[model_search, model_type_filter],
                outputs=[available_models]
            )
            
            # Also trigger search when type filter changes
            model_type_filter.change(
                load_available_models,
                inputs=[model_search, model_type_filter],
                outputs=[available_models]
            )
            
            # Wire up refresh button
            refresh_list_btn.click(
                load_available_models,
                inputs=[model_search, model_type_filter],
                outputs=[available_models]
            )
            
            # Add instruction text
            gr.Markdown("**Select a model to download:** Click on any row in the table to select a model")
            
            # Handle model selection (keeping this for when it works)
            def select_model(evt: gr.SelectData, dataframe_data):
                """Handle model selection from the DataFrame"""
                try:
                    # Debug logging
                    logger.info(f"Selection event - index: {evt.index}, value: {evt.value}")
                    
                    if evt.index is not None:
                        # For Gradio 5.0, evt.index is a list [row, col]
                        if isinstance(evt.index, list) and len(evt.index) >= 1:
                            row_idx = evt.index[0]
                        elif isinstance(evt.index, tuple):
                            row_idx = evt.index[0]
                        else:
                            row_idx = evt.index
                        
                        # Get the model name from the selected row
                        # Check if dataframe_data is not empty using proper pandas method
                        if isinstance(dataframe_data, list) and row_idx < len(dataframe_data):
                            model_name = dataframe_data[row_idx][0]
                            logger.info(f"Selected model: {model_name}")
                            return model_name
                        elif hasattr(dataframe_data, 'iloc') and not dataframe_data.empty and row_idx < len(dataframe_data):
                            # Handle pandas DataFrame
                            model_name = dataframe_data.iloc[row_idx, 0]
                            logger.info(f"Selected model: {model_name}")
                            return model_name
                except Exception as e:
                    logger.error(f"Error selecting model: {e}")
                return ""
            
            # Try using the select event
            available_models.select(
                fn=select_model,
                inputs=[available_models],
                outputs=[selected_model],
                queue=False
            )
            
            # Handle model download
            def download_selected_model(model_name, force=False, progress=gr.Progress()):
                """Download the selected model"""
                if not model_name:
                    return "<div style='color: red;'>Please select a model first.</div>"
                
                try:
                    from ...config import ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, VIDEO_MODELS
                    
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
                    
                    # Debug logging
                    logger.info(f"Attempting to download: {model_name} (key: {model_key}, type: {model_type})")
                    
                    # If force download, delete existing incomplete model first
                    if force:
                        import shutil
                        if model_type == "image":
                            model_path = app.model_manager.models_dir / "image" / model_key
                        elif model_type == "3d":
                            model_path = app.model_manager.models_dir / "3d" / model_key
                        else:
                            model_path = app.model_manager.models_dir / model_type / model_key
                            
                        if model_path.exists():
                            logger.info(f"Force download: Removing existing model at {model_path}")
                            shutil.rmtree(model_path)
                    
                    # Use the model manager to download (starts in background)
                    success, message = app.model_manager.download_model(
                        model_name=model_key,
                        model_type=model_type,
                        progress_callback=None  # Progress is tracked via get_download_progress instead
                    )
                    
                    logger.info(f"Download result: success={success}, message={message}")
                    
                    # The download happens in background, so show initial status
                    if success:
                        return f"<div style='color: blue;'>⏳ Download started for {model_name}. Check progress below...</div>"
                    else:
                        return f"<div style='color: red;'>❌ {message}</div>"
                        
                except Exception as e:
                    logger.error(f"Error downloading model: {e}")
                    return f"<div style='color: red;'>❌ Error downloading model: {str(e)}</div>"
            
            # Wire up download button
            download_btn.click(
                fn=lambda model: download_selected_model(model, force=False),
                inputs=[selected_model],
                outputs=[download_status]
            ).then(
                # Immediately check progress after starting download
                fn=check_download_progress,
                outputs=[current_download_info]
            ).then(
                # After starting download, refresh the model list to update status
                fn=lambda: load_available_models("", "All"),
                outputs=[available_models]
            )
            
            # Wire up force download button
            force_download_btn.click(
                fn=lambda model: download_selected_model(model, force=True),
                inputs=[selected_model],
                outputs=[download_status]
            ).then(
                # Immediately check progress after starting download
                fn=check_download_progress,
                outputs=[current_download_info]
            ).then(
                # After starting download, refresh the model list to update status
                fn=lambda: load_available_models("", "All"),
                outputs=[available_models]
            )
        
        # Manage downloaded models
        with gr.Tab("Installed Models"):
            gr.Markdown("### Manage Installed Models")
            
            # Model type tabs
            with gr.Tabs():
                # Image models
                with gr.Tab("Image Models"):
                    def load_image_models():
                        """Load installed image models"""
                        try:
                            downloaded = app.model_manager.get_downloaded_models("image")
                            models_data = []
                            
                            for model_name in downloaded:
                                model_path = app.model_manager.models_dir / "image" / model_name
                                if not model_path.exists():
                                    # Check other locations
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
                            logger.error(f"Error loading image models: {e}")
                            return []
                    
                    image_models_list = gr.DataFrame(
                        value=load_image_models(),
                        headers=["Name", "Path", "Size", "Last Used"],
                        datatype=["str", "str", "str", "str"],
                    )
                    
                    with gr.Row():
                        refresh_image_btn = create_action_button("🔄 Refresh", size="sm")
                        delete_image_btn = create_action_button("🗑️ Delete Selected", variant="stop", size="sm")
                    
                    refresh_image_btn.click(
                        load_image_models,
                        outputs=[image_models_list]
                    )
                
                # 3D models
                with gr.Tab("3D Models"):
                    def load_3d_models():
                        """Load installed 3D models"""
                        try:
                            downloaded = app.model_manager.get_downloaded_models("3d")
                            models_data = []
                            
                            for model_name in downloaded:
                                model_path = app.model_manager.models_dir / "3d" / model_name
                                
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
                            logger.error(f"Error loading 3D models: {e}")
                            return []
                    
                    threed_models_list = gr.DataFrame(
                        value=load_3d_models(),
                        headers=["Name", "Path", "Size", "Last Used"],
                        datatype=["str", "str", "str", "str"],
                    )
                    
                    with gr.Row():
                        refresh_3d_btn = create_action_button("🔄 Refresh", size="sm")
                        delete_3d_btn = create_action_button("🗑️ Delete Selected", variant="stop", size="sm")
                    
                    refresh_3d_btn.click(
                        load_3d_models,
                        outputs=[threed_models_list]
                    )
                
                # LoRA models
                with gr.Tab("LoRA Models"):
                    lora_models_list = gr.DataFrame(
                        headers=["Name", "Base Model", "Trigger Words", "Size"],
                        datatype=["str", "str", "str", "str"],
                    )
                    
                    with gr.Row():
                        refresh_lora_btn = create_action_button("🔄 Refresh", size="sm")
                        delete_lora_btn = create_action_button("🗑️ Delete Selected", variant="stop", size="sm")
        
        # Texture Components
        with gr.Tab("Texture Components"):
            create_texture_components_tab(app)
        
        # Model conversion tools
        with gr.Tab("Model Tools"):
            gr.Markdown("### Model Conversion & Optimization")
            
            # GGUF conversion
            with gr.Group():
                gr.Markdown("#### Convert to GGUF Format")
                
                # Get list of available models for conversion
                def get_convertible_models():
                    """Get list of downloaded models that can be converted to GGUF"""
                    try:
                        choices = []
                        # Get downloaded image models
                        image_models = app.model_manager.get_downloaded_models("image")
                        for model in image_models:
                            # Check if it's not already a GGUF model
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
                
                # Refresh the dropdown when clicked
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


def create_texture_components_tab(app: Any) -> None:
    """Create UI for texture pipeline components"""
    from ...config import TEXTURE_PIPELINE_COMPONENTS
    
    gr.Markdown("### 🎨 Essential Texture Pipeline Components")
    gr.Markdown("""
    These components are **required** for generating textures on your 3D models. 
    Without them, HunYuan3D will only generate untextured (gray) 3D models.
    """)
    
    # Quick install all button
    with gr.Row():
        install_all_btn = create_action_button(
            "🚀 Install All Texture Components",
            variant="primary",
            size="lg"
        )
        status_all = gr.HTML()
    
    def install_all_components(progress=gr.Progress()):
        """Install all texture components"""
        results = []
        components = list(TEXTURE_PIPELINE_COMPONENTS.items())
        total = len(components)
        
        progress(0, desc="Starting texture component installation...")
        
        # Install each component
        for idx, (comp_name, comp_info) in enumerate(components):
            progress((idx / total), desc=f"Installing {comp_info.get('name', comp_name)}...")
            
            success, message, _ = app.model_manager.download_texture_component(
                component_name=comp_name,
                component_info=comp_info,
                priority=1
            )
            
            status_icon = "✅" if success else "❌"
            results.append(f"{status_icon} {comp_name}: {message}")
            
            # Update progress
            progress((idx + 1) / total, desc=f"Completed {comp_info.get('name', comp_name)}")
        
        # Special handling for RealESRGAN
        progress(0.9, desc="Moving RealESRGAN to HunYuan3D directory...")
        source_path = app.model_manager.models_dir / "texture_components" / "realesrgan" / "RealESRGAN_x4plus.pth"
        target_path = Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
        
        if source_path.exists() and not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(source_path, target_path)
            results.append("✅ RealESRGAN: Moved to HunYuan3D directory")
        
        progress(1.0, desc="Installation complete!")
        
        return "<div style='padding: 10px;'>" + "<br>".join(results) + "</div>"
    
    install_all_btn.click(
        install_all_components,
        outputs=[status_all]
    )
    
    # Individual components
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 🖼️ RealESRGAN (Texture Enhancement)")
            gr.Markdown("**Required for:** High-quality texture generation")
            gr.Markdown("**Size:** ~64 MB")
            
            # Check if installed
            realesrgan_path = Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
            realesrgan_status_display = gr.HTML(
                value=f"<div style='color: {'green' if realesrgan_path.exists() else 'red'};'>{'✅ Installed' if realesrgan_path.exists() else '❌ Not Installed'}</div>"
            )
                
            with gr.Row():
                install_realesrgan_btn = create_action_button("Install RealESRGAN", variant="primary")
                realesrgan_status = gr.HTML()
                
            def install_realesrgan(progress=gr.Progress()):
                progress(0, desc="Starting RealESRGAN download...")
                
                comp_info = TEXTURE_PIPELINE_COMPONENTS.get("realesrgan", {})
                success, message, _ = app.model_manager.download_texture_component(
                    component_name="realesrgan",
                    component_info=comp_info,
                    priority=1
                )
                
                progress(0.5, desc="Download complete, moving to HunYuan3D directory...")
                
                # Move to correct location
                if success:
                    source = app.model_manager.models_dir / "texture_components" / "realesrgan" / "RealESRGAN_x4plus.pth"
                    target = Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
                    if source.exists() and not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy2(source, target)
                        progress(1.0, desc="Installation complete!")
                        result = f"<div style='color: green;'>✅ {message} and moved to HunYuan3D</div>"
                    else:
                        progress(1.0, desc="Installation complete!")
                        result = f"<div style='color: green;'>✅ {message}</div>"
                else:
                    progress(1.0, desc="Installation failed")
                    result = f"<div style='color: red;'>❌ {message}</div>"
                
                # Return both the result and updated status
                return result, "<div style='color: green;'>✅ Installed</div>" if success else "<div style='color: red;'>❌ Not Installed</div>"
            
            install_realesrgan_btn.click(
                install_realesrgan,
                outputs=[realesrgan_status, realesrgan_status_display]
            )
            
        with gr.Column():
            gr.Markdown("#### 📐 xatlas (UV Mapping)")
            gr.Markdown("**Required for:** Proper texture mapping on 3D models")
            gr.Markdown("**Size:** ~2 MB")
            
            # Check if installed
            try:
                import xatlas
                xatlas_installed = True
            except ImportError:
                xatlas_installed = False
                
            xatlas_status_display = gr.HTML(
                value=f"<div style='color: {'green' if xatlas_installed else 'red'};'>{'✅ Installed' if xatlas_installed else '❌ Not Installed'}</div>"
            )
                
            with gr.Row():
                install_xatlas_btn = create_action_button("Install xatlas", variant="primary")
                xatlas_status = gr.HTML()
                
            def install_xatlas(progress=gr.Progress()):
                progress(0, desc="Installing xatlas package...")
                
                comp_info = TEXTURE_PIPELINE_COMPONENTS.get("xatlas", {})
                success, message, _ = app.model_manager.download_texture_component(
                    component_name="xatlas",
                    component_info=comp_info,
                    priority=1
                )
                
                progress(1.0, desc="Installation complete!" if success else "Installation failed")
                
                result = f"<div style='color: {'green' if success else 'red'};'>{'✅' if success else '❌'} {message}</div>"
                status = "<div style='color: green;'>✅ Installed</div>" if success else "<div style='color: red;'>❌ Not Installed</div>"
                
                return result, status
            
            install_xatlas_btn.click(
                install_xatlas,
                outputs=[xatlas_status, xatlas_status_display]
            )
    
    # Optional components
    gr.Markdown("### 🔧 Optional Components")
    gr.Markdown("These components enhance texture quality but are not strictly required.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 🔍 DINOv2 (Feature Extraction)")
            gr.Markdown("**Enhances:** Texture detail and consistency")
            gr.Markdown("**Size:** ~4.5 GB")
            
            # Check if installed
            dinov2_path = app.model_manager.models_dir / "texture_components" / "dinov2"
            dinov2_installed = dinov2_path.exists() and any(dinov2_path.iterdir()) if dinov2_path.exists() else False
            
            dinov2_status_display = gr.HTML(
                value=f"<div style='color: {'green' if dinov2_installed else 'red'};'>{'✅ Installed' if dinov2_installed else '❌ Not Installed'}</div>"
            )
            
            with gr.Row():
                install_dino_btn = create_action_button("Install DINOv2", variant="secondary", size="sm")
                dino_status = gr.HTML()
            
            def install_dinov2(progress=gr.Progress()):
                progress(0, desc="Starting DINOv2 download...")
                
                comp_info = TEXTURE_PIPELINE_COMPONENTS.get("dinov2", {})
                success, message, _ = app.model_manager.download_texture_component(
                    component_name="dinov2",
                    component_info=comp_info,
                    priority=1
                )
                
                progress(1.0, desc="Installation complete!" if success else "Installation failed")
                
                result = f"<div style='color: {'green' if success else 'red'};'>{'✅' if success else '❌'} {message}</div>"
                status = "<div style='color: green;'>✅ Installed</div>" if success else "<div style='color: red;'>❌ Not Installed</div>"
                
                return result, status
            
            install_dino_btn.click(
                install_dinov2,
                outputs=[dino_status, dinov2_status_display]
            )
            
        with gr.Column():
            gr.Markdown("#### 🎨 Background Remover")
            gr.Markdown("**Enhances:** Clean 3D generation from images")
            gr.Markdown("**Size:** ~176 MB")
            
            # Check if installed
            rembg_path = app.model_manager.models_dir / "texture_components" / "background_remover"
            rembg_installed = rembg_path.exists() and any(rembg_path.iterdir()) if rembg_path.exists() else False
            
            rembg_status_display = gr.HTML(
                value=f"<div style='color: {'green' if rembg_installed else 'red'};'>{'✅ Installed' if rembg_installed else '❌ Not Installed'}</div>"
            )
            
            with gr.Row():
                install_rembg_btn = create_action_button("Install RemBG", variant="secondary", size="sm")
                rembg_status = gr.HTML()
            
            def install_rembg(progress=gr.Progress()):
                progress(0, desc="Starting RemBG download...")
                
                comp_info = TEXTURE_PIPELINE_COMPONENTS.get("background_remover", {})
                success, message, _ = app.model_manager.download_texture_component(
                    component_name="background_remover",
                    component_info=comp_info,
                    priority=1
                )
                
                progress(1.0, desc="Installation complete!" if success else "Installation failed")
                
                result = f"<div style='color: {'green' if success else 'red'};'>{'✅' if success else '❌'} {message}</div>"
                status = "<div style='color: green;'>✅ Installed</div>" if success else "<div style='color: red;'>❌ Not Installed</div>"
                
                return result, status
            
            install_rembg_btn.click(
                install_rembg,
                outputs=[rembg_status, rembg_status_display]
            )