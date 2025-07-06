"""LoRA Management UI tab for downloading, organizing, and using LoRAs"""

import gradio as gr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

from ...features.lora.manager import LoRAInfo


def create_lora_tab(app: Any) -> None:
    """Create the LoRA management tab
    
    Args:
        app: The enhanced application instance
    """
    gr.Markdown("""
    ### 🎨 LoRA Studio
    Download, manage, and apply LoRAs to enhance your generations with specific styles and concepts.
    """)
    
    with gr.Tabs():
        # LoRA Gallery & Management
        with gr.Tab("📚 LoRA Gallery"):
            with gr.Row():
                # Search and filters
                with gr.Column(scale=1):
                    search_query = gr.Textbox(
                        label="Search LoRAs",
                        placeholder="Search by name, style, or trigger words..."
                    )
                    
                    category_filter = gr.Dropdown(
                        choices=["All", "Character", "Style", "Concept", "Object", "Environment"],
                        value="All",
                        label="Category Filter"
                    )
                    
                    base_model_filter = gr.Dropdown(
                        choices=["All", "FLUX.1", "SDXL", "SD 1.5"],
                        value="All",
                        label="Base Model"
                    )
                    
                    refresh_gallery_btn = gr.Button("🔄 Refresh Gallery")
                    
                with gr.Column(scale=3):
                    lora_gallery = gr.HTML()
                    
            # Selected LoRA details
            with gr.Row():
                selected_lora_id = gr.Textbox(
                    label="Selected LoRA",
                    visible=False
                )
                
                with gr.Column():
                    lora_details = gr.HTML()
                    
                with gr.Column():
                    # LoRA actions
                    with gr.Group():
                        gr.Markdown("### LoRA Actions")
                        
                        test_lora_btn = gr.Button("🎨 Test Generate")
                        edit_lora_btn = gr.Button("✏️ Edit Details") 
                        delete_lora_btn = gr.Button("🗑️ Delete LoRA", variant="stop")
                        
                        lora_action_status = gr.HTML()
                        
        # Download LoRAs
        with gr.Tab("📥 Download LoRAs"):
            with gr.Tabs():
                # Civitai Search
                with gr.Tab("🎨 Civitai"):
                    with gr.Row():
                        civitai_search_query = gr.Textbox(
                            label="Search Civitai",
                            placeholder="anime, realistic, style, character..."
                        )
                        
                        civitai_model_type = gr.Dropdown(
                            choices=["LORA", "LoCon", "DoRA"],
                            value="LORA",
                            label="Model Type"
                        )
                        
                        civitai_base_model = gr.Dropdown(
                            choices=["FLUX.1", "SDXL", "SD 1.5", "Pony"],
                            value="FLUX.1",
                            label="Base Model"
                        )
                        
                        civitai_search_btn = gr.Button("🔍 Search Civitai", variant="primary")
                        
                    civitai_results = gr.HTML()
                    
                    with gr.Row():
                        selected_civitai_model = gr.Textbox(
                            label="Selected Model ID",
                            visible=False
                        )
                        
                        download_civitai_btn = gr.Button(
                            "📥 Download Selected",
                            variant="primary",
                            visible=False
                        )
                        
                    civitai_download_status = gr.HTML()
                    
                # HuggingFace Search
                with gr.Tab("🤗 HuggingFace"):
                    with gr.Row():
                        hf_search_query = gr.Textbox(
                            label="Search HuggingFace",
                            placeholder="organization/model-name or search terms..."
                        )
                        
                        hf_search_btn = gr.Button("🔍 Search HF", variant="primary")
                        
                    hf_results = gr.HTML()
                    hf_download_status = gr.HTML()
                    
                # Manual Upload
                with gr.Tab("📂 Manual Upload"):
                    with gr.Row():
                        with gr.Column():
                            upload_lora_file = gr.File(
                                label="LoRA File",
                                file_types=[".safetensors", ".ckpt", ".pth"]
                            )
                            
                            upload_name = gr.Textbox(
                                label="LoRA Name",
                                placeholder="My Custom LoRA"
                            )
                            
                            upload_description = gr.Textbox(
                                label="Description",
                                placeholder="Description of this LoRA...",
                                lines=3
                            )
                            
                            upload_base_model = gr.Dropdown(
                                choices=["FLUX.1", "SDXL", "SD 1.5"],
                                value="FLUX.1",
                                label="Base Model"
                            )
                            
                            upload_category = gr.Dropdown(
                                choices=["Character", "Style", "Concept", "Object", "Environment"],
                                value="Style",
                                label="Category"
                            )
                            
                            upload_trigger_words = gr.Textbox(
                                label="Trigger Words",
                                placeholder="trigger1, trigger2, trigger3"
                            )
                            gr.Markdown("*Comma-separated trigger words*")
                            
                            upload_btn = gr.Button(
                                "📂 Upload LoRA",
                                variant="primary"
                            )
                            
                        with gr.Column():
                            upload_preview = gr.Image(
                                label="Preview Image (Optional)",
                                type="pil"
                            )
                            
                            upload_status = gr.HTML()
                            
        # LoRA Generation
        with gr.Tab("🎨 Generate with LoRAs"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Main prompt
                    gen_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Your generation prompt...",
                        lines=3
                    )
                    
                    gen_negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="What to avoid...",
                        lines=2
                    )
                    
                    # LoRA selection
                    with gr.Accordion("🎨 LoRA Selection", open=True):
                        # LoRA 1
                        with gr.Group():
                            gr.Markdown("#### LoRA 1")
                            lora_1_dropdown = gr.Dropdown(
                                label="Select LoRA",
                                choices=[],
                                value=None
                            )
                            lora_1_weight = gr.Slider(
                                minimum=-2.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Weight"
                            )
                            lora_1_triggers = gr.HTML(visible=False)
                            
                        # LoRA 2
                        with gr.Group():
                            gr.Markdown("#### LoRA 2 (Optional)")
                            lora_2_dropdown = gr.Dropdown(
                                label="Select LoRA",
                                choices=[],
                                value=None
                            )
                            lora_2_weight = gr.Slider(
                                minimum=-2.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Weight"
                            )
                            lora_2_triggers = gr.HTML(visible=False)
                            
                        # LoRA 3
                        with gr.Group():
                            gr.Markdown("#### LoRA 3 (Optional)")
                            lora_3_dropdown = gr.Dropdown(
                                label="Select LoRA",
                                choices=[],
                                value=None
                            )
                            lora_3_weight = gr.Slider(
                                minimum=-2.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Weight"
                            )
                            lora_3_triggers = gr.HTML(visible=False)
                            
                    # Auto-suggestion
                    with gr.Accordion("🤖 Auto-Suggest LoRAs", open=False):
                        auto_suggest_btn = gr.Button("🔍 Suggest LoRAs for Prompt")
                        suggestion_results = gr.HTML()
                        
                        apply_suggestions_btn = gr.Button(
                            "✨ Apply Suggestions",
                            variant="secondary",
                            visible=False
                        )
                        
                    # Generation settings
                    with gr.Accordion("⚙️ Generation Settings", open=True):
                        with gr.Row():
                            gen_width = gr.Slider(512, 2048, 1024, step=64, label="Width")
                            gen_height = gr.Slider(512, 2048, 1024, step=64, label="Height")
                            
                        with gr.Row():
                            gen_steps = gr.Slider(4, 50, 28, step=1, label="Steps")
                            gen_guidance = gr.Slider(1.0, 15.0, 3.5, step=0.5, label="Guidance")
                            
                        gen_seed = gr.Number(-1, label="Seed (-1 for random)")
                        
                    generate_lora_btn = gr.Button(
                        "🎨 Generate with LoRAs",
                        variant="primary",
                        size="lg"
                    )
                    
                with gr.Column(scale=1):
                    # Output
                    lora_output_image = gr.Image(
                        label="Generated Image",
                        type="pil"
                    )
                    
                    lora_generation_info = gr.HTML()
                    
        # Batch LoRA Testing
        with gr.Tab("🧪 LoRA Testing"):
            gr.Markdown("""
            Test multiple LoRAs with the same prompt to compare their effects.
            """)
            
            with gr.Row():
                with gr.Column():
                    test_prompt = gr.Textbox(
                        label="Test Prompt",
                        placeholder="a portrait of a woman",
                        lines=2
                    )
                    
                    test_loras = gr.CheckboxGroup(
                        label="LoRAs to Test",
                        choices=[],
                        value=[]
                    )
                    
                    test_weight = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Test Weight"
                    )
                    
                    test_batch_btn = gr.Button(
                        "🧪 Run LoRA Test",
                        variant="primary"
                    )
                    
                with gr.Column():
                    test_progress = gr.HTML()
                    test_results = gr.Gallery(
                        label="Test Results",
                        columns=3,
                        height="auto"
                    )
                    
    # Helper functions
    def render_lora_gallery(search="", category="All", base_model="All"):
        """Render the LoRA gallery"""
        try:
            if not hasattr(app, 'lora_manager'):
                return "<p>LoRA Manager not available</p>"
                
            # Get all LoRAs
            loras = app.lora_manager.scan_lora_directory()
            
            # Apply filters
            filtered_loras = []
            for lora in loras:
                # Search filter
                if search:
                    search_lower = search.lower()
                    if not (search_lower in lora.name.lower() or
                           search_lower in lora.description.lower() or
                           any(search_lower in word.lower() for word in lora.trigger_words)):
                        continue
                        
                # Category filter
                if category != "All" and lora.category != category:
                    continue
                    
                # Base model filter
                if base_model != "All" and lora.base_model != base_model:
                    continue
                    
                filtered_loras.append(lora)
                
            if not filtered_loras:
                return "<p>No LoRAs found matching your criteria.</p>"
                
            # Build gallery HTML
            html = """
            <div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px;'>
            """
            
            for lora in filtered_loras:
                # Get preview image
                preview_img = ""
                if lora.preview_image and lora.preview_image.exists():
                    preview_img = f"<img src='{lora.preview_image}' style='width: 100%; height: 180px; object-fit: cover;'>"
                else:
                    preview_img = "<div style='width: 100%; height: 180px; background: linear-gradient(45deg, #f0f0f0, #e0e0e0); display: flex; align-items: center; justify-content: center; color: #666;'>No Preview</div>"
                    
                # Build trigger words display
                triggers = ", ".join(lora.trigger_words[:3])
                if len(lora.trigger_words) > 3:
                    triggers += f" (+{len(lora.trigger_words)-3} more)"
                    
                html += f"""
                <div class='lora-card' style='border: 1px solid #ddd; border-radius: 12px; overflow: hidden; cursor: pointer; transition: transform 0.2s;'
                     onclick='selectLoRA("{lora.name}")'>
                    {preview_img}
                    <div style='padding: 15px;'>
                        <h4 style='margin: 0 0 8px 0; color: #333;'>{lora.name}</h4>
                        <p style='margin: 0 0 8px 0; font-size: 0.9em; color: #666; height: 40px; overflow: hidden;'>{lora.description[:80]}...</p>
                        <div style='display: flex; gap: 8px; margin-bottom: 8px;'>
                            <span style='background: #e3f2fd; color: #1976d2; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;'>{lora.base_model}</span>
                            <span style='background: #fff3e0; color: #e65100; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;'>{lora.file_size_mb:.1f}MB</span>
                        </div>
                        <p style='margin: 0; font-size: 0.8em; color: #999; font-style: italic;'>
                            Triggers: {triggers}
                        </p>
                    </div>
                </div>
                """
                
            html += """
            </div>
            <script>
            function selectLoRA(loraName) {
                // Update hidden textbox with LoRA name
                document.querySelector('#selected_lora_id textarea').value = loraName;
                document.querySelector('#selected_lora_id textarea').dispatchEvent(new Event('input'));
            }
            </script>
            """
            
            return html
            
        except Exception as e:
            return f"<p style='color: red;'>Error loading LoRA gallery: {str(e)}</p>"
            
    def refresh_lora_gallery():
        """Refresh LoRA gallery and update dropdowns"""
        gallery_html = render_lora_gallery()
        
        # Update LoRA dropdowns
        lora_choices = []
        if hasattr(app, 'lora_manager'):
            loras = app.lora_manager.scan_lora_directory()
            lora_choices = [(f"{lora.name} ({lora.base_model})", lora.name) for lora in loras]
            
        lora_names = [name for _, name in lora_choices]
        
        return (
            gallery_html,
            gr.update(choices=lora_choices),  # lora_1_dropdown
            gr.update(choices=lora_choices),  # lora_2_dropdown
            gr.update(choices=lora_choices),  # lora_3_dropdown
            gr.update(choices=lora_names, value=[])  # test_loras checkboxgroup
        )
        
    refresh_gallery_btn.click(
        refresh_lora_gallery,
        outputs=[lora_gallery, lora_1_dropdown, lora_2_dropdown, lora_3_dropdown, test_loras]
    )
    
    # LoRA details display
    def show_lora_details(lora_name):
        """Show details of selected LoRA"""
        try:
            if not lora_name or not hasattr(app, 'lora_manager'):
                return ""
                
            # Find the LoRA from the scanned list
            loras = app.lora_manager.scan_lora_directory()
            lora = next((l for l in loras if l.name == lora_name), None)
            if not lora:
                return "<p>LoRA not found</p>"
                
            # Build details HTML
            html = f"""
            <div style='padding: 20px; border: 1px solid #ddd; border-radius: 12px;'>
                <h3>{lora.name}</h3>
                <p><strong>Description:</strong> {lora.description}</p>
                <p><strong>Base Model:</strong> {lora.base_model}</p>
                <p><strong>File Size:</strong> {lora.file_size_mb:.1f} MB</p>
                <p><strong>File Path:</strong> {lora.path}</p>
                
                <h4>Trigger Words:</h4>
                <div style='display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0;'>
            """
            
            for trigger in lora.trigger_words:
                html += f"<span style='background: #e8f5e9; color: #2e7d32; padding: 4px 12px; border-radius: 16px; font-size: 0.9em;'>{trigger}</span>"
                
            html += """
                </div>
                
                <h4>Usage Notes:</h4>
                <ul>
            """
            
            if lora.usage_notes:
                for note in lora.usage_notes:
                    html += f"<li>{note}</li>"
            else:
                html += "<li>No specific usage notes</li>"
                
            html += """
                </ul>
            </div>
            """
            
            return html
            
        except Exception as e:
            return f"<p style='color: red;'>Error loading LoRA details: {str(e)}</p>"
            
    selected_lora_id.change(
        show_lora_details,
        inputs=[selected_lora_id],
        outputs=[lora_details]
    )
    
    # LoRA trigger words display
    def show_lora_triggers(lora_name):
        """Show trigger words for selected LoRA"""
        try:
            if not lora_name or not hasattr(app, 'lora_manager'):
                return gr.update(visible=False)
                
            # Find the LoRA from the scanned list
            loras = app.lora_manager.scan_lora_directory()
            lora = next((l for l in loras if l.name == lora_name), None)
            if not lora or not lora.trigger_words:
                return gr.update(visible=False)
                
            triggers_html = f"""
            <div style='margin-top: 8px; padding: 8px; background: #f5f5f5; border-radius: 8px;'>
                <strong>Trigger words:</strong> {', '.join(lora.trigger_words)}
            </div>
            """
            
            return gr.update(value=triggers_html, visible=True)
            
        except:
            return gr.update(visible=False)
            
    # Connect trigger display to dropdowns
    lora_1_dropdown.change(show_lora_triggers, inputs=[lora_1_dropdown], outputs=[lora_1_triggers])
    lora_2_dropdown.change(show_lora_triggers, inputs=[lora_2_dropdown], outputs=[lora_2_triggers])
    lora_3_dropdown.change(show_lora_triggers, inputs=[lora_3_dropdown], outputs=[lora_3_triggers])
    
    # Manual LoRA upload
    def handle_lora_upload(
        file, name, description, base_model, category, trigger_words, preview_img
    ):
        """Upload a LoRA file manually"""
        try:
            if not file:
                return "<p style='color: red;'>Please select a LoRA file</p>"
                
            if not name:
                return "<p style='color: red;'>Please enter a name for the LoRA</p>"
                
            if not hasattr(app, 'lora_manager'):
                return "<p style='color: red;'>LoRA Manager not available</p>"
                
            # Parse trigger words
            triggers = [t.strip() for t in trigger_words.split(',') if t.strip()] if trigger_words else []
            
            # Create LoRA info
            lora_info = LoRAInfo(
                name=name,
                path=Path(file.name),
                base_model=base_model,
                description=description,
                category=category,
                trigger_words=triggers,
                preview_image=Path(preview_img.name) if preview_img else None
            )
            
            # Add to manager
            success = app.lora_manager.add_lora(lora_info)
            
            if success:
                return f"""
                <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                    <h4>✅ LoRA Uploaded Successfully!</h4>
                    <p>"{name}" has been added to your LoRA collection</p>
                </div>
                """
            else:
                return "<p style='color: red;'>Failed to upload LoRA</p>"
                
        except Exception as e:
            return f"<p style='color: red;'>Error: {str(e)}</p>"
            
    upload_btn.click(
        handle_lora_upload,
        inputs=[
            upload_lora_file, upload_name, upload_description,
            upload_base_model, upload_category, upload_trigger_words, upload_preview
        ],
        outputs=[upload_status]
    )
    
    # Auto-suggest LoRAs
    def suggest_loras_for_prompt(prompt):
        """Suggest LoRAs based on prompt analysis"""
        try:
            if not prompt or not hasattr(app, 'lora_suggestion_manager'):
                return ""
                
            suggestions = app.lora_suggestion_manager.suggest_loras(prompt)
            
            if not suggestions:
                return "<p>No LoRA suggestions found for this prompt</p>"
                
            html = """
            <div style='padding: 15px; background: #f8f9fa; border-radius: 8px;'>
                <h4>🤖 Suggested LoRAs:</h4>
                <div style='display: flex; flex-direction: column; gap: 10px;'>
            """
            
            for suggestion in suggestions[:5]:  # Top 5 suggestions
                confidence = suggestion.get('confidence', 0) * 100
                lora_name = suggestion.get('lora_name', 'Unknown')
                reason = suggestion.get('reason', 'Good match for prompt')
                
                html += f"""
                <div style='padding: 10px; background: white; border-radius: 6px; border-left: 4px solid #2196f3;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <strong>{lora_name}</strong>
                        <span style='background: #e3f2fd; color: #1976d2; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;'>{confidence:.0f}% match</span>
                    </div>
                    <p style='margin: 5px 0 0 0; color: #666; font-size: 0.9em;'>{reason}</p>
                </div>
                """
                
            html += """
                </div>
            </div>
            """
            
            return html
            
        except Exception as e:
            return f"<p style='color: red;'>Error suggesting LoRAs: {str(e)}</p>"
            
    auto_suggest_btn.click(
        suggest_loras_for_prompt,
        inputs=[gen_prompt],
        outputs=[suggestion_results]
    )
    
    # Generate with LoRAs
    def generate_with_loras(
        prompt, negative_prompt,
        lora1, weight1, lora2, weight2, lora3, weight3,
        width, height, steps, guidance, seed
    ):
        """Generate image with selected LoRAs"""
        try:
            if not prompt:
                return None, "<p style='color: red;'>Please enter a prompt</p>"
                
            # Collect LoRA configurations
            lora_configs = []
            
            if lora1 and hasattr(app, 'lora_manager'):
                lora_info = app.lora_manager.get_lora(lora1)
                if lora_info:
                    lora_configs.append((lora_info, weight1))
                    
            if lora2 and hasattr(app, 'lora_manager'):
                lora_info = app.lora_manager.get_lora(lora2)
                if lora_info:
                    lora_configs.append((lora_info, weight2))
                    
            if lora3 and hasattr(app, 'lora_manager'):
                lora_info = app.lora_manager.get_lora(lora3)
                if lora_info:
                    lora_configs.append((lora_info, weight3))
                    
            # Submit generation job
            job_params = {
                "model_name": "flux-dev",  # Default model
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "lora_configs": lora_configs,
                "width": int(width),
                "height": int(height),
                "steps": int(steps),
                "guidance_scale": guidance,
                "seed": int(seed)
            }
            
            # This would submit to the queue manager
            job_id = app.submit_generation_job("image", job_params)
            
            # For now, return info about the job
            lora_names = [config[0].name for config in lora_configs]
            info_html = f"""
            <div style='padding: 10px; background: #e3f2fd; border-radius: 5px;'>
                <h4>🎨 Generation Submitted</h4>
                <p><strong>Job ID:</strong> {job_id[:8]}...</p>
                <p><strong>LoRAs Used:</strong> {', '.join(lora_names) if lora_names else 'None'}</p>
                <p>Check the Queue tab for progress</p>
            </div>
            """
            
            return None, info_html
            
        except Exception as e:
            return None, f"<p style='color: red;'>Error: {str(e)}</p>"
            
    generate_lora_btn.click(
        generate_with_loras,
        inputs=[
            gen_prompt, gen_negative_prompt,
            lora_1_dropdown, lora_1_weight,
            lora_2_dropdown, lora_2_weight,
            lora_3_dropdown, lora_3_weight,
            gen_width, gen_height, gen_steps, gen_guidance, gen_seed
        ],
        outputs=[lora_output_image, lora_generation_info]
    )
    
    # Civitai search
    def search_civitai_loras(query, model_type, base_model):
        """Search Civitai for LoRAs"""
        try:
            # This would implement actual Civitai API search
            # For now, return placeholder
            return f"""
            <div style='padding: 15px; background: #f8f9fa; border-radius: 8px;'>
                <h4>🔍 Civitai Search Results</h4>
                <p>Searching for "{query}" {model_type} models for {base_model}...</p>
                <p style='color: #666;'>Civitai integration coming soon!</p>
                <p>You can manually download LoRAs from <a href='https://civitai.com' target='_blank'>Civitai</a> and upload them using the Manual Upload tab.</p>
            </div>
            """
        except Exception as e:
            return f"<p style='color: red;'>Error: {str(e)}</p>"
            
    civitai_search_btn.click(
        search_civitai_loras,
        inputs=[civitai_search_query, civitai_model_type, civitai_base_model],
        outputs=[civitai_results]
    )
    
    # Initialize gallery - HTML components don't support .load()
    # The gallery will be populated when refresh button is clicked