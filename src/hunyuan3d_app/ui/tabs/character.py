"""Character consistency studio tab for enhanced UI"""

import gradio as gr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json


def create_character_studio_tab(app: Any) -> None:
    """Create the character consistency studio tab
    
    Args:
        app: The enhanced application instance
    """
    gr.Markdown("""
    ### 👤 Character Consistency Studio
    Create and manage consistent characters across all your generations.
    """)
    
    with gr.Tabs():
        # Create Character Tab
        with gr.Tab("✨ Create Character"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Character info
                    char_name = gr.Textbox(
                        label="Character Name",
                        placeholder="e.g., Aria the Adventurer"
                    )
                    
                    char_description = gr.Textbox(
                        label="Description",
                        placeholder="A brave explorer with red hair and green eyes...",
                        lines=3
                    )
                    
                    # Reference images
                    reference_images = gr.File(
                        label="Reference Images",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    
                    # Attributes
                    with gr.Accordion("Character Attributes", open=False):
                        gender = gr.Dropdown(
                            choices=["", "male", "female", "other"],
                            label="Gender"
                        )
                        
                        age = gr.Textbox(
                            label="Age",
                            placeholder="e.g., 25, young adult"
                        )
                        
                        hair_color = gr.Textbox(
                            label="Hair Color",
                            placeholder="e.g., red, blonde, black"
                        )
                        
                        eye_color = gr.Textbox(
                            label="Eye Color",
                            placeholder="e.g., green, blue, brown"
                        )
                        
                        style = gr.Textbox(
                            label="Art Style",
                            placeholder="e.g., realistic, anime, fantasy"
                        )
                        
                    # Extraction options
                    with gr.Accordion("Extraction Settings", open=True):
                        extract_face = gr.Checkbox(
                            label="Extract Face Features",
                            value=True
                        )
                        
                        extract_style = gr.Checkbox(
                            label="Extract Style Features",
                            value=True
                        )
                        
                    create_btn = gr.Button(
                        "🎨 Create Character",
                        variant="primary",
                        size="lg"
                    )
                    
                with gr.Column(scale=1):
                    # Preview and status
                    preview_gallery = gr.Gallery(
                        label="Reference Images Preview",
                        columns=2,
                        height="auto"
                    )
                    
                    creation_status = gr.HTML()
                    
                    # Created character info
                    created_char_info = gr.JSON(
                        label="Character Profile",
                        visible=False
                    )
                    
        # Manage Characters Tab
        with gr.Tab("📚 Character Gallery"):
            with gr.Row():
                refresh_gallery_btn = gr.Button("🔄 Refresh Gallery")
                search_box = gr.Textbox(
                    label="Search Characters",
                    placeholder="Search by name or attributes..."
                )
                
            character_gallery = gr.HTML()
            
            with gr.Row():
                selected_char_id = gr.Textbox(
                    label="Selected Character ID",
                    visible=False
                )
                
                with gr.Column():
                    edit_name_btn = gr.Button("✏️ Edit Name")
                    edit_desc_btn = gr.Button("📝 Edit Description")
                    add_images_btn = gr.Button("➕ Add Images")
                    
                with gr.Column():
                    export_btn = gr.Button("📤 Export Character")
                    blend_btn = gr.Button("🔀 Blend Characters")
                    delete_btn = gr.Button("🗑️ Delete Character", variant="stop")
                    
        # Character Blending Tab
        with gr.Tab("🔀 Blend Characters"):
            gr.Markdown("""
            Blend multiple characters to create new unique characters.
            """)
            
            with gr.Row():
                # Character selection
                blend_char1 = gr.Dropdown(
                    label="Character 1",
                    choices=[]
                )
                weight1 = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Weight 1"
                )
                
            with gr.Row():
                blend_char2 = gr.Dropdown(
                    label="Character 2",
                    choices=[]
                )
                weight2 = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Weight 2"
                )
                
            with gr.Row():
                blend_char3 = gr.Dropdown(
                    label="Character 3 (Optional)",
                    choices=["None"]
                )
                weight3 = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Weight 3"
                )
                
            blended_name = gr.Textbox(
                label="Blended Character Name",
                placeholder="e.g., Hybrid Hero"
            )
            
            blend_execute_btn = gr.Button(
                "🔀 Create Blended Character",
                variant="primary"
            )
            
            blend_status = gr.HTML()
            
        # Import/Export Tab
        with gr.Tab("📦 Import/Export"):
            gr.Markdown("""
            Import and export character profiles for sharing or backup.
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Import Character")
                    import_file = gr.File(
                        label="Character Package (.zip)",
                        file_types=[".zip"]
                    )
                    import_btn = gr.Button("📥 Import Character")
                    import_status = gr.HTML()
                    
                with gr.Column():
                    gr.Markdown("### Export Character")
                    export_char_dropdown = gr.Dropdown(
                        label="Select Character to Export",
                        choices=[]
                    )
                    include_images = gr.Checkbox(
                        label="Include Reference Images",
                        value=True
                    )
                    export_execute_btn = gr.Button("📤 Export Character")
                    export_status = gr.HTML()
                    export_download = gr.File(
                        label="Download",
                        visible=False
                    )
                    
    # Helper functions
    def update_preview(files):
        """Update reference image preview"""
        if not files:
            return []
        
        images = []
        for file in files:
            if hasattr(file, 'name'):
                images.append(file.name)
            else:
                images.append(file)
        return images
        
    reference_images.change(
        update_preview,
        inputs=[reference_images],
        outputs=[preview_gallery]
    )
    
    # Create character function
    def create_character(
        name, description, ref_images,
        gender, age, hair_color, eye_color, style,
        extract_face, extract_style
    ):
        """Create a new character profile"""
        try:
            if not name:
                return gr.update(), "<p style='color: red;'>Please enter a character name</p>", gr.update()
                
            if not ref_images:
                return gr.update(), "<p style='color: red;'>Please upload at least one reference image</p>", gr.update()
                
            # Prepare attributes
            attributes = {}
            if gender:
                attributes["gender"] = gender
            if age:
                attributes["age"] = age
            if hair_color:
                attributes["hair_color"] = hair_color
            if eye_color:
                attributes["eye_color"] = eye_color
            if style:
                attributes["style"] = style
                
            # Get image paths
            image_paths = []
            for img in ref_images:
                if hasattr(img, 'name'):
                    image_paths.append(img.name)
                else:
                    image_paths.append(img)
                    
            # Create character
            character, msg = app.character_consistency_manager.create_character(
                name=name,
                reference_images=image_paths,
                description=description,
                attributes=attributes,
                extract_style=extract_style,
                extract_face=extract_face
            )
            
            if character:
                status_html = f"""
                <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                    <h4>✅ Character Created Successfully!</h4>
                    <p><strong>ID:</strong> {character.id}</p>
                    <p><strong>Name:</strong> {character.name}</p>
                    <p>{msg}</p>
                </div>
                """
                
                return (
                    gr.update(value=""),  # Clear name
                    status_html,
                    gr.update(value=character.to_dict(), visible=True)
                )
            else:
                return (
                    gr.update(),
                    f"<p style='color: red;'>{msg}</p>",
                    gr.update()
                )
                
        except Exception as e:
            return (
                gr.update(),
                f"<p style='color: red;'>Error: {str(e)}</p>",
                gr.update()
            )
            
    create_btn.click(
        create_character,
        inputs=[
            char_name, char_description, reference_images,
            gender, age, hair_color, eye_color, style,
            extract_face, extract_style
        ],
        outputs=[char_name, creation_status, created_char_info]
    )
    
    # Character gallery functions
    def render_character_gallery(search_query=""):
        """Render the character gallery"""
        try:
            characters = app.character_consistency_manager.list_characters()
            
            # Filter by search query
            if search_query:
                query_lower = search_query.lower()
                characters = [
                    c for c in characters
                    if query_lower in c.name.lower() or
                    query_lower in c.description.lower() or
                    any(query_lower in str(v).lower() for v in c.attributes.values())
                ]
                
            if not characters:
                return "<p>No characters found. Create your first character!</p>"
                
            # Build gallery HTML
            html = """
            <div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px;'>
            """
            
            for char in characters:
                # Get first reference image if available
                preview_img = ""
                if char.reference_images and char.reference_images[0].exists():
                    preview_img = f"<img src='{char.reference_images[0]}' style='width: 100%; height: 150px; object-fit: cover;'>"
                else:
                    preview_img = "<div style='width: 100%; height: 150px; background: #ddd; display: flex; align-items: center; justify-content: center;'>No Preview</div>"
                    
                html += f"""
                <div class='character-card' style='border: 1px solid #ddd; border-radius: 8px; overflow: hidden; cursor: pointer;'
                     onclick='selectCharacter("{char.id}")'>
                    {preview_img}
                    <div style='padding: 10px;'>
                        <h4 style='margin: 0 0 5px 0;'>{char.name}</h4>
                        <p style='margin: 0; font-size: 0.9em; color: #666;'>{char.description[:50]}...</p>
                        <p style='margin: 5px 0 0 0; font-size: 0.8em; color: #999;'>
                            ID: {char.id[:8]}... | Images: {len(char.reference_images)}
                        </p>
                    </div>
                </div>
                """
                
            html += """
            </div>
            <script>
            function selectCharacter(charId) {
                // Update hidden textbox with character ID
                document.querySelector('#selected_char_id textarea').value = charId;
                document.querySelector('#selected_char_id textarea').dispatchEvent(new Event('input'));
            }
            </script>
            """
            
            return html
            
        except Exception as e:
            return f"<p style='color: red;'>Error loading gallery: {str(e)}</p>"
            
    def refresh_gallery():
        """Refresh the character gallery"""
        gallery_html = render_character_gallery()
        
        # Update dropdowns
        characters = app.character_consistency_manager.list_characters()
        char_choices = [(f"{c.name} ({c.id[:8]}...)", c.id) for c in characters]
        char_choices_with_none = [("None", "None")] + char_choices
        
        return (
            gallery_html,
            gr.update(choices=char_choices),  # blend_char1
            gr.update(choices=char_choices),  # blend_char2
            gr.update(choices=char_choices_with_none),  # blend_char3
            gr.update(choices=char_choices)   # export dropdown
        )
        
    refresh_gallery_btn.click(
        refresh_gallery,
        outputs=[character_gallery, blend_char1, blend_char2, blend_char3, export_char_dropdown]
    )
    
    search_box.change(
        render_character_gallery,
        inputs=[search_box],
        outputs=[character_gallery]
    )
    
    # Character blending
    def blend_characters(char1_id, weight1, char2_id, weight2, char3_id, weight3, blend_name):
        """Blend multiple characters"""
        try:
            if not char1_id or not char2_id:
                return "<p style='color: red;'>Please select at least 2 characters to blend</p>"
                
            if not blend_name:
                return "<p style='color: red;'>Please enter a name for the blended character</p>"
                
            # Prepare character IDs and weights
            char_ids = [char1_id, char2_id]
            weights = [weight1, weight2]
            
            if char3_id and char3_id != "None":
                char_ids.append(char3_id)
                weights.append(weight3)
                
            # Blend characters
            blended, msg = app.character_consistency_manager.blend_characters(
                character_ids=char_ids,
                weights=weights,
                name=blend_name
            )
            
            if blended:
                return f"""
                <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                    <h4>✅ Characters Blended Successfully!</h4>
                    <p><strong>New Character:</strong> {blended.name}</p>
                    <p><strong>ID:</strong> {blended.id}</p>
                    <p>{msg}</p>
                </div>
                """
            else:
                return f"<p style='color: red;'>{msg}</p>"
                
        except Exception as e:
            return f"<p style='color: red;'>Error: {str(e)}</p>"
            
    blend_execute_btn.click(
        blend_characters,
        inputs=[blend_char1, weight1, blend_char2, weight2, blend_char3, weight3, blended_name],
        outputs=[blend_status]
    )
    
    # Character export
    def export_character(char_id, include_imgs):
        """Export a character"""
        try:
            if not char_id:
                return "<p style='color: red;'>Please select a character to export</p>", gr.update()
                
            output_path = Path(app.output_dir) / f"character_{char_id[:8]}.zip"
            
            success, msg = app.character_consistency_manager.export_character(
                character_id=char_id,
                output_path=output_path,
                include_images=include_imgs
            )
            
            if success:
                return (
                    f"""
                    <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                        <h4>✅ Character Exported!</h4>
                        <p>{msg}</p>
                    </div>
                    """,
                    gr.update(value=str(output_path), visible=True)
                )
            else:
                return f"<p style='color: red;'>{msg}</p>", gr.update()
                
        except Exception as e:
            return f"<p style='color: red;'>Error: {str(e)}</p>", gr.update()
            
    export_execute_btn.click(
        export_character,
        inputs=[export_char_dropdown, include_images],
        outputs=[export_status, export_download]
    )
    
    # Add refresh button since HTML components don't support .load()
    refresh_gallery_btn = gr.Button("🔄 Refresh Gallery", variant="secondary")
    refresh_gallery_btn.click(
        refresh_gallery,
        outputs=[character_gallery, blend_char1, blend_char2, blend_char3, export_char_dropdown]
    )