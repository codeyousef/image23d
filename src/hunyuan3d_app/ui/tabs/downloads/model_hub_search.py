"""Model hub search functionality"""

import gradio as gr
from typing import Optional, List, Dict, Any


def create_model_hub_search_tab(app):
    """Create model hub search interface"""
    
    gr.Markdown("### 🔍 Search Model Hubs")
    gr.Markdown("Search and download models from Hugging Face and Civitai")
    
    with gr.Tabs():
        # Hugging Face Search
        create_huggingface_search_tab(app)
            
        # Civitai Search  
        create_civitai_search_tab(app)
            
    # Download queue status
    create_download_status_section(app)


def create_huggingface_search_tab(app):
    """Create Hugging Face search tab"""
    with gr.Tab("🤗 Hugging Face"):
        with gr.Row():
            search_query = gr.Textbox(
                label="Search Query",
                placeholder="stable-diffusion, flux, controlnet..."
            )
            search_type = gr.Dropdown(
                choices=["text-to-image", "image-to-image", "controlnet", "lora"],
                value="text-to-image",
                label="Model Type"
            )
            search_btn = gr.Button("Search", variant="primary")
            
        search_results = gr.HTML()
        
        def search_huggingface(query, model_type):
            # This would search HuggingFace models
            # For now, return placeholder
            return f"""
            <div class="info-box">
                <h4>🔍 Search Results</h4>
                <p>Searching for "{query}" in {model_type} models...</p>
                <p>HuggingFace search integration coming soon!</p>
            </div>
            """
            
        search_btn.click(
            search_huggingface,
            inputs=[search_query, search_type],
            outputs=[search_results]
        )


def create_civitai_search_tab(app):
    """Create Civitai search tab"""
    with gr.Tab("🎨 Civitai"):
        with gr.Row():
            civitai_query = gr.Textbox(
                label="Search Query",
                placeholder="anime, realistic, fantasy..."
            )
            civitai_type = gr.Dropdown(
                choices=["Checkpoint", "LORA", "TextualInversion", "Hypernetwork"],
                value="LORA",
                label="Model Type"
            )
            base_model = gr.Dropdown(
                choices=["SDXL", "SD 1.5", "FLUX.1"],
                value="SDXL",
                label="Base Model"
            )
            civitai_search_btn = gr.Button("Search Civitai", variant="primary")
            
        civitai_results = gr.HTML()
        
        def search_civitai(query, model_type, base):
            # This would search Civitai
            # For now, return placeholder
            return f"""
            <div class="info-box">
                <h4>🎨 Civitai Search</h4>
                <p>Searching for "{query}" {model_type} models for {base}...</p>
                <p>Civitai integration with automatic downloads coming soon!</p>
            </div>
            """
            
        civitai_search_btn.click(
            search_civitai,
            inputs=[civitai_query, civitai_type, base_model],
            outputs=[civitai_results]
        )


def create_download_status_section(app):
    """Create download status section"""
    gr.Markdown("### 📊 Download Status")
    with gr.Row():
        with gr.Column():
            queue_status = gr.HTML(
                value="""
                <div class="stat-card">
                    <h4>Download Queue</h4>
                    <p>No active downloads</p>
                </div>
                """
            )
            
        with gr.Column():
            storage_status = gr.HTML(
                value=app.model_manager.get_storage_status()
            )
            
    # Refresh button
    refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
    
    def refresh_status():
        return app.model_manager.get_storage_status()
        
    refresh_btn.click(
        refresh_status,
        outputs=[storage_status]
    )


def format_search_results(results: List[Dict[str, Any]], source: str) -> str:
    """Format search results as HTML
    
    Args:
        results: List of search results
        source: Source of results (huggingface, civitai)
        
    Returns:
        HTML formatted results
    """
    if not results:
        return "<p>No results found</p>"
    
    html = f"<div class='search-results'>"
    html += f"<h4>Found {len(results)} results</h4>"
    
    for result in results[:10]:  # Limit to 10 results
        html += "<div class='search-result' style='border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 5px;'>"
        
        if source == "huggingface":
            html += f"<strong>{result.get('modelId', 'Unknown')}</strong><br>"
            html += f"<small>Author: {result.get('author', 'Unknown')}</small><br>"
            html += f"<small>Downloads: {result.get('downloads', 0):,}</small><br>"
            if result.get('description'):
                html += f"<p>{result['description'][:200]}...</p>"
            html += f"<button onclick='downloadModel(\"{result['modelId']}\")'>Download</button>"
            
        elif source == "civitai":
            html += f"<strong>{result.get('name', 'Unknown')}</strong><br>"
            html += f"<small>Type: {result.get('type', 'Unknown')}</small><br>"
            html += f"<small>Base Model: {result.get('baseModel', 'Unknown')}</small><br>"
            html += f"<small>Downloads: {result.get('stats', {}).get('downloadCount', 0):,}</small><br>"
            if result.get('description'):
                html += f"<p>{result['description'][:200]}...</p>"
            html += f"<button onclick='downloadCivitaiModel({result['id']})'>Download</button>"
            
        html += "</div>"
    
    html += "</div>"
    return html