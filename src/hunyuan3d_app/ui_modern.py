"""Modern UI components with bento grid layout"""

import gradio as gr
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from .credential_manager import CredentialManager
from .civitai_manager import CivitaiManager
from .lora_manager import LoRAManager
from .queue_manager import QueueManager
from .history_manager import HistoryManager


def load_modern_css() -> str:
    """Load modern CSS with bento grid layout"""
    return """
    /* Modern Bento Grid Layout */
    .bento-container {
        display: grid;
        gap: 1rem;
        padding: 1rem;
        background-color: var(--background-fill-primary);
    }
    
    /* Responsive grid layouts */
    @media (min-width: 768px) {
        .bento-grid-2 {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .bento-grid-3 {
            grid-template-columns: repeat(3, 1fr);
        }
        
        .bento-grid-4 {
            grid-template-columns: repeat(4, 1fr);
        }
    }
    
    /* Bento cards */
    .bento-card {
        background: var(--panel-background-fill);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid var(--border-color-primary);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .bento-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        border-color: var(--color-accent);
    }
    
    /* Card variants */
    .bento-card-primary {
        background: linear-gradient(135deg, var(--color-accent) 0%, var(--color-accent-soft) 100%);
        color: white;
    }
    
    .bento-card-feature {
        grid-column: span 2;
        min-height: 200px;
    }
    
    .bento-card-tall {
        grid-row: span 2;
    }
    
    /* Stats cards */
    .stat-card {
        text-align: center;
        padding: 2rem 1rem;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, var(--color-accent) 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: var(--body-text-color-subdued);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Progress indicators */
    .progress-ring {
        transform: rotate(-90deg);
        width: 120px;
        height: 120px;
        margin: 0 auto;
    }
    
    .progress-ring-circle {
        fill: none;
        stroke-width: 8;
    }
    
    .progress-ring-bg {
        stroke: var(--border-color-primary);
    }
    
    .progress-ring-progress {
        stroke: var(--color-accent);
        stroke-linecap: round;
        transition: stroke-dashoffset 0.5s ease;
    }
    
    /* Feature cards with icons */
    .feature-card {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .feature-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: var(--color-accent-soft);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        flex-shrink: 0;
    }
    
    .feature-content h3 {
        margin: 0 0 0.25rem 0;
        font-size: 1.125rem;
    }
    
    .feature-content p {
        margin: 0;
        color: var(--body-text-color-subdued);
        font-size: 0.875rem;
    }
    
    /* Model selector cards */
    .model-card {
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .model-card.selected {
        border-color: var(--color-accent);
        background: var(--color-accent-soft);
    }
    
    .model-card-header {
        display: flex;
        justify-content: space-between;
        align-items: start;
        margin-bottom: 0.75rem;
    }
    
    .model-badge {
        font-size: 0.75rem;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        background: var(--color-accent);
        color: white;
    }
    
    /* Quick action buttons */
    .quick-actions {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    
    .quick-action-btn {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: 1px solid var(--border-color-primary);
        background: var(--background-fill-primary);
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.875rem;
    }
    
    .quick-action-btn:hover {
        background: var(--color-accent);
        color: white;
        transform: translateY(-1px);
    }
    
    /* Animation classes */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: slideIn 0.5s ease forwards;
    }
    
    /* Loading skeleton */
    .skeleton {
        background: linear-gradient(90deg, 
            var(--border-color-primary) 25%, 
            var(--panel-background-fill) 50%, 
            var(--border-color-primary) 75%
        );
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Dark mode enhancements */
    .dark {
        --panel-background-fill: #1a1a1a;
        --border-color-primary: #333;
        --color-accent-soft: rgba(99, 102, 241, 0.1);
    }
    
    .dark .bento-card {
        background: #1a1a1a;
        border-color: #333;
    }
    
    .dark .bento-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.2);
    }
    """


class ModernUI:
    """Modern UI components and layouts"""
    
    @staticmethod
    def create_bento_grid(components: List[gr.components.Component], columns: int = 3) -> gr.components.Component:
        """Create a bento grid layout
        
        Args:
            components: List of components to arrange
            columns: Number of columns
            
        Returns:
            Grid container
        """
        with gr.HTML(f'<div class="bento-container bento-grid-{columns}">') as container:
            for component in components:
                component.render()
        gr.HTML('</div>')
        return container
    
    @staticmethod
    def create_stat_card(
        value: str,
        label: str,
        icon: str = "📊",
        color: Optional[str] = None
    ) -> gr.HTML:
        """Create a statistics card
        
        Args:
            value: Main value to display
            label: Label for the value
            icon: Emoji or icon
            color: Optional color override
            
        Returns:
            HTML component
        """
        color_style = f"color: {color};" if color else ""
        
        return gr.HTML(f"""
            <div class="bento-card stat-card animate-in">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div class="stat-value" style="{color_style}">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
        """)
    
    @staticmethod
    def create_feature_card(
        title: str,
        description: str,
        icon: str = "✨",
        action_label: Optional[str] = None,
        action_callback: Optional[Callable] = None
    ) -> gr.Group:
        """Create a feature card with icon
        
        Args:
            title: Card title
            description: Card description
            icon: Emoji or icon
            action_label: Optional action button label
            action_callback: Optional action callback
            
        Returns:
            Group component
        """
        with gr.Group(elem_classes="bento-card feature-card") as card:
            gr.HTML(f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-content">
                        <h3>{title}</h3>
                        <p>{description}</p>
                    </div>
                </div>
            """)
            
            if action_label and action_callback:
                gr.Button(
                    action_label,
                    size="sm",
                    elem_classes="quick-action-btn"
                ).click(action_callback)
                
        return card
    
    @staticmethod
    def create_model_selector_grid(
        models: List[Dict[str, Any]],
        selected_callback: Callable
    ) -> gr.components.Component:
        """Create a grid of model selector cards
        
        Args:
            models: List of model info dictionaries
            selected_callback: Callback when model is selected
            
        Returns:
            Grid component
        """
        with gr.HTML('<div class="bento-container bento-grid-3">') as container:
            for model in models:
                model_html = f"""
                    <div class="bento-card model-card" onclick="selectModel('{model['id']}')">
                        <div class="model-card-header">
                            <h4>{model['name']}</h4>
                            <span class="model-badge">{model['type']}</span>
                        </div>
                        <p style="font-size: 0.875rem; color: var(--body-text-color-subdued);">
                            {model['description']}
                        </p>
                        <div style="margin-top: 0.5rem;">
                            <span style="font-size: 0.75rem;">💾 {model['size']}</span>
                            <span style="font-size: 0.75rem; margin-left: 1rem;">🎯 {model['vram']}</span>
                        </div>
                    </div>
                """
                gr.HTML(model_html)
                
        gr.HTML('</div>')
        
        # Add selection script
        gr.HTML("""
            <script>
                function selectModel(modelId) {
                    // Remove previous selection
                    document.querySelectorAll('.model-card').forEach(card => {
                        card.classList.remove('selected');
                    });
                    
                    // Add selection to clicked card
                    event.currentTarget.classList.add('selected');
                    
                    // Trigger callback
                    // This would need proper Gradio event handling
                }
            </script>
        """)
        
        return container
    
    @staticmethod
    def create_progress_ring(
        progress: float,
        label: str,
        size: int = 120
    ) -> gr.HTML:
        """Create a circular progress indicator
        
        Args:
            progress: Progress value (0-1)
            label: Label to display
            size: Size in pixels
            
        Returns:
            HTML component
        """
        radius = (size - 16) / 2
        circumference = 2 * 3.14159 * radius
        offset = circumference - (progress * circumference)
        
        return gr.HTML(f"""
            <div class="bento-card" style="text-align: center; padding: 2rem;">
                <svg class="progress-ring" width="{size}" height="{size}">
                    <circle
                        class="progress-ring-circle progress-ring-bg"
                        cx="{size/2}"
                        cy="{size/2}"
                        r="{radius}"
                    />
                    <circle
                        class="progress-ring-circle progress-ring-progress"
                        cx="{size/2}"
                        cy="{size/2}"
                        r="{radius}"
                        style="stroke-dasharray: {circumference}; stroke-dashoffset: {offset};"
                    />
                </svg>
                <div style="margin-top: 1rem;">
                    <div class="stat-value">{int(progress * 100)}%</div>
                    <div class="stat-label">{label}</div>
                </div>
            </div>
        """)
    
    @staticmethod
    def create_quick_actions(
        actions: List[Tuple[str, str, Callable]]
    ) -> gr.Group:
        """Create a quick actions panel
        
        Args:
            actions: List of (label, icon, callback) tuples
            
        Returns:
            Group component
        """
        with gr.Group(elem_classes="bento-card") as panel:
            gr.Markdown("### Quick Actions")
            
            with gr.Row(elem_classes="quick-actions"):
                for label, icon, callback in actions:
                    gr.Button(
                        f"{icon} {label}",
                        elem_classes="quick-action-btn",
                        size="sm"
                    ).click(callback)
                    
        return panel
    
    @staticmethod
    def create_dashboard_layout(
        stats: Dict[str, Any],
        recent_generations: List[Dict[str, Any]],
        queue_status: Dict[str, Any]
    ) -> gr.components.Component:
        """Create a dashboard layout with bento grid
        
        Args:
            stats: Statistics data
            recent_generations: Recent generation data
            queue_status: Queue status data
            
        Returns:
            Dashboard component
        """
        with gr.HTML('<div class="bento-container bento-grid-4">') as dashboard:
            # Stats cards
            ModernUI.create_stat_card(
                str(stats.get('total_generations', 0)),
                "Total Generations",
                "🎨"
            )
            
            ModernUI.create_stat_card(
                str(stats.get('models_loaded', 0)),
                "Models Loaded",
                "🤖"
            )
            
            ModernUI.create_stat_card(
                f"{stats.get('vram_used', 0):.1f}GB",
                "VRAM Used",
                "💾"
            )
            
            ModernUI.create_stat_card(
                str(queue_status.get('pending', 0)),
                "Queue Pending",
                "⏳"
            )
            
            # Feature panel (spans 2 columns)
            with gr.HTML('<div class="bento-card bento-card-feature">'):
                gr.Markdown("### 🚀 Recent Activity")
                
                if recent_generations:
                    for gen in recent_generations[:5]:
                        gr.HTML(f"""
                            <div style="display: flex; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid var(--border-color-primary);">
                                <img src="{gen.get('thumbnail', '')}" style="width: 48px; height: 48px; border-radius: 8px; margin-right: 1rem;">
                                <div style="flex: 1;">
                                    <div style="font-weight: 500;">{gen.get('model', 'Unknown')}</div>
                                    <div style="font-size: 0.875rem; color: var(--body-text-color-subdued);">
                                        {gen.get('prompt', '')[:50]}...
                                    </div>
                                </div>
                                <div style="font-size: 0.75rem; color: var(--body-text-color-subdued);">
                                    {gen.get('time_ago', '')}
                                </div>
                            </div>
                        """)
                else:
                    gr.HTML('<p style="text-align: center; color: var(--body-text-color-subdued);">No recent generations</p>')
                    
            gr.HTML('</div>')
            
            # Progress indicators
            ModernUI.create_progress_ring(
                stats.get('gpu_usage', 0) / 100,
                "GPU Usage"
            )
            
            ModernUI.create_progress_ring(
                stats.get('memory_usage', 0) / 100,
                "Memory Usage"
            )
            
        gr.HTML('</div>')
        
        return dashboard
    
    @staticmethod
    def create_model_comparison_grid(
        models: List[Dict[str, Any]]
    ) -> gr.components.Component:
        """Create a model comparison grid
        
        Args:
            models: List of models to compare
            
        Returns:
            Comparison grid component
        """
        with gr.HTML('<div class="bento-container">') as comparison:
            # Header
            gr.Markdown("### 🔍 Model Comparison")
            
            # Comparison table
            headers = ["Model", "Type", "VRAM", "Speed", "Quality", "Actions"]
            
            table_html = """
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid var(--border-color-primary);">
            """
            
            for header in headers:
                table_html += f'<th style="padding: 1rem; text-align: left;">{header}</th>'
                
            table_html += """
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for model in models:
                table_html += f"""
                    <tr style="border-bottom: 1px solid var(--border-color-primary);">
                        <td style="padding: 1rem; font-weight: 500;">{model['name']}</td>
                        <td style="padding: 1rem;">{model['type']}</td>
                        <td style="padding: 1rem;">{model['vram']}</td>
                        <td style="padding: 1rem;">
                            <div style="display: flex; align-items: center;">
                                <div style="width: 100px; height: 8px; background: var(--border-color-primary); border-radius: 4px; margin-right: 0.5rem;">
                                    <div style="width: {model['speed']}%; height: 100%; background: var(--color-accent); border-radius: 4px;"></div>
                                </div>
                                {model['speed']}%
                            </div>
                        </td>
                        <td style="padding: 1rem;">
                            <div style="display: flex; align-items: center;">
                                <div style="width: 100px; height: 8px; background: var(--border-color-primary); border-radius: 4px; margin-right: 0.5rem;">
                                    <div style="width: {model['quality']}%; height: 100%; background: #10b981; border-radius: 4px;"></div>
                                </div>
                                {model['quality']}%
                            </div>
                        </td>
                        <td style="padding: 1rem;">
                            <button class="quick-action-btn" onclick="loadModel('{model['id']}')">Load</button>
                        </td>
                    </tr>
                """
                
            table_html += """
                    </tbody>
                </table>
            """
            
            gr.HTML(table_html)
            
        gr.HTML('</div>')
        
        return comparison
    
    @staticmethod
    def create_workflow_builder() -> gr.components.Component:
        """Create a visual workflow builder
        
        Returns:
            Workflow builder component
        """
        with gr.HTML('<div class="bento-container">') as builder:
            gr.Markdown("### 🔄 Workflow Builder")
            
            # This would be a more complex component in practice
            # For now, showing a placeholder
            gr.HTML("""
                <div class="bento-card" style="min-height: 400px; display: flex; align-items: center; justify-content: center;">
                    <div style="text-align: center;">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">🚧</div>
                        <h3>Visual Workflow Builder</h3>
                        <p style="color: var(--body-text-color-subdued);">
                            Drag and drop nodes to create custom generation workflows
                        </p>
                        <button class="quick-action-btn" style="margin-top: 1rem;">
                            Coming Soon
                        </button>
                    </div>
                </div>
            """)
            
        gr.HTML('</div>')
        
        return builder