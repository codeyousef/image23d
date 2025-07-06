"""Common UI components and utilities."""

import gradio as gr
from typing import List, Any, Tuple, Optional


def update_model_dropdowns_helper(choices: List[List[str]]) -> List[gr.update]:
    """Helper function to safely convert model choices to gr.update objects.
    
    Args:
        choices: List of choice lists for dropdowns
        
    Returns:
        List of gr.update objects
    """
    try:
        return [
            gr.update(choices=choices[0]),
            gr.update(choices=choices[1]),
            gr.update(choices=choices[2]),
            gr.update(choices=choices[3])
        ]
    except Exception as e:
        print(f"Error updating model dropdowns: {e}")
        return [
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update()
        ]


def create_generation_settings(show_advanced: bool = True) -> Tuple[Any, ...]:
    """Create common generation settings components.
    
    Args:
        show_advanced: Whether to show advanced settings
        
    Returns:
        Tuple of generation setting components
    """
    with gr.Accordion("Advanced Settings", open=not show_advanced):
        seed = gr.Slider(
            -1, 2147483647, -1,
            step=1,
            label="Seed (-1 for random)"
        )
        
        with gr.Row():
            steps = gr.Slider(
                20, 50, 30,
                step=5,
                label="Inference Steps",
                info="More steps = better quality but slower"
            )
            cfg = gr.Slider(
                1, 20, 7.5,
                step=0.5,
                label="Guidance Scale",
                info="Higher = follows prompt more closely"
            )
        
        with gr.Row():
            width = gr.Slider(
                512, 2048, 1024,
                step=64,
                label="Image Width"
            )
            height = gr.Slider(
                512, 2048, 1024,
                step=64,
                label="Image Height"
            )
    
    return seed, steps, cfg, width, height


def create_progress_display() -> Tuple[gr.Progress, gr.HTML]:
    """Create progress display components.
    
    Returns:
        Tuple of (progress bar, status HTML)
    """
    progress = gr.Progress()
    status = gr.HTML()
    
    return progress, status


def create_output_display(
    show_image: bool = True,
    show_3d: bool = True
) -> Tuple[Optional[gr.Image], Optional[gr.Model3D], gr.HTML]:
    """Create output display components.
    
    Args:
        show_image: Whether to show image output
        show_3d: Whether to show 3D output
        
    Returns:
        Tuple of output components
    """
    output_image = None
    output_3d = None
    
    if show_image:
        output_image = gr.Image(
            label="Generated Image",
            type="pil",
            elem_id="output-image"
        )
    
    if show_3d:
        output_3d = gr.Model3D(
            label="3D Model",
            elem_id="output-3d"
        )
    
    output_info = gr.HTML()
    
    return output_image, output_3d, output_info


def create_model_selector(
    label: str,
    choices: List[str],
    value: Optional[str] = None,
    info: Optional[str] = None
) -> gr.Dropdown:
    """Create a model selector dropdown.
    
    Args:
        label: Label for the dropdown
        choices: List of model choices
        value: Default value
        info: Additional info text
        
    Returns:
        Dropdown component
    """
    return gr.Dropdown(
        choices=choices,
        label=label,
        value=value or (choices[0] if choices else None),
        info=info,
        interactive=True
    )


def create_action_button(
    label: str,
    variant: str = "primary",
    size: str = "lg",
    icon: Optional[str] = None
) -> gr.Button:
    """Create an action button.
    
    Args:
        label: Button label
        variant: Button variant (primary, secondary, stop)
        size: Button size (sm, lg)
        icon: Optional icon
        
    Returns:
        Button component
    """
    return gr.Button(
        value=label,
        variant=variant,
        size=size,
        icon=icon
    )