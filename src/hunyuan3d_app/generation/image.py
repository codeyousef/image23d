import logging
import threading
import queue
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Optional, Any, Dict, Callable

import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from ..utils.performance import get_performance_monitor, profile_generation
from ..utils.memory import get_memory_manager
from ..models.gguf import GGUFModelManager, GGUFModelInfo
from ..models.flux_production import (
    ProductionFluxPipeline,
    GenerationRequest,
    GenerationResult,
    create_production_pipeline
)

logger = logging.getLogger(__name__)

# Global queue for tracking active generation tasks
generation_queue = queue.Queue()
# Lock for thread-safe operations
generation_lock = threading.Lock()

class ImageGenerator:
    def __init__(self, device: str, output_dir: Path):
        self.device = device
        self.output_dir = output_dir
        self.background_remover = None
        self.stop_generation_flag = False
        self.gguf_manager = GGUFModelManager(cache_dir="image_models/gguf")
        self.flux_pipeline = None  # Will be created when needed

    def stop_generation(self):
        """Stop the current generation process"""
        self.stop_generation_flag = True
        return "Generation stopping... Please wait for current step to complete."

    def reset_stop_flag(self):
        """Reset the stop generation flag"""
        self.stop_generation_flag = False

    def remove_background(self, image):
        """Remove background from image"""
        try:
            if not self.background_remover:
                self.background_remover = pipeline("image-segmentation",
                                                   image_model="briaai/RMBG-1.4",
                                                   trust_remote_code=True,
                                                   device=self.device)

            # Remove background
            result = self.background_remover(image)
            
            # The result should be an image with transparent background
            # If it's a dict with 'mask', we need to apply the mask
            if isinstance(result, dict) and 'mask' in result:
                from PIL import Image
                import numpy as np
                
                # Convert to RGBA
                if image.mode != 'RGBA':
                    image = image.convert('RGBA')
                
                # Apply mask
                mask = result['mask']
                if isinstance(mask, Image.Image):
                    mask = np.array(mask)
                
                # Create alpha channel from mask
                img_array = np.array(image)
                img_array[:, :, 3] = mask
                
                return Image.fromarray(img_array, 'RGBA')
            elif isinstance(result, list) and len(result) > 0:
                # Handle list of results
                if isinstance(result[0], dict) and 'mask' in result[0]:
                    mask = result[0]['mask']
                    # Apply mask as above
                    from PIL import Image
                    import numpy as np
                    
                    if image.mode != 'RGBA':
                        image = image.convert('RGBA')
                    
                    if isinstance(mask, Image.Image):
                        mask = np.array(mask)
                    
                    img_array = np.array(image)
                    img_array[:, :, 3] = mask
                    
                    return Image.fromarray(img_array, 'RGBA')
                else:
                    # Return first result if it's an image
                    return result[0]
            else:
                # Return the result as-is
                return result

        except Exception as e:
            logger.error(f"Error removing background: {str(e)}")
            logger.warning("Returning original image without background removal")
            return image

    @profile_generation
    def generate_image(
            self,
            image_model,
            image_model_name,
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            guidance_scale,
            seed,
            progress
    ):
        """Generate an image from text prompt using a background thread"""
        
        # Check if image_model is loaded
        if image_model is None:
            logger.error("No image model loaded")
            return None, "❌ No image model loaded"
            
        logger.info(f"[IMAGE_GENERATOR] Starting generation with model: {type(image_model).__name__}")
        logger.info(f"[IMAGE_GENERATOR] Model name: {image_model_name}")
        logger.info(f"[IMAGE_GENERATOR] Is GGUF: {getattr(image_model, '_is_gguf_model', False)}")

        # Reset stop flag at the beginning of generation
        self.reset_stop_flag()

        # Create result container
        result_container = {"image": None, "info": "", "error": None, "completed": False, "current_step": 0, "stopped": False}

        # Set seed - use int32 max value to avoid overflow
        if seed == -1:
            # Use 2**31 - 1 as the upper bound to avoid int32 overflow in numpy
            seed = np.random.randint(0, 2147483647)

        # Check model type for callback
        image_model_type = type(image_model).__name__
        is_flux_pipeline = "FluxPipeline" in image_model_type
        is_gguf_wrapped = hasattr(image_model, '_is_gguf_model') and image_model._is_gguf_model
        
        # Define callback function for real-time progress updates
        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            # Log callback invocation
            logger.debug(f"Progress callback: step {step_index}/{steps}, timestep: {timestep}")
            
            # Update the current step in the result container
            result_container["current_step"] = step_index
            
            # Call the external progress callback if provided
            if progress:
                progress_value = (step_index + 1) / steps
                progress_msg = f"Step {step_index + 1}/{steps}"
                
                # Add memory info for FLUX models periodically
                if (is_flux_pipeline or is_gguf_wrapped) and (step_index + 1) % 5 == 0:
                    try:
                        from ..utils.memory_optimization import get_memory_manager
                        memory_manager = get_memory_manager()
                        free_vram = memory_manager.get_available_memory()
                        progress_msg += f" (VRAM: {free_vram:.1f}GB free)"
                    except:
                        pass  # Don't fail on memory check
                
                progress(progress_value, progress_msg)

            # Check if generation should be stopped
            if self.stop_generation_flag:
                result_container["stopped"] = True
                # Raise a StopIteration exception to halt the generation process
                raise StopIteration("Generation stopped by user")

            return callback_kwargs

        # Define the generation function that will run in a background thread
        def generate_in_background():
            start_time = time.time()
            try:
                # Don't need to capture - just use the closure variables directly
                
                # Check if this is a GGUF model
                if hasattr(image_model, '_is_gguf_model') and image_model._is_gguf_model:
                    logger.info(f"Using GGUF model for generation: {image_model_name}")
                    logger.info(f"GGUF model type: {type(image_model).__name__}")
                    logger.info(f"Has _real_pipeline: {hasattr(image_model, '_real_pipeline') and image_model._real_pipeline is not None}")
                    logger.info(f"Is placeholder: {getattr(image_model, '_is_placeholder', 'unknown')}")
                    # GGUF models are already properly configured through the pipeline
                    # Q8 models are much smaller and don't need special handling
                
                # Acquire lock to ensure thread safety
                with generation_lock:
                    # Create generator with seed
                    # Check if the image_model has a specific device or is using device_map
                    generator = None
                    use_generator = True
                    
                    # Check if image_model uses device_map (for multi-device image_models)
                    has_device_map = hasattr(image_model, 'hf_device_map') and image_model.hf_device_map
                    image_model_type = type(image_model).__name__
                    is_flux_image_model = "Flux" in image_model_type or "flux" in image_model_name.lower()
                    
                    # For FLUX image_models or image_models with device_map, don't use generator
                    if is_flux_image_model or has_device_map:
                        logger.info(f"Model type '{image_model_type}' with device_map={has_device_map}, skipping generator creation")
                        use_generator = False
                        generator = None
                        
                        # For reproducibility, set global seed instead
                        if seed != -1:
                            torch.manual_seed(seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(seed)
                            logger.info(f"Set global torch seed to {seed} for FLUX/device_map image_model")
                    else:
                        try:
                            # For single-device image_models, create generator on the image_model's device
                            if hasattr(image_model, 'device') and image_model.device != torch.device("meta"):
                                generator_device = image_model.device
                            else:
                                generator_device = self.device
                            
                            generator = torch.Generator(generator_device).manual_seed(seed)
                            logger.info(f"Created generator on device: {generator_device}")
                            
                        except Exception as gen_error:
                            logger.warning(f"Error creating generator: {str(gen_error)}")
                            # Skip generator for safety
                            use_generator = False
                            generator = None

                    # Free up memory before generation
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Use a context manager to ensure proper cleanup
                    with torch.no_grad():
                        try:
                            # Generate the image with callback for progress updates
                            # Check if the image_model is a FluxPipeline or FLUX model
                            image_model_type = type(image_model).__name__
                            is_flux_pipeline = "FluxPipeline" in image_model_type
                            is_flux_model = (is_flux_pipeline or 
                                           "flux" in image_model_name.lower() or 
                                           "FLUX" in image_model_name or
                                           (hasattr(image_model, '_is_gguf_model') and 
                                            image_model._is_gguf_model and 
                                            "flux" in str(getattr(image_model, 'model_name', '')).lower()))
                            logger.info(f"Model type: {image_model_type}, is FLUX: {is_flux_model}")

                            # Check if we already have a loaded GGUF model - don't create new pipeline
                            is_loaded_gguf = (hasattr(image_model, '_is_gguf_model') and 
                                            image_model._is_gguf_model and
                                            hasattr(image_model, '_real_pipeline') and 
                                            image_model._real_pipeline is not None)
                            
                            logger.info(f"GGUF check - is_loaded_gguf: {is_loaded_gguf}")
                            if is_loaded_gguf:
                                logger.info("Skipping production pipeline - using loaded GGUF model directly")
                            
                            if is_flux_model and not is_loaded_gguf:
                                # Use our new production FLUX pipeline only if we don't have a loaded GGUF
                                logger.info("Using production FLUX pipeline for generation")
                                
                                # Create production pipeline if not exists
                                if self.flux_pipeline is None:
                                    logger.info("Creating production FLUX pipeline...")
                                    # Determine variant from model name
                                    if "q8" in image_model_name.lower():
                                        variant = "gguf_q8"
                                    elif "q6" in image_model_name.lower():
                                        variant = "gguf_q6"
                                    elif "q4" in image_model_name.lower():
                                        variant = "gguf_q4"
                                    else:
                                        variant = "auto"
                                    
                                    self.flux_pipeline = create_production_pipeline(model_variant=variant)
                                    logger.info(f"Production pipeline created with variant: {variant}")
                                
                                # Create generation request
                                request = GenerationRequest(
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    width=width,
                                    height=height,
                                    num_images=1,
                                    seed=seed if seed != -1 else None,
                                    style="photorealistic",  # Could be extracted from prompt
                                    quality="high",
                                    use_acceleration=True,
                                    compile_model=False,  # Skip compilation for now
                                    enhance_output=False,  # Skip enhancement for speed
                                    model_variant=variant
                                )
                                
                                # Define a custom progress callback that works with the production pipeline
                                def prod_progress_callback(current, total):
                                    result_container["current_step"] = current
                                    if progress:
                                        progress_value = (current + 1) / total
                                        progress(progress_value, f"Step {current + 1}/{total}")
                                    
                                    if self.stop_generation_flag:
                                        result_container["stopped"] = True
                                        raise StopIteration("Generation stopped by user")
                                
                                # Override the pipeline's internal progress callback
                                # Note: This is a simplified approach - in production you'd integrate
                                # the callback more deeply into the pipeline
                                original_steps = steps
                                
                                # Generate with production pipeline
                                gen_result = self.flux_pipeline.generate(request)
                                
                                if gen_result.success and gen_result.images:
                                    image = gen_result.images[0]
                                    logger.info(f"FLUX generation successful in {gen_result.timing.get('total', 0):.2f}s")
                                    # Update progress to match expected steps
                                    result_container["current_step"] = original_steps
                                else:
                                    raise RuntimeError(f"FLUX generation failed: {gen_result.error}")
                                    
                            elif is_flux_pipeline:
                                # FluxPipeline should never use generator to avoid device conflicts
                                logger.info("Using FluxPipeline without generator")
                                
                                # Import memory manager for FLUX-specific optimizations
                                from ..utils.memory_optimization import get_memory_manager
                                memory_manager = get_memory_manager()
                                
                                # Pre-generation memory check
                                memory_manager.aggressive_memory_clear()
                                memory_manager.monitor_memory_usage("flux-pre-generation")
                                
                                # Check if image_model has device_map (components split across devices)
                                if hasattr(image_model, 'hf_device_map') and image_model.hf_device_map:
                                    logger.info(f"FLUX image_model has device_map: {image_model.hf_device_map}")
                                    
                                    # For device_map models, the VAE is on CPU while other components are on CUDA
                                    # We need to temporarily move the VAE to CUDA for generation
                                    moved_vae = False
                                    try:
                                        # Check if VAE is on CPU
                                        if hasattr(image_model, 'vae'):
                                            # Get the actual parameter device, not the module device
                                            vae_device = next(image_model.vae.parameters()).device
                                            logger.info(f"VAE is on device: {vae_device}")
                                            if vae_device.type == 'cpu':
                                                logger.info("Moving VAE from CPU to CUDA for generation")
                                                image_model.vae = image_model.vae.to('cuda')
                                                moved_vae = True
                                        
                                        # Try generation with detailed logging
                                        logger.info(f"Starting FLUX generation with device_map")
                                        logger.info(f"Prompt: {prompt[:50]}...")
                                        logger.info(f"Steps: {steps}, Guidance: {guidance_scale}")
                                        
                                        result = image_model(
                                            prompt=prompt,
                                            negative_prompt=negative_prompt,
                                            width=width,
                                            height=height,
                                            num_inference_steps=steps,
                                            guidance_scale=guidance_scale,
                                            callback_on_step_end=progress_callback
                                        )
                                        
                                        logger.info("FLUX generation completed")
                                        image = result.images[0]
                                        
                                    except Exception as e:
                                        logger.error(f"FLUX generation failed: {e}")
                                        error_msg = str(e)
                                        logger.error(f"FULL FLUX ERROR: {error_msg}")
                                        
                                        # Check if it's a CUDA OOM error
                                        if "CUDA out of memory" in error_msg:
                                            memory_manager.monitor_memory_usage("flux-oom-error")
                                            logger.error("CUDA OOM detected - attempting memory recovery")
                                            memory_manager.aggressive_memory_clear()
                                            
                                            # Add helpful message
                                            error_msg += "\n\nMemory optimization suggestions:\n"
                                            error_msg += "- Try reducing image resolution\n"
                                            error_msg += "- Use a smaller quantization (Q5, Q4)\n"
                                            error_msg += "- Close other GPU applications\n"
                                        
                                        # No gradient fallback - re-raise the error
                                        raise RuntimeError(f"FLUX generation failed: {error_msg}")
                                    finally:
                                        # Restore VAE to CPU if we moved it
                                        if moved_vae and hasattr(image_model, 'vae'):
                                            logger.info("Restoring VAE to CPU")
                                            try:
                                                image_model.vae = image_model.vae.to('cpu')
                                            except Exception as e:
                                                logger.warning(f"Could not restore VAE to CPU: {e}")
                                else:
                                    # Standard FLUX without device_map
                                    result = image_model(
                                        prompt=prompt,
                                        negative_prompt=negative_prompt,
                                        width=width,
                                        height=height,
                                        num_inference_steps=steps,
                                        guidance_scale=guidance_scale,
                                        callback_on_step_end=progress_callback
                                    )
                                    image = result.images[0]
                            else:
                                # For other pipelines, check if they support callback_steps
                                # GGUF image_models wrapped in StandaloneGGUFPipeline use FluxPipeline internally
                                is_gguf_wrapped = hasattr(image_model, '_is_gguf_model') and image_model._is_gguf_model
                                
                                if is_gguf_wrapped:
                                    logger.info("Using GGUF pipeline parameters (no callback_steps)")
                                    # Log detailed info about the GGUF model
                                    logger.info(f"GGUF model info:")
                                    logger.info(f"  - Has device_map: {getattr(image_model, '_has_device_map', False)}")
                                    logger.info(f"  - Real pipeline: {hasattr(image_model, '_real_pipeline') and image_model._real_pipeline is not None}")
                                    logger.info(f"  - Is placeholder: {getattr(image_model, '_is_placeholder', False)}")
                                    
                                    # Memory optimization for GGUF models
                                    from ..utils.memory_optimization import get_memory_manager
                                    memory_manager = get_memory_manager()
                                    memory_manager.aggressive_memory_clear()
                                    logger.info("Cleared memory for GGUF generation")
                                    # GGUF models may use device_map internally
                                    try:
                                        logger.info("Calling GGUF model for generation...")
                                        if use_generator and generator is not None:
                                            result = image_model(
                                                prompt=prompt,
                                                negative_prompt=negative_prompt,
                                                width=width,
                                                height=height,
                                                num_inference_steps=steps,
                                                guidance_scale=guidance_scale,
                                                generator=generator,
                                                seed=seed,  # Pass seed for StandaloneGGUFPipeline
                                                callback_on_step_end=progress_callback
                                            )
                                        else:
                                            result = image_model(
                                                prompt=prompt,
                                                negative_prompt=negative_prompt,
                                                width=width,
                                                height=height,
                                                num_inference_steps=steps,
                                                guidance_scale=guidance_scale,
                                                seed=seed,  # Pass seed for StandaloneGGUFPipeline
                                                callback_on_step_end=progress_callback
                                            )
                                        logger.info("GGUF model call completed")
                                    except Exception as gguf_error:
                                        logger.error(f"GGUF model generation failed: {gguf_error}")
                                        logger.error(f"GGUF error type: {type(gguf_error).__name__}")
                                        raise
                                else:
                                    logger.info("Using standard pipeline parameters (with callback_steps)")
                                    if use_generator and generator is not None:
                                        result = image_model(
                                            prompt=prompt,
                                            negative_prompt=negative_prompt,
                                            width=width,
                                            height=height,
                                            num_inference_steps=steps,
                                            guidance_scale=guidance_scale,
                                            generator=generator,
                                            callback_on_step_end=progress_callback,
                                            callback_steps=1  # Update on every step
                                        )
                                    else:
                                        # No generator, use global seed
                                        if seed != -1:
                                            torch.manual_seed(seed)
                                            if torch.cuda.is_available():
                                                torch.cuda.manual_seed_all(seed)
                                        result = image_model(
                                            prompt=prompt,
                                            negative_prompt=negative_prompt,
                                            width=width,
                                            height=height,
                                            num_inference_steps=steps,
                                            guidance_scale=guidance_scale,
                                            callback_on_step_end=progress_callback,
                                            callback_steps=1  # Update on every step
                                        )
                                image = result.images[0]
                        except StopIteration as e:
                            # Handle user-initiated stop
                            logger.info(f"Generation stopped by user at step {result_container['current_step']}/{steps}")
                            result_container["info"] = f"""
<div class="warning-box">
    <h4>⚠️ Generation Stopped</h4>
    <p>Image generation was stopped at step {result_container['current_step']}/{steps}.</p>
</div>
"""
                            return
                        except RuntimeError as e:
                            # Re-raise runtime errors - device mismatches should be fixed now
                            logger.error(f"Runtime error during generation: {str(e)}")
                            raise

                    # Clean up memory after generation
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Save image
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = self.output_dir / f"generated_{timestamp}.png"
                    image.save(image_path)

                    # Get performance stats
                    perf_monitor = get_performance_monitor()
                    gpu_stats = perf_monitor.get_gpu_stats()
                    
                    # Calculate generation time
                    generation_time = time.time() - start_time
                    
                    # Store results
                    result_container["image"] = image
                    result_container["info"] = f"""
<div class="info-box">
    <h4>✅ Image Generated!</h4>
    <ul>
        <li><strong>Model:</strong> {image_model_name}</li>
        <li><strong>Resolution:</strong> {width}x{height}</li>
        <li><strong>Seed:</strong> {seed}</li>
        <li><strong>Time:</strong> {generation_time:.1f}s ({steps/generation_time:.2f} it/s)</li>
        <li><strong>Saved to:</strong> {image_path.name}</li>
    </ul>
"""
                    
                    # Add GPU stats if available
                    if gpu_stats and "pytorch" in gpu_stats:
                        pytorch_stats = gpu_stats["pytorch"]
                        result_container["info"] += f"""
    <ul>
        <li><strong>VRAM Used:</strong> {pytorch_stats['allocated_gb']:.1f}/{pytorch_stats['total_gb']:.1f}GB</li>
    </ul>
"""
                    
                    result_container["info"] += "</div>"
            except Exception as e:
                import traceback
                error_msg = str(e) if str(e) else "Unknown error occurred"
                logger.error(f"Error generating image: {error_msg}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                result_container["error"] = f"{type(e).__name__}: {error_msg}"
            finally:
                # Mark as completed
                result_container["completed"] = True
                # Remove from queue
                try:
                    generation_queue.get_nowait()
                    generation_queue.task_done()
                except queue.Empty:
                    pass

        try:
            # Add to queue and start thread
            generation_queue.put(1)
            logger.info(f"Creating thread with image_model type: {type(image_model).__name__ if image_model else 'None'}")
            thread = threading.Thread(target=generate_in_background)
            thread.daemon = True  # Allow the thread to be terminated when the main program exits
            thread.start()

            # Show progress while waiting for the thread to complete
            progress(0.1, "Preparing for image generation...")

            # Wait for the thread to complete with detailed progress updates
            last_step = 0
            same_step_count = 0
            # Check if this is a GGUF model for timeout adjustment
            is_gguf_model = hasattr(image_model, '_is_gguf_model') and image_model._is_gguf_model
            while not result_container["completed"]:
                # Check if user requested to stop generation
                if self.stop_generation_flag and not result_container["stopped"]:
                    progress(min(0.1 + (0.85 * (result_container["current_step"] / steps)), 0.95), 
                            f"Stopping generation at step {result_container['current_step']}/{steps}...")
                    # Mark as stopped in the result container
                    result_container["stopped"] = True
                    # Continue waiting for the thread to complete

                current_step = result_container["current_step"]

                # Only update UI when step changes for efficiency
                if current_step > last_step or current_step == 0:
                    # Calculate progress percentage based on current step
                    if steps > 0:
                        # Progress from 10% to 95% based on actual generation steps
                        progress_value = 0.1 + (0.85 * (current_step / steps))
                    else:
                        # Fallback if steps is 0
                        progress_value = 0.5

                    # Create detailed progress message
                    if current_step == 0:
                        progress_msg = "Initializing generation pipeline..."
                    else:
                        progress_msg = f"Generating image: Step {current_step}/{steps} ({(current_step/steps*100):.1f}%)"

                        # Add more details for better user feedback
                        if current_step < steps * 0.3:
                            progress_msg += " - Initial noise reduction"
                        elif current_step < steps * 0.6:
                            progress_msg += " - Forming basic shapes"
                        elif current_step < steps * 0.9:
                            progress_msg += " - Refining details"
                        else:
                            progress_msg += " - Final enhancements"

                    # Update the progress bar
                    progress(min(progress_value, 0.95), progress_msg)
                    last_step = current_step
                    same_step_count = 0  # Reset counter when step changes
                else:
                    # If step hasn't changed, increment counter
                    same_step_count += 1

                # Small sleep to prevent UI freezing
                time.sleep(0.1)

                # Timeout after 5 minutes of no progress for GGUF models (they're slow!)
                # GGUF Q8 models can take 3-4 minutes per step on consumer GPUs
                timeout_iterations = 3000 if is_gguf_model else 300  # 5 min vs 30 sec
                if last_step > 0 and same_step_count > timeout_iterations:
                    raise TimeoutError(f"Image generation appears to be stuck (no progress for {same_step_count * 0.1:.0f} seconds)")

            # Update progress based on completion status
            if result_container["stopped"]:
                progress(1.0, "Image generation stopped!")
            else:
                progress(1.0, "Image generation complete!")

            # Check for errors
            if result_container["error"]:
                # Add memory info to error messages
                if "CUDA out of memory" in str(result_container["error"]):
                    from ..utils.memory_optimization import get_memory_manager
                    memory_manager = get_memory_manager()
                    memory_info = f"\n\nMemory Status:\n"
                    memory_info += f"- Free VRAM: {memory_manager.get_available_memory():.1f} GB\n"
                    memory_info += f"- Total VRAM: {memory_manager.vram_gb:.1f} GB\n"
                    memory_info += "\nTry using a smaller model or reducing resolution."
                    result_container["error"] = str(result_container["error"]) + memory_info
                raise Exception(result_container["error"])

            # Return the results - if stopped and no image was generated, return a message
            if result_container["stopped"] and not result_container["image"]:
                return None, result_container["info"] or f"""
<div class="warning-box">
    <h4>⚠️ Generation Stopped</h4>
    <p>Image generation was stopped before an image could be created.</p>
</div>
"""

            # Return the results
            return result_container["image"], result_container["info"]

        except Exception as e:
            logger.error(f"Error in image generation process: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, f"❌ Error: {str(e)}"
