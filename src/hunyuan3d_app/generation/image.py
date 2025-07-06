import logging
import threading
import queue
import time
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
        self.gguf_manager = GGUFModelManager(cache_dir="models/gguf")

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
                                                   model="briaai/RMBG-1.4",
                                                   trust_remote_code=True,
                                                   device=self.device)

            # Remove background
            result = self.background_remover([image])
            return result[0]['mask']

        except Exception as e:
            logger.error(f"Error removing background: {str(e)}")
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

        # Reset stop flag at the beginning of generation
        self.reset_stop_flag()

        # Create result container
        result_container = {"image": None, "info": "", "error": None, "completed": False, "current_step": 0, "stopped": False}

        # Set seed - use int32 max value to avoid overflow
        if seed == -1:
            # Use 2**31 - 1 as the upper bound to avoid int32 overflow in numpy
            seed = np.random.randint(0, 2147483647)

        # Define callback function for real-time progress updates
        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            # Update the current step in the result container
            result_container["current_step"] = step_index
            
            # Call the external progress callback if provided
            if progress:
                progress_value = (step_index + 1) / steps
                progress(progress_value, f"Step {step_index + 1}/{steps}")

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
                # Check if this is a GGUF model
                if hasattr(image_model, '_is_gguf_model') and image_model._is_gguf_model:
                    logger.info(f"Using GGUF model for generation: {image_model_name}")
                    # GGUF models are already properly configured through the pipeline
                
                # Acquire lock to ensure thread safety
                with generation_lock:
                    # Create generator with seed
                    # Check if the model has a specific device or is using device_map
                    try:
                        if hasattr(image_model, 'device') and image_model.device != torch.device("meta"):
                            # Use the model's device for the generator
                            generator_device = image_model.device
                        else:
                            # Fall back to the default device
                            generator_device = self.device

                        # For FluxPipeline models, check if we can use GPU generator
                        model_type = type(image_model).__name__
                        if "FluxPipeline" in model_type:
                            # Try to use GPU generator if model is on GPU
                            if generator_device == "cuda":
                                logger.info("FluxPipeline detected, attempting to use GPU generator")
                            else:
                                logger.info("FluxPipeline detected, using CPU generator")
                                generator_device = "cpu"

                        generator = torch.Generator(generator_device).manual_seed(seed)
                        logger.info(f"Created generator on device: {generator_device}")
                    except Exception as gen_error:
                        logger.warning(f"Error creating generator with specific device: {str(gen_error)}")
                        # Fall back to default generator without specifying device
                        generator = torch.Generator().manual_seed(seed)
                        logger.info("Created generator with default device")

                    # Free up memory before generation
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Use a context manager to ensure proper cleanup
                    with torch.no_grad():
                        try:
                            # Generate the image with callback for progress updates
                            # Check if the model is a FluxPipeline or mock pipeline
                            model_type = type(image_model).__name__
                            is_flux_pipeline = "FluxPipeline" in model_type
                            is_mock_pipeline = "Mock" in model_type
                            logger.info(f"Model type: {model_type}, is FluxPipeline: {is_flux_pipeline}")

                            if is_mock_pipeline:
                                # Handle mock pipelines differently
                                logger.info("Using mock pipeline - generating test image")
                                result = image_model(
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    width=width,
                                    height=height,
                                    num_inference_steps=steps,
                                    guidance_scale=guidance_scale,
                                    seed=seed
                                )
                                # Check if it has images attribute or is direct image
                                if hasattr(result, 'images'):
                                    image = result.images[0]
                                else:
                                    image = result  # Direct image return
                            elif is_flux_pipeline:
                                # FluxPipeline doesn't accept callback_steps parameter
                                logger.info("Using FluxPipeline-specific parameters (without callback_steps)")
                                result = image_model(
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    width=width,
                                    height=height,
                                    num_inference_steps=steps,
                                    guidance_scale=guidance_scale,
                                    generator=generator,
                                    callback_on_step_end=progress_callback
                                )
                                image = result.images[0]
                            else:
                                # For other pipelines, check if they support callback_steps
                                # GGUF models wrapped in StandaloneGGUFPipeline use FluxPipeline internally
                                is_gguf_wrapped = hasattr(image_model, '_is_gguf_model') and image_model._is_gguf_model
                                
                                if is_gguf_wrapped:
                                    logger.info("Using GGUF pipeline parameters (no callback_steps)")
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
                                    logger.info("Using standard pipeline parameters (with callback_steps)")
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
                            # Check if this is a device mismatch error
                            if ("device" in str(e).lower() and "meta" in str(e).lower()) or "tensor on device meta is not on the expected device" in str(e).lower():
                                logger.error(f"Device mismatch error: {str(e)}")
                                # Try again without specifying a generator (let the model handle device internally)
                                logger.info("Retrying without explicit generator device...")
                                try:
                                    # Check if the model is a FluxPipeline
                                    model_type = type(image_model).__name__
                                    is_flux_pipeline = "FluxPipeline" in model_type
                                    logger.info(f"Retry - Model type: {model_type}, is FluxPipeline: {is_flux_pipeline}")

                                    if is_flux_pipeline:
                                        # FluxPipeline doesn't accept callback_steps parameter
                                        logger.info("Retry - Using FluxPipeline-specific parameters (without callback_steps)")
                                        # Create a CPU generator for FluxPipeline to avoid device mismatch
                                        cpu_generator = torch.Generator("cpu").manual_seed(seed)
                                        logger.info("Created CPU generator for FluxPipeline retry")
                                        try:
                                            image = image_model(
                                                prompt=prompt,
                                                negative_prompt=negative_prompt,
                                                width=width,
                                                height=height,
                                                num_inference_steps=steps,
                                                guidance_scale=guidance_scale,
                                                generator=cpu_generator,
                                                seed=seed,  # Pass seed for StandaloneGGUFPipeline
                                                callback_on_step_end=progress_callback
                                            ).images[0]
                                        except RuntimeError as e2:
                                            # If we still get a device mismatch error, try one final approach
                                            if "tensor on device meta is not on the expected device" in str(e2).lower():
                                                logger.error(f"Still getting device mismatch in retry: {str(e2)}")
                                                logger.info("Final retry attempt - using CPU device for all operations")

                                                # Keep model on GPU instead of moving to CPU
                                                logger.warning("Device mismatch detected, but keeping model on GPU for performance")

                                                # Final attempt without any generator
                                                image = image_model(
                                                    prompt=prompt,
                                                    negative_prompt=negative_prompt,
                                                    width=width,
                                                    height=height,
                                                    num_inference_steps=steps,
                                                    guidance_scale=guidance_scale,
                                                    seed=seed,  # Pass seed for StandaloneGGUFPipeline
                                                    # No generator at all for final attempt
                                                    callback_on_step_end=progress_callback
                                                ).images[0]
                                            else:
                                                # Re-raise if it's not a device mismatch error
                                                raise
                                    else:
                                        # For other pipelines, check if GGUF
                                        is_gguf_wrapped = hasattr(image_model, '_is_gguf_model') and image_model._is_gguf_model
                                        logger.info(f"Retry - Using {'GGUF' if is_gguf_wrapped else 'standard'} pipeline parameters")
                                        try:
                                            # Try with a CPU generator first
                                            cpu_generator = torch.Generator("cpu").manual_seed(seed)
                                            logger.info("Created CPU generator for pipeline retry")
                                            
                                            if is_gguf_wrapped:
                                                image = image_model(
                                                    prompt=prompt,
                                                    negative_prompt=negative_prompt,
                                                    width=width,
                                                    height=height,
                                                    num_inference_steps=steps,
                                                    guidance_scale=guidance_scale,
                                                    generator=cpu_generator,
                                                    seed=seed,  # Pass seed for StandaloneGGUFPipeline
                                                    callback_on_step_end=progress_callback
                                                ).images[0]
                                            else:
                                                image = image_model(
                                                    prompt=prompt,
                                                    negative_prompt=negative_prompt,
                                                    width=width,
                                                    height=height,
                                                    num_inference_steps=steps,
                                                    guidance_scale=guidance_scale,
                                                    generator=cpu_generator,
                                                    callback_on_step_end=progress_callback,
                                                    callback_steps=1  # Update on every step
                                                ).images[0]
                                        except RuntimeError as e2:
                                            # If we still get a device mismatch error, try one final approach
                                            if "tensor on device meta is not on the expected device" in str(e2).lower():
                                                logger.error(f"Still getting device mismatch in retry: {str(e2)}")
                                                logger.info("Final retry attempt - using CPU device for all operations")

                                                # Keep model on GPU instead of moving to CPU
                                                logger.warning("Device mismatch detected, but keeping model on GPU for performance")

                                                # Final attempt without any generator
                                                if is_gguf_wrapped:
                                                    image = image_model(
                                                        prompt=prompt,
                                                        negative_prompt=negative_prompt,
                                                        width=width,
                                                        height=height,
                                                        num_inference_steps=steps,
                                                        guidance_scale=guidance_scale,
                                                        seed=seed,  # Pass seed for StandaloneGGUFPipeline
                                                        # No generator at all for final attempt
                                                        callback_on_step_end=progress_callback
                                                    ).images[0]
                                                else:
                                                    image = image_model(
                                                        prompt=prompt,
                                                        negative_prompt=negative_prompt,
                                                        width=width,
                                                        height=height,
                                                        num_inference_steps=steps,
                                                        guidance_scale=guidance_scale,
                                                        # No generator at all for final attempt
                                                        callback_on_step_end=progress_callback,
                                                        callback_steps=1  # Update on every step
                                                    ).images[0]
                                            else:
                                                # Re-raise if it's not a device mismatch error
                                                raise
                                except StopIteration as e:
                                    # Handle user-initiated stop in the retry
                                    logger.info(f"Generation stopped by user at step {result_container['current_step']}/{steps}")
                                    result_container["info"] = f"""
<div class="warning-box">
    <h4>⚠️ Generation Stopped</h4>
    <p>Image generation was stopped at step {result_container['current_step']}/{steps}.</p>
</div>
"""
                                    return
                            else:
                                # Re-raise other runtime errors
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
                logger.error(f"Error generating image: {str(e)}")
                result_container["error"] = str(e)
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
            thread = threading.Thread(target=generate_in_background)
            thread.daemon = True  # Allow the thread to be terminated when the main program exits
            thread.start()

            # Show progress while waiting for the thread to complete
            progress(0.1, "Preparing for image generation...")

            # Wait for the thread to complete with detailed progress updates
            last_step = 0
            same_step_count = 0
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

                # Timeout after 30 seconds of no progress (300 iterations at 0.1s each)
                if last_step > 0 and same_step_count > 300:
                    raise TimeoutError("Image generation appears to be stuck")

            # Update progress based on completion status
            if result_container["stopped"]:
                progress(1.0, "Image generation stopped!")
            else:
                progress(1.0, "Image generation complete!")

            # Check for errors
            if result_container["error"]:
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
            return None, f"❌ Error: {str(e)}"
