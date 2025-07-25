"""Job Processors Mixin

Handles processing of various job types (image, 3D, video, face swap, etc.).
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

from ...config import OUTPUT_DIR
from ...features.face_swap import FaceSwapParams

logger = logging.getLogger(__name__)


class JobProcessorMixin:
    """Mixin for processing different job types"""
    
    def _process_image_job(self, params: Dict[str, Any], progress_callback):
        """Process an image generation job"""
        # Extract parameters
        model_name = params.get("model_name")
        prompt = params.get("prompt")
        negative_prompt = params.get("negative_prompt", "")
        width = params.get("width", 1024)
        height = params.get("height", 1024)
        steps = params.get("steps", 30)
        guidance_scale = params.get("guidance_scale", 7.5)
        seed = params.get("seed", -1)
        lora_configs = params.get("lora_configs", [])
        job_id = params.get("job_id", "unknown")
        
        # Enhanced progress callback that also sends to WebSocket
        def enhanced_progress(progress_value, message):
            logger.info(f"Enhanced progress called: {progress_value:.2f} - {message}")
            # Call the queue progress callback with the raw progress value
            progress_callback(progress_value, message)
            # Send to WebSocket progress manager
            try:
                self.progress_manager.send_progress_update(
                    task_id=job_id,
                    progress=progress_value,
                    message=message,
                    task_type="image_generation"
                )
            except Exception as e:
                logger.warning(f"Failed to send WebSocket progress: {e}")
        
        # Load model if needed
        try:
            if self.image_model_name != model_name:
                enhanced_progress(0.1, f"Loading model {model_name}...")
                status, model, model_name_loaded = self.model_manager.load_image_model(
                    model_name, self.image_model, self.image_model_name, "cuda", progress=enhanced_progress
                )
                if "❌" in status:
                    raise Exception(f"Failed to load model: {status}")
                self.image_model = model
                self.image_model_name = model_name_loaded
                enhanced_progress(0.3, "Model loaded successfully")
            else:
                enhanced_progress(0.3, "Using cached model")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            enhanced_progress(1.0, f"Error: {str(e)}")
            return {"image": None, "info": f"❌ Model loading failed: {str(e)}", "path": None, "image_path": None, "error": str(e)}
                
        # Apply LoRAs if specified
        if lora_configs and self.image_model:
            enhanced_progress(0.2, "Applying LoRAs...")
            self.lora_manager.apply_multiple_loras(self.image_model, lora_configs)
            
        # Generate image
        enhanced_progress(0.3, "Generating image...")
        try:
            image, info = self.image_generator.generate_image(
                self.image_model,
                self.image_model_name,
                prompt,
                negative_prompt,
                width,
                height,
                steps,
                guidance_scale,
                seed,
                progress=lambda p, msg: enhanced_progress(0.3 + p * 0.6, msg)
            )
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            enhanced_progress(1.0, f"Error: {str(e)}")
            return {"image": None, "info": f"❌ Image generation failed: {str(e)}", "path": None, "image_path": None, "error": str(e)}
        
        if image:
            # Save to history
            progress_callback(0.9, "Saving to history...")
            generation_id = str(uuid.uuid4())
            
            # Save image
            image_path = OUTPUT_DIR / f"image_{generation_id}.png"
            image.save(image_path)
            
            # Add to history
            self.history_manager.add_generation(
                generation_id=generation_id,
                generation_type="image",
                model_name=model_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                parameters={
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "loras": [{"name": lora.name, "weight": weight} for lora, weight in lora_configs]
                },
                output_paths=[str(image_path)],
                metadata={"info": info}
            )
        else:
            # No image generated
            logger.warning("No image was generated")
            return {"image": None, "info": info or "❌ No image generated", "path": None, "image_path": None}
        
        progress_callback(1.0, "Complete!")
        return {"image": image, "info": info, "path": str(image_path) if image else None, "image_path": str(image_path) if image else None}
    
    def _process_3d_job(self, params: Dict[str, Any], progress_callback):
        """Process a 3D generation job"""
        # Extract parameters
        model_name = params.get("model_name", "hunyuan3d-21")
        prompt = params.get("prompt")
        input_image_path = params.get("input_image")
        negative_prompt = params.get("negative_prompt", "")
        quality_preset = params.get("quality_preset", "Standard")
        export_format = params.get("export_format", "glb")
        job_id = params.get("job_id", "unknown")
        
        # Enhanced progress callback with WebSocket
        def enhanced_progress(progress_value, message):
            logger.info(f"3D progress: {progress_value:.2f} - {message}")
            progress_callback(progress_value, message)
            try:
                self.progress_manager.send_progress_update(
                    task_id=job_id,
                    progress=progress_value,
                    message=message,
                    task_type="3d_generation"
                )
            except Exception as e:
                logger.warning(f"Failed to send WebSocket progress: {e}")
        
        # Load model if needed
        enhanced_progress(0.1, f"Loading 3D model {model_name}...")
        try:
            if self.threed_model_name != model_name:
                status, model, model_name_loaded = self.model_manager.load_3d_model(
                    model_name, self.threed_model, self.threed_model_name, "cuda", progress=enhanced_progress
                )
                if "❌" in status:
                    raise Exception(f"Failed to load model: {status}")
                self.threed_model = model
                self.threed_model_name = model_name_loaded
                enhanced_progress(0.2, "3D model loaded successfully")
            else:
                enhanced_progress(0.2, "Using cached 3D model")
        except Exception as e:
            logger.error(f"3D model loading failed: {e}")
            enhanced_progress(1.0, f"Error: {str(e)}")
            return {"mesh_path": None, "info": f"❌ Model loading failed: {str(e)}", "error": str(e)}
        
        # Load input image if provided
        input_image = None
        if input_image_path:
            try:
                if isinstance(input_image_path, str):
                    input_image = Image.open(input_image_path)
                else:
                    input_image = input_image_path
            except Exception as e:
                logger.error(f"Failed to load input image: {e}")
                
        # Generate 3D
        enhanced_progress(0.3, "Generating 3D model...")
        generation_id = str(uuid.uuid4())
        output_dir = OUTPUT_DIR / "3d" / generation_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mesh_path, info = self.threed_generator.generate_3d(
            self.threed_model,
            model_name,
            prompt,
            negative_prompt,
            input_image,
            quality_preset,
            export_format,
            str(output_dir),
            progress=lambda p, msg: enhanced_progress(0.3 + p * 0.6, msg)
        )
        
        if mesh_path and Path(mesh_path).exists():
            # Save to history
            enhanced_progress(0.9, "Saving to history...")
            
            # Get all output files in the directory
            output_files = list(output_dir.glob("*"))
            output_paths = [str(f) for f in output_files]
            
            self.history_manager.add_generation(
                generation_id=generation_id,
                generation_type="3d",
                model_name=model_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                parameters={
                    "quality_preset": quality_preset,
                    "export_format": export_format,
                    "has_input_image": input_image is not None
                },
                output_paths=output_paths,
                metadata={"info": info}
            )
            
            enhanced_progress(1.0, "Complete!")
            return {"mesh_path": mesh_path, "info": info, "output_dir": str(output_dir)}
        else:
            enhanced_progress(1.0, "Failed to generate 3D model")
            return {"mesh_path": None, "info": info or "❌ Failed to generate 3D model", "error": "Generation failed"}
    
    def _process_full_pipeline_job(self, params: Dict[str, Any], progress_callback):
        """Process a full pipeline job (image generation + 3D conversion)"""
        # Stage 1: Generate image
        progress_callback(0.0, "Stage 1: Generating image...")
        
        image_params = {
            "model_name": params.get("image_model_name"),
            "prompt": params.get("prompt"),
            "negative_prompt": params.get("negative_prompt", ""),
            "width": params.get("width", 1024),
            "height": params.get("height", 1024),
            "steps": params.get("steps", 30),
            "guidance_scale": params.get("guidance_scale", 7.5),
            "seed": params.get("seed", -1),
            "lora_configs": params.get("lora_configs", []),
            "job_id": params.get("job_id", "unknown") + "_image"
        }
        
        # Process image generation with scaled progress (0-50%)
        image_result = self._process_image_job(
            image_params,
            lambda p, msg: progress_callback(p * 0.5, f"Image: {msg}")
        )
        
        if not image_result.get("image"):
            return {
                "image": None,
                "mesh_path": None,
                "info": f"Image generation failed: {image_result.get('info', 'Unknown error')}",
                "error": image_result.get('error', 'Image generation failed')
            }
        
        # Stage 2: Generate 3D from image
        progress_callback(0.5, "Stage 2: Converting to 3D...")
        
        threed_params = {
            "model_name": params.get("threed_model_name", "hunyuan3d-21"),
            "prompt": params.get("prompt"),
            "input_image": image_result["image"],
            "negative_prompt": params.get("negative_prompt", ""),
            "quality_preset": params.get("quality_preset", "Standard"),
            "export_format": params.get("export_format", "glb"),
            "job_id": params.get("job_id", "unknown") + "_3d"
        }
        
        # Process 3D generation with scaled progress (50-100%)
        threed_result = self._process_3d_job(
            threed_params,
            lambda p, msg: progress_callback(0.5 + p * 0.5, f"3D: {msg}")
        )
        
        # Combine results
        return {
            "image": image_result.get("image"),
            "image_path": image_result.get("path"),
            "mesh_path": threed_result.get("mesh_path"),
            "output_dir": threed_result.get("output_dir"),
            "info": f"Image: {image_result.get('info', 'OK')}\n3D: {threed_result.get('info', 'OK')}"
        }
    
    def _process_video_job(self, params: Dict[str, Any], progress_callback):
        """Process a video generation job"""
        # Extract parameters
        model_name = params.get("model_name", "AnimateDiff-Lightning")
        prompt = params.get("prompt")
        negative_prompt = params.get("negative_prompt", "")
        num_frames = params.get("num_frames", 16)
        fps = params.get("fps", 8)
        width = params.get("width", 512)
        height = params.get("height", 512)
        guidance_scale = params.get("guidance_scale", 7.5)
        num_inference_steps = params.get("num_inference_steps", 25)
        
        # Generate video
        progress_callback(0.1, "Loading video model...")
        
        try:
            video_path, info = self.video_generator.generate_video(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_name=model_name,
                num_frames=num_frames,
                fps=fps,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                progress_callback=lambda p, msg: progress_callback(0.1 + p * 0.8, msg)
            )
            
            if video_path:
                # Save to history
                progress_callback(0.9, "Saving to history...")
                generation_id = str(uuid.uuid4())
                
                self.history_manager.add_generation(
                    generation_id=generation_id,
                    generation_type="video",
                    model_name=model_name,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    parameters={
                        "num_frames": num_frames,
                        "fps": fps,
                        "width": width,
                        "height": height,
                        "guidance_scale": guidance_scale,
                        "num_inference_steps": num_inference_steps
                    },
                    output_paths=[video_path],
                    metadata={"info": info}
                )
                
                progress_callback(1.0, "Complete!")
                return {"video_path": video_path, "info": info}
            else:
                return {"video_path": None, "info": info or "❌ Failed to generate video"}
                
        except Exception as e:
            logger.error(f"Video generation error: {e}")
            return {"video_path": None, "info": f"❌ Error: {str(e)}"}
    
    def _process_face_swap_job(self, params: Dict[str, Any], progress_callback):
        """Process a face swap job"""
        # Extract parameters
        source_image = params.get("source_image")
        target_image = params.get("target_image")
        face_restore = params.get("face_restore", True)
        blend_ratio = params.get("blend_ratio", 0.95)
        face_index = params.get("face_index", 0)
        similarity_threshold = params.get("similarity_threshold", 0.6)
        
        # Initialize models if needed
        progress_callback(0.1, "Initializing face swap models...")
        
        if not self.face_swap_manager.facefusion_loaded:
            initialized, msg = self.face_swap_manager.initialize_models()
            if not initialized:
                return {"path": None, "info": f"❌ Failed to initialize: {msg}"}
        
        # Create parameters
        progress_callback(0.3, "Processing face swap...")
        
        swap_params = FaceSwapParams(
            swap_mode="single",
            face_selector_mode="many",
            face_analyser_order="best",
            face_restore=face_restore,
            face_restore_fidelity=0.5,
            blend_ratio=blend_ratio,
            similarity_threshold=similarity_threshold,
            selected_face_index=face_index
        )
        
        # Perform swap
        start_time = time.time()
        result_image, swap_info = self.face_swap_manager.swap_face(
            source_image=source_image,
            target_image=target_image,
            params=swap_params,
            progress_callback=lambda p, msg: progress_callback(0.3 + p * 0.6, msg)
        )
        
        processing_time = time.time() - start_time
        
        if result_image:
            # Save result
            progress_callback(0.9, "Saving result...")
            generation_id = str(uuid.uuid4())
            output_path = OUTPUT_DIR / f"face_swap_{generation_id}.png"
            result_image.save(output_path)
            
            # Save to history
            self.history_manager.add_generation(
                generation_id=generation_id,
                generation_type="face_swap",
                model_name="FaceFusion",
                prompt="Face Swap",
                negative_prompt="",
                parameters={
                    "face_restore": face_restore,
                    "blend_ratio": blend_ratio,
                    "face_index": face_index,
                    "similarity_threshold": similarity_threshold
                },
                output_paths=[str(output_path)],
                metadata={
                    "info": swap_info,
                    "processing_time": processing_time
                }
            )
            
            progress_callback(1.0, "Complete!")
            
            # Parse swap info for additional details
            info_dict = {}
            if isinstance(swap_info, str):
                # Extract numbers from info string
                import re
                source_match = re.search(r"Source faces detected: (\d+)", swap_info)
                target_match = re.search(r"Target faces detected: (\d+)", swap_info)
                swapped_match = re.search(r"Faces swapped: (\d+)", swap_info)
                
                if source_match:
                    info_dict['source_faces'] = int(source_match.group(1))
                if target_match:
                    info_dict['target_faces'] = int(target_match.group(1))
                if swapped_match:
                    info_dict['swapped_faces'] = int(swapped_match.group(1))
            
            info_dict['processing_time'] = processing_time
            
            return {
                "path": str(output_path),
                "info": info_dict,
                "processing_time": processing_time
            }
        else:
            return {
                "path": None,
                "info": swap_info or "❌ Face swap failed"
            }