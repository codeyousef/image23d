"""GGUF model optimization utilities"""

import logging
import torch

logger = logging.getLogger(__name__)


class GGUFOptimizer:
    """Handles optimization for GGUF models"""
    
    @staticmethod
    def apply_q6_q8_optimizations(pipeline):
        """Apply special optimizations for Q6/Q8 GGUF models"""
        try:
            # Import optimization utilities
            from ...utils.memory_optimization import get_memory_manager
            memory_manager = get_memory_manager()
            
            logger.info("Applying Q6/Q8 optimizations...")
            
            # 1. Enable sequential CPU offload for memory efficiency
            # For Q6/Q8, we can't use enable_model_cpu_offload due to GGUF constraints
            # Instead, we carefully manage memory
            
            # 2. Optimize attention for Q6/Q8
            if hasattr(pipeline, 'enable_attention_slicing'):
                # Use smaller slice size for Q6/Q8 to reduce memory usage
                pipeline.enable_attention_slicing(slice_size=2)
                logger.info("Enabled attention slicing with size=2")
            
            # 3. Enable VAE slicing for memory efficiency
            if hasattr(pipeline, 'enable_vae_slicing'):
                pipeline.enable_vae_slicing()
                logger.info("Enabled VAE slicing")
            
            # 4. Ensure components are on the right device
            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                # Keep VAE on CPU to save GPU memory for Q6/Q8
                pipeline.vae = pipeline.vae.to('cpu')
                logger.info("Moved VAE to CPU for memory efficiency")
            
            # 5. Optimize text encoders
            if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
                pipeline.text_encoder = pipeline.text_encoder.to('cpu')
                logger.info("Moved CLIP text encoder to CPU")
                
            if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
                # T5 is large, keep on CPU
                pipeline.text_encoder_2 = pipeline.text_encoder_2.to('cpu')
                logger.info("Moved T5 text encoder to CPU")
            
            # 6. Clear memory after optimizations
            memory_manager.aggressive_memory_clear()
            
            logger.info("Q6/Q8 optimizations applied successfully")
            
        except Exception as e:
            logger.warning(f"Failed to apply some Q6/Q8 optimizations: {e}")
            # Continue anyway, some optimizations are better than none
    
    @staticmethod
    def enable_memory_optimizations(pipeline, device: str = "cuda"):
        """Enable various memory optimizations for the pipeline"""
        try:
            # Enable attention slicing
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
                logger.info("Enabled attention slicing")
                
            # Enable VAE slicing
            if hasattr(pipeline, 'enable_vae_slicing'):
                pipeline.enable_vae_slicing()
                logger.info("Enabled VAE slicing")
                
            # Enable xformers if available
            if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to apply some memory optimizations: {e}")