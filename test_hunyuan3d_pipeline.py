#!/usr/bin/env python3
"""
Standalone test script for HunYuan3D pipeline.
Run this to test the entire 3D generation process without starting the NiceGUI app.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import gc

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import hunyuan3d_app first to trigger global torch.load security patch
import hunyuan3d_app

from hunyuan3d_app.models.threed.orchestrator import ThreeDOrchestrator
from hunyuan3d_app.models.threed.hunyuan3d.config import HunYuan3DConfig
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    memory_gb = process.memory_info().rss / 1024**3
    
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
        return memory_gb, gpu_memory_gb
    return memory_gb, 0


def create_test_image(prompt=None):
    """Create a test image or use an existing one."""
    # Check if we have example images
    example_path = Path("Hunyuan3D/assets/example_images/004.png")
    if example_path.exists():
        logger.info(f"Using example image: {example_path}")
        return Image.open(example_path)
    
    # Create a simple test image
    logger.info("Creating test image...")
    # Create a simple gradient test image
    img = Image.new('RGB', (512, 512))
    pixels = img.load()
    for i in range(512):
        for j in range(512):
            pixels[i, j] = (int(i/2), int(j/2), 128)
    return img


def test_pipeline(args):
    """Test the HunYuan3D pipeline."""
    logger.info(f"Testing HunYuan3D pipeline with model: {args.model}")
    logger.info(f"Device: {args.device}")
    
    # Track timing
    start_time = time.time()
    stage_times = {}
    
    # Initial memory
    cpu_mem, gpu_mem = get_memory_usage()
    logger.info(f"Initial memory - CPU: {cpu_mem:.2f}GB, GPU: {gpu_mem:.2f}GB")
    
    try:
        # Create test image
        stage_start = time.time()
        if args.image_path:
            test_image = Image.open(args.image_path)
            logger.info(f"Loaded image from: {args.image_path}")
        else:
            test_image = create_test_image(args.prompt)
        stage_times['image_creation'] = time.time() - stage_start
        
        # Save test image
        test_image_path = Path("test_input.png")
        test_image.save(test_image_path)
        logger.info(f"Saved test image to: {test_image_path}")
        
        # Initialize orchestrator
        stage_start = time.time()
        logger.info("Initializing 3D orchestrator...")
        config = HunYuan3DConfig(
            model_variant=args.model,
            device=args.device,
            enable_texture=not args.skip_texture
        )
        
        orchestrator = ThreeDOrchestrator(config)
        stage_times['orchestrator_init'] = time.time() - stage_start
        
        cpu_mem, gpu_mem = get_memory_usage()
        logger.info(f"After init - CPU: {cpu_mem:.2f}GB, GPU: {gpu_mem:.2f}GB")
        
        cpu_mem, gpu_mem = get_memory_usage()
        logger.info(f"After orchestrator init - CPU: {cpu_mem:.2f}GB, GPU: {gpu_mem:.2f}GB")
        
        # Generate 3D
        stage_start = time.time()
        logger.info("Starting 3D generation...")
        
        def progress_callback(step, total, message=""):
            if args.verbose:
                logger.info(f"Progress: {step}/{total} - {message}")
        
        result = orchestrator.generate(
            input_data=test_image,
            progress_callback=progress_callback
        )
        stage_times['generation'] = time.time() - stage_start
        
        cpu_mem, gpu_mem = get_memory_usage()
        logger.info(f"After generation - CPU: {cpu_mem:.2f}GB, GPU: {gpu_mem:.2f}GB")
        
        # Check results
        if result and result.get("success"):
            logger.info("✅ 3D generation successful!")
            logger.info(f"Output saved to: {result.get('output_path', 'Unknown')}")
            
            # Print mesh info if available
            if 'mesh_info' in result:
                info = result['mesh_info']
                logger.info(f"Mesh stats - Vertices: {info.get('vertices', 0)}, Faces: {info.get('faces', 0)}")
        else:
            logger.error("❌ 3D generation failed!")
            if result:
                logger.error(f"Error: {result.get('error', 'Unknown error')}")
            raise RuntimeError("Generation failed")
        
        # Cleanup
        if args.cleanup:
            logger.info("Cleaning up...")
            del orchestrator
            gc.collect()
            torch.cuda.empty_cache()
            
            cpu_mem, gpu_mem = get_memory_usage()
            logger.info(f"After cleanup - CPU: {cpu_mem:.2f}GB, GPU: {gpu_mem:.2f}GB")
        
        # Summary
        total_time = time.time() - start_time
        logger.info("\n" + "="*50)
        logger.info("SUMMARY")
        logger.info("="*50)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info("\nStage times:")
        for stage, duration in stage_times.items():
            logger.info(f"  {stage}: {duration:.2f}s")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        logger.exception("Full traceback:")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test HunYuan3D pipeline")
    parser.add_argument(
        "--model", 
        type=str, 
        default="hunyuan3d-21",
        choices=["hunyuan3d-21", "hunyuan3d-2mini", "hunyuan3d-2mv", "hunyuan3d-2standard"],
        help="Model variant to test"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cute cartoon character",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        help="Path to input image (optional)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--skip-texture",
        action="store_true",
        help="Skip texture generation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=True,
        help="Clean up models after test"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print system info
    logger.info("System Information:")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    logger.info("")
    
    # Run test
    success = test_pipeline(args)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()