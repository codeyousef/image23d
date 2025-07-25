"""Enhanced pytest configuration for comprehensive testing of NeuralForge Studio.

This configuration provides fixtures for testing:
- Desktop app (NiceGUI) with local generation
- Backend API (FastAPI) with cloud generation
- Web app (React) client
"""

import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import tempfile
from pathlib import Path
import psutil
import json
import asyncio
import time
from typing import Dict, Any, List, Optional
import threading
from contextlib import contextmanager

# Try to import optional dependencies
try:
    from nicegui import ui
    from nicegui.testing import User
    NICEGUI_AVAILABLE = True
except ImportError:
    NICEGUI_AVAILABLE = False
    User = None

try:
    import GPUtil
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

# =============================================================================
# Core Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_root():
    """Root directory for all test files."""
    with tempfile.TemporaryDirectory(prefix="neuralforge_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_dir(test_root):
    """Temporary directory for individual tests."""
    test_dir = test_root / f"test_{time.time_ns()}"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    # Cleanup handled by test_root


@pytest.fixture
def mock_models_dir(temp_dir):
    """Mock models directory with structure."""
    models_dir = temp_dir / "models"
    
    # Create full directory structure
    subdirs = [
        "image", "3d", "video", "loras", "gguf", 
        "facefusion", "texture_components", "cache"
    ]
    for subdir in subdirs:
        (models_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return models_dir


@pytest.fixture
def mock_output_dir(temp_dir):
    """Mock output directory with structure."""
    output_dir = temp_dir / "outputs"
    
    # Create output subdirectories
    subdirs = ["images", "3d", "videos", "temp", "thumbnails"]
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return output_dir


# =============================================================================
# Model and Pipeline Fixtures
# =============================================================================

@pytest.fixture
def mock_local_models():
    """Mock local model loading and inference for desktop app."""
    with patch('diffusers.DiffusionPipeline.from_pretrained') as mock_pipeline:
        # Create a comprehensive mock pipeline
        mock_pipe = MagicMock()
        mock_pipe.device = torch.device("cpu")
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.enable_attention_slicing = MagicMock()
        mock_pipe.enable_model_cpu_offload = MagicMock()
        mock_pipe.enable_xformers_memory_efficient_attention = MagicMock()
        
        # Mock generation
        def mock_generate(**kwargs):
            result = MagicMock()
            result.images = [Image.new('RGB', (512, 512), color='blue')]
            return result
        
        mock_pipe.__call__ = MagicMock(side_effect=mock_generate)
        mock_pipeline.return_value = mock_pipe
        
        yield mock_pipeline


@pytest.fixture
def mock_gpu_environment():
    """Mock GPU environment for testing."""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1), \
         patch('torch.cuda.get_device_properties') as mock_props, \
         patch('torch.cuda.mem_get_info', return_value=(4*1024**3, 8*1024**3)), \
         patch('torch.cuda.empty_cache'):
        
        mock_device = MagicMock()
        mock_device.total_memory = 8 * 1024**3  # 8GB
        mock_device.name = "NVIDIA GeForce RTX 3080"
        mock_device.major = 8
        mock_device.minor = 6
        mock_props.return_value = mock_device
        
        if GPU_UTILS_AVAILABLE:
            with patch('GPUtil.getGPUs') as mock_gpus:
                mock_gpu = MagicMock()
                mock_gpu.memoryTotal = 8192
                mock_gpu.memoryUsed = 2048
                mock_gpu.memoryFree = 6144
                mock_gpu.name = "NVIDIA GeForce RTX 3080"
                mock_gpus.return_value = [mock_gpu]
                yield
        else:
            yield


@pytest.fixture
def mock_hunyuan3d_model():
    """Mock HunYuan3D model for 3D generation."""
    model = MagicMock()
    
    def mock_generate_3d(image, **kwargs):
        # Create mock mesh data
        vertices = np.random.rand(1000, 3).astype(np.float32)
        faces = np.random.randint(0, 1000, size=(2000, 3)).astype(np.int32)
        
        mesh = MagicMock()
        mesh.vertices = vertices
        mesh.faces = faces
        mesh.export = MagicMock()
        
        # Mock texture
        texture = Image.new('RGB', (1024, 1024), color='red')
        
        # Mock multiview images
        multiview_images = [
            Image.new('RGB', (256, 256), color='blue') 
            for _ in range(6)
        ]
        
        return {
            "mesh": mesh,
            "texture": texture,
            "multiview_images": multiview_images,
            "vertices": vertices,
            "faces": faces
        }
    
    model.generate = MagicMock(side_effect=mock_generate_3d)
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    
    return model


@pytest.fixture
def mock_video_model():
    """Mock video generation model."""
    model = MagicMock()
    
    def mock_generate_video(image=None, prompt=None, **kwargs):
        # Create mock video frames
        num_frames = kwargs.get('num_frames', 30)
        frames = []
        for i in range(num_frames):
            frame = Image.new('RGB', (512, 512), 
                            color=(i*8 % 255, 0, 255-i*8 % 255))
            frames.append(frame)
        
        return {
            "frames": frames,
            "fps": kwargs.get('fps', 24),
            "duration": num_frames / kwargs.get('fps', 24)
        }
    
    model.generate = MagicMock(side_effect=mock_generate_video)
    return model


# =============================================================================
# Desktop App (NiceGUI) Fixtures
# =============================================================================

@pytest.fixture
def nicegui_user():
    """Create NiceGUI test user."""
    if not NICEGUI_AVAILABLE:
        pytest.skip("NiceGUI not available")
    return User()


@pytest.fixture
def mock_local_model_manager(mock_models_dir):
    """Mock local model manager for desktop app."""
    from unittest.mock import MagicMock
    
    manager = MagicMock()
    manager.models_dir = mock_models_dir
    manager.current_model_name = None
    manager.current_model = None
    
    # Available models based on VRAM
    manager.detect_available_vram = MagicMock(return_value=8.0)
    manager.get_recommended_models = MagicMock(return_value=[
        "SDXL-Turbo", "LCM-LoRA", "Wan2.1-1.3B", "HunYuan3D-Mini"
    ])
    
    # Model loading
    def mock_load_model(model_name, progress_callback=None):
        if progress_callback:
            for i in range(5):
                progress_callback(f"Loading {model_name}", i/5)
        manager.current_model_name = model_name
        manager.current_model = MagicMock()
        return manager.current_model
    
    manager.load_model = MagicMock(side_effect=mock_load_model)
    manager.unload_model = MagicMock()
    manager.get_model_info = MagicMock(return_value={
        "name": "Test Model",
        "size_gb": 4.5,
        "vram_required_gb": 6.0
    })
    
    return manager


@pytest.fixture
async def desktop_app(mock_models_dir, mock_output_dir, mock_local_model_manager):
    """Create desktop app instance for testing."""
    # Mock the desktop app
    app = MagicMock()
    app.model_manager = mock_local_model_manager
    app.output_dir = mock_output_dir
    app.models_dir = mock_models_dir
    
    # Mock workflow execution
    async def mock_generate_workflow(workflow_type, params, progress_callback=None):
        if progress_callback:
            progress_callback("model_loading", 0.2, "Loading models...")
            await asyncio.sleep(0.01)
            progress_callback("generation", 0.6, "Generating...")
            await asyncio.sleep(0.01)
            progress_callback("post_processing", 0.9, "Post-processing...")
            await asyncio.sleep(0.01)
        
        # Return mock results based on workflow type
        if workflow_type == "text_to_3d":
            return {
                "status": "completed",
                "outputs": {
                    "image": Image.new('RGB', (512, 512)),
                    "mesh": str(mock_output_dir / "mesh.glb"),
                    "texture": str(mock_output_dir / "texture.png")
                },
                "metadata": {
                    "generation_time": 15.5,
                    "model_used": params.get("threed_model", "HunYuan3D-Mini")
                }
            }
        elif workflow_type == "image_to_video":
            return {
                "status": "completed",
                "outputs": {
                    "video_path": str(mock_output_dir / "output.mp4")
                },
                "metadata": {
                    "generation_time": 8.3,
                    "model_used": params.get("video_model", "LTX-Video")
                }
            }
        else:
            return {
                "status": "completed",
                "outputs": {"image": Image.new('RGB', (512, 512))},
                "metadata": {"generation_time": 2.5}
            }
    
    app.generate_workflow = AsyncMock(side_effect=mock_generate_workflow)
    
    # Mock batch generation
    async def mock_batch_generate(workflow_type, items, shared_params):
        results = []
        for i, item in enumerate(items):
            result = await mock_generate_workflow(
                workflow_type, 
                {**shared_params, "prompt": item}
            )
            result["metadata"]["prompt"] = item
            results.append(result)
        return results
    
    app.batch_generate = AsyncMock(side_effect=mock_batch_generate)
    
    # Mock initialization
    app.initialize = AsyncMock()
    await app.initialize()
    
    return app


# =============================================================================
# Backend API (FastAPI) Fixtures
# =============================================================================

@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    from backend.main import app
    
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create authenticated headers for API requests."""
    return {"Authorization": "Bearer test-token-12345"}


@pytest.fixture
def mock_job_service():
    """Mock job service for API testing."""
    service = MagicMock()
    
    # Mock job creation
    def create_job(job_type, params, user_id):
        return {
            "job_id": f"job-{time.time_ns()}",
            "type": job_type,
            "status": "queued",
            "params": params,
            "user_id": user_id,
            "created_at": time.time()
        }
    
    service.create_job = MagicMock(side_effect=create_job)
    service.get_job_status = MagicMock(return_value={
        "job_id": "test-job-123",
        "status": "processing",
        "progress": 0.5,
        "message": "Generating..."
    })
    service.cancel_job = MagicMock(return_value=True)
    
    return service


@pytest.fixture
def mock_websocket_manager():
    """Mock WebSocket manager for API testing."""
    manager = MagicMock()
    manager.connect = AsyncMock()
    manager.disconnect = AsyncMock()
    manager.send_progress = AsyncMock()
    manager.broadcast = AsyncMock()
    
    return manager


# =============================================================================
# Web App (React) Fixtures
# =============================================================================

@pytest.fixture
def mock_api_responses():
    """Mock API responses for React testing."""
    return {
        "models": {
            "models": [
                {"name": "FLUX.1-dev", "type": "image", "status": "available"},
                {"name": "SDXL-Turbo", "type": "image", "status": "available"},
                {"name": "HunYuan3D-2.1", "type": "3d", "status": "available"},
                {"name": "LTX-Video", "type": "video", "status": "available"},
                {"name": "Wan2.1-1.3B", "type": "video", "status": "available"}
            ]
        },
        "job_create": {
            "job_id": "job-12345",
            "status": "queued",
            "queue_position": 1
        },
        "job_status": {
            "job_id": "job-12345",
            "status": "completed",
            "progress": 1.0,
            "result": {
                "output_url": "https://api.example.com/outputs/result.png"
            }
        }
    }


@pytest.fixture
def mock_websocket_client():
    """Mock WebSocket client for React testing."""
    client = MagicMock()
    client.connect = MagicMock()
    client.disconnect = MagicMock()
    client.on = MagicMock()
    client.send = MagicMock()
    
    return client


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_images(temp_dir):
    """Create sample test images."""
    images = {}
    
    # Create various test images
    sizes = [(512, 512), (1024, 1024), (256, 256)]
    names = ["test_portrait", "test_landscape", "test_object"]
    
    for name, size in zip(names, sizes):
        img = Image.new('RGB', size)
        pixels = img.load()
        # Create gradient
        for i in range(size[0]):
            for j in range(size[1]):
                pixels[i, j] = (i % 256, j % 256, (i+j) % 256)
        
        path = temp_dir / f"{name}.png"
        img.save(path)
        images[name] = {"image": img, "path": path}
    
    return images


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return {
        "simple": [
            "a red cube",
            "a blue sphere",
            "a green pyramid"
        ],
        "complex": [
            "a futuristic robot with glowing eyes and metallic armor",
            "an ancient temple covered in vines with golden artifacts",
            "a magical crystal floating above a mystical fountain"
        ],
        "character": [
            "a brave knight in shining armor",
            "a wise wizard with a long beard",
            "a fierce warrior princess"
        ]
    }


@pytest.fixture
def sample_3d_mesh(temp_dir):
    """Create a sample 3D mesh for testing."""
    import trimesh
    
    # Create a simple cube mesh
    mesh = trimesh.creation.box()
    
    # Save in different formats
    formats = {
        "glb": temp_dir / "test_mesh.glb",
        "obj": temp_dir / "test_mesh.obj",
        "ply": temp_dir / "test_mesh.ply"
    }
    
    for fmt, path in formats.items():
        mesh.export(str(path))
    
    return {"mesh": mesh, "paths": formats}


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            self.peak_memory = None
            self.metrics = {}
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss
            self.peak_memory = self.start_memory
            
        def update(self):
            current_memory = psutil.Process().memory_info().rss
            self.peak_memory = max(self.peak_memory, current_memory)
            
        def stop(self):
            self.end_time = time.time()
            self.end_memory = psutil.Process().memory_info().rss
            
            self.metrics = {
                "duration": self.end_time - self.start_time,
                "memory_start_mb": self.start_memory / 1024**2,
                "memory_end_mb": self.end_memory / 1024**2,
                "memory_peak_mb": self.peak_memory / 1024**2,
                "memory_increase_mb": (self.end_memory - self.start_memory) / 1024**2
            }
            
            return self.metrics
    
    return PerformanceMonitor()


@pytest.fixture
def concurrent_executor():
    """Execute tasks concurrently for testing."""
    class ConcurrentExecutor:
        def __init__(self):
            self.results = []
            self.errors = []
            
        async def run_async(self, tasks):
            """Run async tasks concurrently."""
            self.results = await asyncio.gather(*tasks, return_exceptions=True)
            self.errors = [r for r in self.results if isinstance(r, Exception)]
            return self.results
            
        def run_threaded(self, func, args_list, max_workers=5):
            """Run function with different args in threads."""
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(func, *args) for args in args_list]
                self.results = [f.result() for f in futures]
            
            return self.results
    
    return ConcurrentExecutor()


# =============================================================================
# Utility Functions
# =============================================================================

def assert_valid_image(image: Any, expected_size: Optional[tuple] = None):
    """Assert that an image is valid."""
    assert isinstance(image, Image.Image)
    if expected_size:
        assert image.size == expected_size
    assert image.mode in ["RGB", "RGBA", "L"]


def assert_valid_3d_result(result: Dict[str, Any]):
    """Assert that a 3D generation result is valid."""
    assert "mesh" in result or "mesh_path" in result
    if "vertices" in result:
        assert isinstance(result["vertices"], np.ndarray)
        assert result["vertices"].shape[1] == 3
    if "faces" in result:
        assert isinstance(result["faces"], np.ndarray)
        assert result["faces"].shape[1] == 3
    if "texture" in result:
        assert_valid_image(result["texture"])


def assert_valid_video_result(result: Dict[str, Any]):
    """Assert that a video generation result is valid."""
    assert "frames" in result or "video_path" in result
    if "frames" in result:
        assert len(result["frames"]) > 0
        for frame in result["frames"]:
            assert_valid_image(frame)
    if "video_path" in result:
        assert Path(result["video_path"]).suffix in ['.mp4', '.avi', '.mov', '.webm']


def create_mock_progress_tracker():
    """Create a mock progress tracker."""
    class MockProgressTracker:
        def __init__(self):
            self.updates = []
            
        def update(self, stage: str, progress: float, message: str = ""):
            self.updates.append({
                "stage": stage,
                "progress": progress,
                "message": message,
                "timestamp": time.time()
            })
            
        def get_stages(self) -> List[str]:
            return [u["stage"] for u in self.updates]
            
        def assert_complete(self):
            assert len(self.updates) > 0
            assert self.updates[-1]["progress"] >= 0.99
            
        def assert_stages_present(self, expected_stages: List[str]):
            stages = self.get_stages()
            for stage in expected_stages:
                assert stage in stages, f"Stage '{stage}' not found in {stages}"
    
    return MockProgressTracker()


# =============================================================================
# Markers and Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "ui: marks UI-related tests")
    config.addinivalue_line("markers", "api: marks API tests")
    config.addinivalue_line("markers", "desktop: marks desktop app tests")
    config.addinivalue_line("markers", "web: marks web app tests")
    config.addinivalue_line("markers", "performance: marks performance tests")


# =============================================================================
# Session-level Setup
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment for all tests."""
    import os
    
    # Set test mode environment variables
    os.environ["NEURALFORGE_TEST_MODE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Disable GPU for tests by default (can be overridden)
    if "NEURALFORGE_TEST_GPU" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    yield
    
    # Cleanup
    os.environ.pop("NEURALFORGE_TEST_MODE", None)