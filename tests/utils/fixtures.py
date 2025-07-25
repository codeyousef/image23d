"""Additional pytest fixtures for the test suite."""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json
import time
from typing import Dict, List, Any, Optional

# Import test helpers
from .test_helpers import (
    MockModelManager, 
    TestDataGenerator, 
    MockGPUEnvironment,
    FileSystemHelper,
    APITestHelper
)


# =============================================================================
# Core Application Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_models_dir(tmp_path):
    """Create mock models directory with test model files."""
    models_dir = tmp_path / "models"
    
    model_configs = {
        "SDXL-Turbo": {
            "type": "image",
            "size_mb": 800,
            "gpu_memory_gb": 2.0,
            "architecture": "diffusion",
            "supported_resolutions": [512, 768, 1024]
        },
        "FLUX.1-dev": {
            "type": "image", 
            "size_mb": 1500,
            "gpu_memory_gb": 4.0,
            "architecture": "flux",
            "supported_resolutions": [512, 768, 1024, 1344]
        },
        "LCM-LoRA": {
            "type": "image",
            "size_mb": 200,
            "gpu_memory_gb": 1.0,
            "architecture": "lcm",
            "supported_resolutions": [512, 768]
        },
        "HunYuan3D-Mini": {
            "type": "3d",
            "size_mb": 1200,
            "gpu_memory_gb": 3.0,
            "architecture": "hunyuan3d",
            "max_resolution": 512
        },
        "HunYuan3D-2.1": {
            "type": "3d",
            "size_mb": 2500,
            "gpu_memory_gb": 6.0,
            "architecture": "hunyuan3d",
            "max_resolution": 1024
        },
        "LTX-Video": {
            "type": "video",
            "size_mb": 1800,
            "gpu_memory_gb": 5.0,
            "architecture": "ltx",
            "max_duration": 10.0
        }
    }
    
    FileSystemHelper.create_mock_model_files(models_dir, model_configs)
    return models_dir


@pytest.fixture
def mock_output_dir(tmp_path):
    """Create mock output directory with test files."""
    output_dir = tmp_path / "outputs"
    FileSystemHelper.create_test_outputs(output_dir, count=10)
    return output_dir


@pytest.fixture
def mock_cache_dir(tmp_path):
    """Create mock cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create some cached files
    for i in range(5):
        cache_file = cache_dir / f"cached_item_{i}.bin"
        cache_file.write_bytes(b"cached data " * 100)
    
    return cache_dir


@pytest.fixture
def mock_config_file(tmp_path):
    """Create mock configuration file."""
    config_path = tmp_path / "config.json"
    
    config = {
        "models": {
            "default_image_model": "SDXL-Turbo",
            "default_3d_model": "HunYuan3D-Mini",
            "default_video_model": "LTX-Video"
        },
        "generation": {
            "default_width": 512,
            "default_height": 512,
            "default_steps": 20,
            "default_guidance_scale": 7.5
        },
        "system": {
            "max_gpu_memory_gb": 8.0,
            "enable_memory_optimization": True,
            "output_format": "png"
        },
        "api": {
            "base_url": "http://localhost:8000",
            "timeout": 30,
            "max_retries": 3
        }
    }
    
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


# =============================================================================
# Model and Processing Fixtures
# =============================================================================

@pytest.fixture
def mock_local_model_manager():
    """Provide mock local model manager."""
    return MockModelManager()


@pytest.fixture
def mock_hunyuan3d_model():
    """Mock HunYuan3D model for testing."""
    mock_model = Mock()
    
    def mock_generate(*args, **kwargs):
        # Simulate generation time
        time.sleep(0.1)
        
        return {
            "mesh": {
                "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "faces": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                "normals": [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, -1]]
            },
            "texture": TestDataGenerator.create_test_image(1024, 1024),
            "multiview_images": [
                TestDataGenerator.create_test_image(256, 256) for _ in range(6)
            ],
            "metadata": {
                "generation_time": 15.3,
                "model_version": "2.1",
                "quality": "standard"
            }
        }
    
    mock_model.generate = mock_generate
    mock_model.model_name = "HunYuan3D-2.1"
    mock_model.is_loaded = True
    
    return mock_model


@pytest.fixture
def mock_image_model():
    """Mock image generation model."""
    mock_model = Mock()
    
    def mock_generate(prompt, **kwargs):
        time.sleep(0.05)  # Simulate quick generation
        width = kwargs.get('width', 512)
        height = kwargs.get('height', 512)
        return TestDataGenerator.create_test_image(width, height)
    
    mock_model.generate = mock_generate
    mock_model.model_name = "SDXL-Turbo"
    mock_model.is_loaded = True
    
    return mock_model


@pytest.fixture
def mock_video_model():
    """Mock video generation model."""
    mock_model = Mock()
    
    def mock_generate(prompt, **kwargs):
        time.sleep(0.2)  # Simulate video generation
        duration = kwargs.get('duration', 3.0)
        fps = kwargs.get('fps', 24)
        frames = int(duration * fps)
        
        return {
            "frames": [
                TestDataGenerator.create_test_image(512, 512) for _ in range(frames)
            ],
            "metadata": {
                "duration": duration,
                "fps": fps,
                "frame_count": frames
            }
        }
    
    mock_model.generate = mock_generate
    mock_model.model_name = "LTX-Video"
    mock_model.is_loaded = True
    
    return mock_model


# =============================================================================
# GPU and System Fixtures
# =============================================================================

@pytest.fixture
def mock_gpu_environment():
    """Provide mock GPU environment."""
    with MockGPUEnvironment(gpu_count=1, gpu_memory_gb=8.0) as gpu_env:
        yield gpu_env


@pytest.fixture
def mock_cpu_only_environment():
    """Mock CPU-only environment (no CUDA)."""
    with MockGPUEnvironment(cuda_available=False) as gpu_env:
        yield gpu_env


@pytest.fixture
def mock_low_memory_environment():
    """Mock low GPU memory environment."""
    with MockGPUEnvironment(gpu_memory_gb=2.0) as gpu_env:
        yield gpu_env


@pytest.fixture
def mock_multi_gpu_environment():
    """Mock multi-GPU environment."""
    with MockGPUEnvironment(gpu_count=2, gpu_memory_gb=16.0) as gpu_env:
        yield gpu_env


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def test_images():
    """Provide various test images."""
    generator = TestDataGenerator()
    
    return {
        "small_square": generator.create_test_image(256, 256, pattern="random"),
        "large_square": generator.create_test_image(1024, 1024, pattern="gradient"),
        "portrait": generator.create_test_image(512, 768, pattern="checkerboard"),
        "landscape": generator.create_test_image(768, 512, pattern="random"),
        "test_face": generator.create_test_image(512, 512, pattern="gradient"),
        "test_object": generator.create_test_image(512, 512, pattern="checkerboard"),
    }


@pytest.fixture
def test_prompts():
    """Provide various test prompts."""
    generator = TestDataGenerator()
    
    return {
        "simple": generator.create_test_prompts(10, style="simple"),
        "detailed": generator.create_test_prompts(10, style="detailed"),
        "varied": generator.create_test_prompts(10, style="varied"),
        "long_prompt": "A highly detailed, photorealistic image of a majestic ancient castle perched on a cliff overlooking a stormy ocean, with dramatic lighting from the setting sun breaking through dark storm clouds, rendered in the style of classical romantic landscape painting with intricate architectural details and atmospheric perspective",
        "short_prompt": "Red car",
        "empty_prompt": "",
        "special_chars": "Test with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?",
    }


@pytest.fixture
def test_generation_configs():
    """Provide various generation configurations."""
    generator = TestDataGenerator()
    
    return {
        "image_fast": generator.create_generation_config("image", "fast"),
        "image_standard": generator.create_generation_config("image", "standard"),
        "image_high_quality": generator.create_generation_config("image", "high_quality"),
        "3d_draft": generator.create_generation_config("3d", "draft"),
        "3d_standard": generator.create_generation_config("3d", "standard"),
        "3d_high_quality": generator.create_generation_config("3d", "high_quality"),
        "video_short": {**generator.create_generation_config("video"), "duration": 1.0},
        "video_long": {**generator.create_generation_config("video"), "duration": 5.0},
    }


# =============================================================================
# API and Network Fixtures
# =============================================================================

@pytest.fixture
def mock_api_client():
    """Mock API client for testing."""
    mock_client = Mock()
    helper = APITestHelper()
    
    # Mock common API responses
    mock_client.get_models.return_value = helper.create_mock_response_data("models_list")
    mock_client.create_job.return_value = helper.create_mock_response_data("job_created")
    mock_client.get_job_status.return_value = helper.create_mock_response_data("job_completed")
    
    return mock_client


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection."""
    mock_ws = Mock()
    mock_ws.send = AsyncMock()
    mock_ws.recv = AsyncMock()
    mock_ws.close = AsyncMock()
    
    # Mock message queue
    mock_ws.message_queue = []
    
    async def mock_recv():
        if mock_ws.message_queue:
            return mock_ws.message_queue.pop(0)
        await asyncio.sleep(0.1)  # Simulate waiting
        return '{"type": "heartbeat", "timestamp": ' + str(time.time()) + '}'
    
    mock_ws.recv = mock_recv
    
    return mock_ws


@pytest.fixture
def mock_http_server():
    """Mock HTTP server responses."""
    
    responses = {
        "/api/v1/models": {
            "models": [
                {"id": "flux-1-dev", "name": "FLUX.1-dev", "type": "image"},
                {"id": "sdxl-turbo", "name": "SDXL-Turbo", "type": "image"},
                {"id": "hunyuan3d-mini", "name": "HunYuan3D-Mini", "type": "3d"}
            ]
        },
        "/api/v1/user/profile": {
            "user_id": "test-user-123",
            "email": "test@example.com",
            "credits": 1000,
            "tier": "pro"
        },
        "/api/v1/jobs/create": {
            "job_id": "test-job-123",
            "status": "queued",
            "created_at": time.time(),
            "queue_position": 1
        }
    }
    
    return responses


# =============================================================================
# Performance and Load Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_monitor():
    """Provide performance monitoring tools."""
    import psutil
    
    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.start_time = None
            self.start_memory = None
            self.measurements = []
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss
            self.measurements = []
        
        def measure(self, label: str = ""):
            current_time = time.time()
            current_memory = self.process.memory_info().rss
            
            measurement = {
                "label": label,
                "elapsed_time": current_time - self.start_time if self.start_time else 0,
                "memory_mb": current_memory / (1024 * 1024),
                "memory_delta_mb": (current_memory - self.start_memory) / (1024 * 1024) if self.start_memory else 0,
                "cpu_percent": self.process.cpu_percent()
            }
            
            self.measurements.append(measurement)
            return measurement
        
        def stop(self):
            final_measurement = self.measure("final")
            return {
                "total_time": final_measurement["elapsed_time"],
                "peak_memory_mb": max(m["memory_mb"] for m in self.measurements),
                "total_memory_delta_mb": final_measurement["memory_delta_mb"],
                "measurements": self.measurements
            }
    
    return PerformanceMonitor()


@pytest.fixture
def load_test_data():
    """Provide data for load testing."""
    
    return {
        "concurrent_requests": [
            {"prompt": f"Load test image {i}", "seed": i} 
            for i in range(50)
        ],
        "batch_prompts": [
            f"Batch test prompt {i}: A {['red', 'blue', 'green'][i % 3]} {['car', 'house', 'tree'][i % 3]}"
            for i in range(20)
        ],
        "stress_configs": [
            {"width": 512, "height": 512, "steps": 20},
            {"width": 768, "height": 768, "steps": 30},
            {"width": 1024, "height": 1024, "steps": 50},
        ]
    }


# =============================================================================
# Error and Edge Case Fixtures
# =============================================================================

@pytest.fixture
def error_scenarios():
    """Provide various error scenarios for testing."""
    
    return {
        "network_timeout": {
            "exception": "requests.exceptions.Timeout",
            "message": "Request timed out"
        },
        "gpu_oom": {
            "exception": "torch.cuda.OutOfMemoryError", 
            "message": "CUDA out of memory"
        },
        "invalid_model": {
            "exception": "ValueError",
            "message": "Unknown model: invalid-model"
        },
        "file_not_found": {
            "exception": "FileNotFoundError",
            "message": "Model file not found"
        },
        "permission_denied": {
            "exception": "PermissionError",
            "message": "Permission denied accessing model file"
        },
        "api_rate_limit": {
            "status_code": 429,
            "message": "Rate limit exceeded"
        },
        "api_server_error": {
            "status_code": 500,
            "message": "Internal server error"
        }
    }


@pytest.fixture
def edge_case_inputs():
    """Provide edge case inputs for testing."""
    
    return {
        "empty_prompt": "",
        "very_long_prompt": "A " + "very " * 1000 + "long prompt",
        "unicode_prompt": "测试中文提示词 with émojis 🎨🖼️",
        "special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
        "extreme_dimensions": {"width": 2048, "height": 2048},
        "tiny_dimensions": {"width": 64, "height": 64},
        "invalid_seed": -1,
        "huge_seed": 2**64,
        "zero_steps": 0,
        "negative_guidance": -1.0,
        "extreme_guidance": 100.0
    }


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    temp_files = []
    temp_dirs = []
    
    def track_temp_file(path):
        temp_files.append(Path(path))
    
    def track_temp_dir(path):
        temp_dirs.append(Path(path))
    
    yield track_temp_file, track_temp_dir
    
    # Cleanup
    for temp_file in temp_files:
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            pass
    
    for temp_dir in temp_dirs:
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    # Reset any global variables or singletons
    yield
    
    # Cleanup after test
    import gc
    gc.collect()


# =============================================================================
# Marker Fixtures
# =============================================================================

@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no GPU available."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def skip_if_no_internet():
    """Skip test if no internet connection."""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
    except (socket.timeout, socket.gaierror):
        pytest.skip("Internet connection not available")


# =============================================================================
# Session-scoped Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_session_data():
    """Session-wide test data that persists across tests."""
    
    return {
        "session_id": f"test_session_{int(time.time())}",
        "start_time": time.time(),
        "test_counter": 0,
        "global_config": {
            "enable_performance_tracking": True,
            "log_level": "DEBUG",
            "test_timeout": 300
        }
    }


if __name__ == "__main__":
    # Test fixtures functionality
    print("Testing fixtures...")
    
    # Test temporary directory creation
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test model directory creation
        models_dir = temp_path / "models"
        FileSystemHelper.create_mock_model_files(models_dir, {
            "test-model": {"type": "image", "size_mb": 100}
        })
        
        assert (models_dir / "test-model" / "model.safetensors").exists()
        print("✓ Mock model files created")
        
        # Test output directory creation
        output_dir = temp_path / "outputs"
        FileSystemHelper.create_test_outputs(output_dir, count=3)
        
        assert len(list(output_dir.glob("*.png"))) == 3
        print("✓ Test output files created")
    
    print("All fixtures working correctly!")