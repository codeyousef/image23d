"""Pytest configuration and fixtures for Hunyuan3D app tests."""

# Import mock dependencies first
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import mock_dependencies

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from PIL import Image
import logging

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

# Ensure we're using a test-specific temp directory
TEST_TEMP_DIR = Path(tempfile.mkdtemp(prefix="hunyuan3d_test_"))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment for all tests."""
    # Set environment variables for testing
    import os
    os.environ["HUNYUAN3D_TEST_MODE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU for tests by default
    
    yield
    
    # Cleanup
    if TEST_TEMP_DIR.exists():
        shutil.rmtree(TEST_TEMP_DIR)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    temp_path = TEST_TEMP_DIR / f"test_{np.random.randint(1000000)}"
    temp_path.mkdir(parents=True, exist_ok=True)
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def mock_models_dir(temp_dir):
    """Create a mock models directory."""
    models_dir = temp_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (models_dir / "image").mkdir(exist_ok=True)
    (models_dir / "3d").mkdir(exist_ok=True)
    (models_dir / "loras").mkdir(exist_ok=True)
    (models_dir / "gguf").mkdir(exist_ok=True)
    
    return models_dir


@pytest.fixture
def mock_output_dir(temp_dir):
    """Create a mock output directory."""
    output_dir = temp_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    # Create a simple 512x512 RGB image
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_image_path(temp_dir, sample_image):
    """Save sample image to a file and return the path."""
    img_path = temp_dir / "sample_image.png"
    sample_image.save(img_path)
    return img_path


@pytest.fixture
def mock_torch_model():
    """Create a mock PyTorch model."""
    model = MagicMock()
    model.device = torch.device("cpu")
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    model.half = MagicMock(return_value=model)
    model.float = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_diffusion_pipeline():
    """Create a mock diffusion pipeline."""
    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")
    pipeline.to = MagicMock(return_value=pipeline)
    pipeline.enable_attention_slicing = MagicMock()
    pipeline.enable_model_cpu_offload = MagicMock()
    pipeline.enable_xformers_memory_efficient_attention = MagicMock()
    
    # Mock the __call__ method to return images
    def mock_generate(**kwargs):
        result = MagicMock()
        result.images = [Image.new("RGB", (512, 512), color="red")]
        return result
    
    pipeline.__call__ = MagicMock(side_effect=mock_generate)
    pipeline.vae = MagicMock()
    pipeline.text_encoder = MagicMock()
    pipeline.unet = MagicMock()
    
    return pipeline


@pytest.fixture
def mock_flux_pipeline(mock_diffusion_pipeline):
    """Create a mock FLUX pipeline with device_map."""
    pipeline = mock_diffusion_pipeline
    pipeline.__class__.__name__ = "FluxPipeline"
    pipeline.hf_device_map = {"transformer": 0, "vae": 0}
    return pipeline


@pytest.fixture
def mock_gradio_progress():
    """Create a mock Gradio Progress object."""
    progress = MagicMock()
    progress.__call__ = MagicMock()
    return progress


@pytest.fixture
def mock_hunyuan3d_model():
    """Create a mock Hunyuan3D model."""
    model = {
        "type": "hunyuan3d",
        "name": "hunyuan3d-21",
        "version": "2.1",
        "status": "loaded"
    }
    return model


@pytest.fixture
def mock_mesh_data():
    """Create mock 3D mesh data."""
    # Create a mock mesh object
    mesh = MagicMock()
    mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mesh.faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    mesh.export = MagicMock()
    return mesh


@pytest.fixture
def mock_app_config(mock_models_dir, mock_output_dir, temp_dir):
    """Mock application configuration."""
    with patch("hunyuan3d_app.config.MODELS_DIR", mock_models_dir), \
         patch("hunyuan3d_app.config.OUTPUT_DIR", mock_output_dir), \
         patch("hunyuan3d_app.config.CACHE_DIR", temp_dir / "cache"):
        yield {
            "models_dir": mock_models_dir,
            "output_dir": mock_output_dir,
            "cache_dir": temp_dir / "cache"
        }


@pytest.fixture
def mock_model_manager():
    """Create a mock ModelManager."""
    manager = MagicMock()
    manager.get_downloaded_models = MagicMock(return_value=["FLUX.1-dev", "SDXL-base"])
    manager.download_model = MagicMock(return_value=(True, "Model downloaded"))
    manager.load_image_model = MagicMock(return_value=("✅ Loaded", Mock(), "FLUX.1-dev"))
    manager.load_model = MagicMock(return_value=(True, "Model loaded"))
    manager.check_model_complete = MagicMock(return_value=True)
    return manager


@pytest.fixture
def mock_queue_manager():
    """Create a mock QueueManager."""
    from hunyuan3d_app.services.queue import GenerationJob, JobStatus, JobPriority
    
    manager = MagicMock()
    
    # Create a sample job
    job = GenerationJob(
        id="test-job-123",
        type="image",
        params={"prompt": "test"},
        priority=JobPriority.NORMAL,
        status=JobStatus.COMPLETED,
        progress=1.0,
        result={"image": Image.new("RGB", (512, 512))}
    )
    
    manager.submit_job = MagicMock(return_value=job)
    manager.get_job = MagicMock(return_value=job)
    manager.get_queue_status = MagicMock(return_value={
        "total_jobs": 1,
        "pending": 0,
        "active": 0,
        "completed": 1
    })
    
    return manager


@pytest.fixture
def mock_history_manager():
    """Create a mock HistoryManager."""
    manager = MagicMock()
    manager.add_generation = MagicMock()
    manager.get_history = MagicMock(return_value=[])
    manager.get_statistics = MagicMock(return_value={
        "total_generations": 10,
        "favorites": 2
    })
    return manager


# GPU/CUDA related fixtures
@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_properties") as mock_props:
        
        # Mock device properties
        props = MagicMock()
        props.name = "Mock GPU"
        props.total_memory = 8 * 1024**3  # 8GB
        mock_props.return_value = props
        
        yield


@pytest.fixture
def mock_no_cuda():
    """Mock no CUDA available."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


# Test data fixtures
@pytest.fixture
def test_prompts():
    """Common test prompts."""
    return [
        "a cute cat wearing a hat",
        "futuristic robot in neon city",
        "ancient temple in the jungle",
        "abstract colorful patterns"
    ]


@pytest.fixture
def test_model_configs():
    """Test model configurations."""
    return {
        "FLUX.1-dev": {
            "repo_id": "black-forest-labs/FLUX.1-dev",
            "size": "24GB",
            "vram_required": "16GB"
        },
        "SDXL-base": {
            "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "size": "7GB",
            "vram_required": "8GB"
        }
    }


# Utility functions for tests
def assert_image_valid(image, expected_size=(512, 512)):
    """Assert that an image is valid."""
    assert isinstance(image, Image.Image)
    assert image.size == expected_size
    assert image.mode in ["RGB", "RGBA"]


def assert_file_exists(file_path):
    """Assert that a file exists."""
    assert Path(file_path).exists()
    assert Path(file_path).is_file()


def create_mock_model_files(model_dir, model_type="diffusion"):
    """Create mock model files in a directory."""
    if model_type == "diffusion":
        # Create basic diffusion model structure
        (model_dir / "model_index.json").write_text("{}")
        (model_dir / "unet").mkdir(exist_ok=True)
        (model_dir / "vae").mkdir(exist_ok=True)
        (model_dir / "text_encoder").mkdir(exist_ok=True)
    elif model_type == "gguf":
        # Create GGUF file
        (model_dir / "model.gguf").write_bytes(b"GGUF")
    elif model_type == "hunyuan3d":
        # Create Hunyuan3D model structure
        (model_dir / "dit").mkdir(exist_ok=True)
        (model_dir / "vae").mkdir(exist_ok=True)


# Markers for different test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "ui: marks UI-related tests")