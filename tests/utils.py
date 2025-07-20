"""Test utilities and helper functions."""

import time
import asyncio
import json
from contextlib import contextmanager
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import torch
import trimesh


class MockProgressTracker:
    """Track progress calls for testing."""
    
    def __init__(self):
        self.calls = []
        self.last_progress = 0.0
        self.last_message = ""
    
    def __call__(self, progress, message=""):
        self.calls.append((progress, message))
        self.last_progress = progress
        self.last_message = message
    
    def get_progress_values(self):
        """Get list of progress values."""
        return [call[0] for call in self.calls]
    
    def get_messages(self):
        """Get list of messages."""
        return [call[1] for call in self.calls]
    
    def assert_progress_increases(self):
        """Assert that progress values increase monotonically."""
        values = self.get_progress_values()
        for i in range(1, len(values)):
            assert values[i] >= values[i-1], f"Progress decreased: {values[i-1]} -> {values[i]}"
    
    def assert_reaches_complete(self):
        """Assert that progress reaches 1.0."""
        assert any(p >= 1.0 for p in self.get_progress_values()), "Progress never reached 1.0"


@contextmanager
def mock_model_download(success=True, progress_updates=10):
    """Mock model downloading with progress updates."""
    def mock_download(*args, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        if progress_callback:
            for i in range(progress_updates):
                progress_callback(i / progress_updates, f"Downloading... {i}/{progress_updates}")
                time.sleep(0.01)  # Simulate download time
        
        if success:
            return True, "Download complete"
        else:
            return False, "Download failed"
    
    with patch("hunyuan3d_app.models.manager.ModelManager.download_model", side_effect=mock_download):
        yield


@contextmanager
def mock_image_generation(generation_time=1.0, steps=20):
    """Mock image generation with realistic progress."""
    def mock_generate(*args, **kwargs):
        progress = kwargs.get("progress")
        if progress:
            for step in range(steps):
                progress(step / steps, f"Step {step + 1}/{steps}")
                time.sleep(generation_time / steps)
        
        # Return a generated image
        image = Image.new("RGB", (512, 512), color="blue")
        info = {
            "prompt": kwargs.get("prompt", "test prompt"),
            "seed": kwargs.get("seed", 42),
            "steps": steps
        }
        return image, info
    
    with patch("hunyuan3d_app.generation.image.ImageGenerator.generate_image", side_effect=mock_generate):
        yield


def create_test_image(size=(512, 512), color=None):
    """Create a test image with optional pattern."""
    if color:
        return Image.new("RGB", size, color=color)
    else:
        # Create a gradient pattern
        width, height = size
        array = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                array[y, x] = [x % 256, y % 256, (x + y) % 256]
        return Image.fromarray(array)


def create_test_mesh(num_vertices: Optional[int] = None):
    """Create a simple test mesh."""
    if num_vertices is None:
        # Create a simple pyramid
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, 0.5, 1]
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3]
        ])
    else:
        # Create a mesh with specified vertices
        vertices = np.random.rand(num_vertices, 3)
        # Simple triangulation
        faces = []
        for i in range(0, num_vertices - 2, 3):
            if i + 2 < num_vertices:
                faces.append([i, i + 1, i + 2])
        faces = np.array(faces)
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def assert_generates_output(func, *args, **kwargs):
    """Assert that a generation function produces output."""
    result = func(*args, **kwargs)
    assert result is not None, "Function returned None"
    
    if isinstance(result, tuple):
        assert all(item is not None for item in result), "Function returned None in tuple"
    
    return result


def assert_saves_file(func, output_dir, *args, **kwargs):
    """Assert that a function saves a file to the output directory."""
    initial_files = set(output_dir.glob("*"))
    
    result = func(*args, **kwargs)
    
    final_files = set(output_dir.glob("*"))
    new_files = final_files - initial_files
    
    assert len(new_files) > 0, "No new files were created"
    
    return result, list(new_files)


class AsyncMock(MagicMock):
    """Mock for async functions."""
    
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


def mock_websocket_server():
    """Create a mock WebSocket server."""
    server = AsyncMock()
    server.start_server = AsyncMock()
    server.send_progress_update = AsyncMock()
    server.shutdown = AsyncMock()
    return server


def compare_images(img1, img2, threshold=0.95):
    """Compare two images for similarity."""
    if img1.size != img2.size:
        return False
    
    # Convert to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Calculate similarity
    diff = np.abs(arr1.astype(float) - arr2.astype(float))
    similarity = 1.0 - (diff.mean() / 255.0)
    
    return similarity >= threshold


def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
    """Wait for a condition to become true."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        time.sleep(interval)
    return False


def mock_gpu_memory(total_gb=24, used_gb=8):
    """Mock GPU memory statistics."""
    total_bytes = int(total_gb * 1024**3)
    used_bytes = int(used_gb * 1024**3)
    free_bytes = total_bytes - used_bytes
    
    with patch("torch.cuda.mem_get_info", return_value=(free_bytes, total_bytes)), \
         patch("torch.cuda.memory_allocated", return_value=used_bytes):
        yield


def create_mock_job(job_id="test-123", status="completed", progress=1.0):
    """Create a mock GenerationJob."""
    from hunyuan3d_app.services.queue import GenerationJob, JobStatus, JobPriority
    
    status_map = {
        "pending": JobStatus.PENDING,
        "running": JobStatus.RUNNING,
        "completed": JobStatus.COMPLETED,
        "failed": JobStatus.FAILED
    }
    
    job = GenerationJob(
        id=job_id,
        type="test",
        params={"test": True},
        priority=JobPriority.NORMAL,
        status=status_map.get(status, JobStatus.COMPLETED),
        progress=progress,
        progress_message=f"Progress: {progress * 100:.0f}%",
        result={"success": True} if status == "completed" else None,
        error="Test error" if status == "failed" else None
    )
    
    return job


def run_async(coro):
    """Run an async coroutine in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Performance testing utilities
class PerformanceTimer:
    """Simple performance timer for tests."""
    
    def __init__(self):
        self.times = {}
    
    @contextmanager
    def time(self, name):
        start = time.time()
        yield
        elapsed = time.time() - start
        self.times[name] = elapsed
    
    def assert_under(self, name, max_seconds):
        """Assert operation completed under time limit."""
        assert name in self.times, f"No timing recorded for {name}"
        assert self.times[name] < max_seconds, f"{name} took {self.times[name]:.2f}s, expected < {max_seconds}s"


def create_test_face_data(position: Tuple[int, int] = (256, 256)):
    """Create mock face detection data."""
    face = Mock()
    face.bbox = [position[0] - 50, position[1] - 50, position[0] + 50, position[1] + 50]
    face.kps = np.array([
        [position[0] - 20, position[1] - 20],  # Left eye
        [position[0] + 20, position[1] - 20],  # Right eye
        [position[0], position[1]],            # Nose
        [position[0] - 20, position[1] + 20],  # Left mouth
        [position[0] + 20, position[1] + 20]   # Right mouth
    ])
    face.embedding = np.random.rand(512)
    face.age = 25
    face.gender = "M"
    face.occlusion_score = 0.1
    return face


def mock_hunyuan3d_model():
    """Create a mock Hunyuan3D model."""
    model = {
        "type": "hunyuan3d",
        "name": "hunyuan3d-21",
        "version": "2.1",
        "status": "loaded"
    }
    
    def generate_3d(image, **kwargs):
        progress = kwargs.get("progress_callback")
        if progress:
            for i in range(10):
                progress(i/10, f"Generating 3D model: {i*10}%")
        return create_test_mesh()
    
    model["generate_3d"] = generate_3d
    return model


def create_test_video_frames(num_frames: int = 8, size: Tuple[int, int] = (512, 512)):
    """Create test video frames."""
    frames = []
    for i in range(num_frames):
        # Create frames with gradual color change
        color = (255 * i // num_frames, 128, 255 - 255 * i // num_frames)
        frame = Image.new("RGB", size, color=color)
        frames.append(frame)
    return frames