"""Memory-mapped file loading for large model files."""

import mmap
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Union, BinaryIO
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryMappedFile:
    """Memory-mapped file wrapper for efficient large file access."""
    
    def __init__(self, file_path: Path, mode: str = 'r'):
        """Initialize memory-mapped file.
        
        Args:
            file_path: Path to the file
            mode: Access mode ('r' for read-only, 'r+' for read-write)
        """
        self.file_path = Path(file_path)
        self.mode = mode
        self._file: Optional[BinaryIO] = None
        self._mmap: Optional[mmap.mmap] = None
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def open(self):
        """Open the file and create memory map."""
        if self._mmap is not None:
            return  # Already open
            
        try:
            # Open file in binary mode
            if self.mode == 'r':
                self._file = open(self.file_path, 'rb')
                access = mmap.ACCESS_READ
            elif self.mode == 'r+':
                self._file = open(self.file_path, 'r+b')
                access = mmap.ACCESS_WRITE
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            
            # Create memory map
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=access)
            logger.debug(f"Memory-mapped file opened: {self.file_path}")
            
        except Exception as e:
            logger.error(f"Failed to memory-map file {self.file_path}: {e}")
            if self._file:
                self._file.close()
                self._file = None
            raise
    
    def close(self):
        """Close the memory map and file."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
            
        if self._file is not None:
            self._file.close()
            self._file = None
            
        logger.debug(f"Memory-mapped file closed: {self.file_path}")
    
    def read(self, offset: int = 0, size: int = -1) -> bytes:
        """Read data from memory-mapped file.
        
        Args:
            offset: Starting position
            size: Number of bytes to read (-1 for all remaining)
            
        Returns:
            Bytes read from file
        """
        if self._mmap is None:
            raise RuntimeError("File not opened")
            
        self._mmap.seek(offset)
        
        if size == -1:
            return self._mmap.read()
        else:
            return self._mmap.read(size)
    
    def read_array(self, dtype: np.dtype, offset: int = 0, count: int = -1) -> np.ndarray:
        """Read numpy array from memory-mapped file.
        
        Args:
            dtype: Numpy data type
            offset: Starting position in bytes
            count: Number of elements to read (-1 for all)
            
        Returns:
            Numpy array
        """
        if self._mmap is None:
            raise RuntimeError("File not opened")
            
        # Calculate element size
        element_size = dtype.itemsize
        
        # Determine number of elements
        if count == -1:
            remaining = len(self._mmap) - offset
            count = remaining // element_size
        
        # Read data
        self._mmap.seek(offset)
        data = self._mmap.read(count * element_size)
        
        # Convert to numpy array
        return np.frombuffer(data, dtype=dtype)
    
    @property
    def size(self) -> int:
        """Get file size."""
        if self._mmap is None:
            return self.file_path.stat().st_size
        return len(self._mmap)


@contextmanager
def memory_mapped_weights(file_path: Path, dtype: str = 'float32'):
    """Context manager for memory-mapped weight loading.
    
    Args:
        file_path: Path to weights file
        dtype: Data type of weights
        
    Yields:
        Numpy array view of the weights
    """
    import torch
    
    # Map string dtype to numpy dtype
    dtype_map = {
        'float32': np.float32,
        'float16': np.float16,
        'int8': np.int8,
        'uint8': np.uint8,
    }
    
    np_dtype = dtype_map.get(dtype, np.float32)
    
    try:
        # Use numpy's memory-mapped array
        weights = np.memmap(file_path, dtype=np_dtype, mode='r')
        logger.info(f"Memory-mapped weights from {file_path}, shape: {weights.shape}")
        
        # Convert to torch tensor without copying
        if torch.cuda.is_available():
            # For GPU, we need to copy but can do it in chunks
            tensor = torch.from_numpy(weights).to('cuda')
        else:
            # For CPU, create a view without copying
            tensor = torch.from_numpy(weights)
        
        yield tensor
        
    finally:
        # Cleanup is automatic for memmap
        del weights
        if 'tensor' in locals():
            del tensor


def load_safetensors_mmap(file_path: Path) -> dict:
    """Load safetensors file using memory mapping.
    
    Args:
        file_path: Path to safetensors file
        
    Returns:
        Dictionary of tensors
    """
    try:
        import safetensors
        from safetensors import safe_open
        
        tensors = {}
        
        # Open safetensors file with memory mapping
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Get tensor without loading into memory
                tensors[key] = f.get_tensor(key)
                
        logger.info(f"Loaded {len(tensors)} tensors from {file_path} using memory mapping")
        return tensors
        
    except ImportError:
        logger.warning("safetensors not available, falling back to regular loading")
        import torch
        return torch.load(file_path, map_location='cpu')


def estimate_memory_usage(file_path: Path, dtype: str = 'float32') -> dict:
    """Estimate memory usage for a model file.
    
    Args:
        file_path: Path to model file
        dtype: Expected data type
        
    Returns:
        Dictionary with memory estimates
    """
    file_size = file_path.stat().st_size
    
    # Estimate based on dtype
    dtype_sizes = {
        'float32': 4,
        'float16': 2,
        'int8': 1,
        'uint8': 1,
    }
    
    bytes_per_param = dtype_sizes.get(dtype, 4)
    estimated_params = file_size / bytes_per_param
    
    return {
        'file_size_mb': file_size / (1024 * 1024),
        'estimated_params': int(estimated_params),
        'estimated_memory_mb': file_size / (1024 * 1024),
        'recommended_ram_gb': (file_size / (1024 * 1024 * 1024)) * 2.5  # 2.5x for overhead
    }