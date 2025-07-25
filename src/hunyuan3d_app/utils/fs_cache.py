"""File system operations cache for improved performance."""

import os
import time
import logging
from pathlib import Path
from functools import lru_cache, wraps
from typing import List, Optional, Tuple, Dict, Any
import threading

logger = logging.getLogger(__name__)


class FileSystemCache:
    """Cache for expensive file system operations."""
    
    def __init__(self, ttl: float = 5.0):
        """Initialize with time-to-live in seconds."""
        self.ttl = ttl
        self._cache = {}
        self._lock = threading.Lock()
    
    def _cache_key(self, func_name: str, args: tuple) -> str:
        """Generate cache key from function name and arguments."""
        # Convert Path objects to strings for hashability
        normalized_args = []
        for arg in args:
            if isinstance(arg, Path):
                normalized_args.append(str(arg.absolute()))
            else:
                normalized_args.append(arg)
        return f"{func_name}:{':'.join(map(str, normalized_args))}"
    
    def get(self, func_name: str, args: tuple) -> Optional[Any]:
        """Get cached value if not expired."""
        key = self._cache_key(func_name, args)
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    # Expired, remove from cache
                    del self._cache[key]
        return None
    
    def set(self, func_name: str, args: tuple, value: Any):
        """Set cache value with current timestamp."""
        key = self._cache_key(func_name, args)
        with self._lock:
            self._cache[key] = (value, time.time())
    
    def clear(self):
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
    
    def clear_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        with self._lock:
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp >= self.ttl
            ]
            for key in expired_keys:
                del self._cache[key]


# Global cache instance
_fs_cache = FileSystemCache(ttl=5.0)


def cached_fs_operation(func):
    """Decorator to cache file system operation results."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Only cache if no kwargs (for simplicity)
        if kwargs:
            return func(*args, **kwargs)
        
        # Check cache
        cached_value = _fs_cache.get(func.__name__, args)
        if cached_value is not None:
            return cached_value
        
        # Execute and cache
        result = func(*args, **kwargs)
        _fs_cache.set(func.__name__, args, result)
        return result
    
    return wrapper


# Cached versions of common file system operations
@cached_fs_operation
def cached_exists(path: Path) -> bool:
    """Cached version of Path.exists()."""
    return path.exists()


@cached_fs_operation
def cached_is_dir(path: Path) -> bool:
    """Cached version of Path.is_dir()."""
    return path.is_dir()


@cached_fs_operation
def cached_is_file(path: Path) -> bool:
    """Cached version of Path.is_file()."""
    return path.is_file()


@cached_fs_operation
def cached_stat(path: Path) -> os.stat_result:
    """Cached version of Path.stat()."""
    return path.stat()


@cached_fs_operation
def cached_listdir(path: Path) -> List[str]:
    """Cached version of os.listdir()."""
    return sorted(os.listdir(path))


@lru_cache(maxsize=128)
def cached_glob(path: Path, pattern: str) -> List[Path]:
    """Cached version of Path.glob() - uses lru_cache for longer caching."""
    return list(path.glob(pattern))


@lru_cache(maxsize=128)
def cached_iterdir(path: Path) -> List[Path]:
    """Cached version of Path.iterdir() - uses lru_cache for longer caching."""
    try:
        return sorted(path.iterdir())
    except (OSError, PermissionError):
        return []


class CachedPath:
    """Path wrapper with cached operations."""
    
    def __init__(self, path: Path):
        self._path = Path(path)
    
    @property
    def path(self) -> Path:
        return self._path
    
    def exists(self) -> bool:
        return cached_exists(self._path)
    
    def is_dir(self) -> bool:
        return cached_is_dir(self._path)
    
    def is_file(self) -> bool:
        return cached_is_file(self._path)
    
    def stat(self) -> os.stat_result:
        return cached_stat(self._path)
    
    def iterdir(self) -> List[Path]:
        return cached_iterdir(self._path)
    
    def glob(self, pattern: str) -> List[Path]:
        return cached_glob(self._path, pattern)
    
    def __truediv__(self, other):
        """Support path joining with /."""
        return CachedPath(self._path / other)
    
    def __str__(self):
        return str(self._path)
    
    def __repr__(self):
        return f"CachedPath({self._path!r})"


def optimize_path_checks(paths: List[Path]) -> Dict[Path, Dict[str, Any]]:
    """Batch check multiple paths efficiently."""
    results = {}
    
    for path in paths:
        # Single stat call gets all info
        try:
            stat_info = path.stat()
            results[path] = {
                'exists': True,
                'is_file': os.path.isfile(path),
                'is_dir': os.path.isdir(path),
                'size': stat_info.st_size,
                'mtime': stat_info.st_mtime
            }
        except (OSError, FileNotFoundError):
            results[path] = {
                'exists': False,
                'is_file': False,
                'is_dir': False,
                'size': 0,
                'mtime': 0
            }
    
    return results


def clear_fs_cache():
    """Clear the file system cache."""
    _fs_cache.clear()
    cached_glob.cache_clear()
    cached_iterdir.cache_clear()
    logger.debug("File system cache cleared")