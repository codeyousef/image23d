"""File operation utilities."""

import os
import shutil
import hashlib
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import tempfile


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_move_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """Safely move file to destination.
    
    Args:
        src: Source file path
        dst: Destination path (file or directory)
        
    Returns:
        Path to moved file
        
    Raises:
        FileNotFoundError: If source doesn't exist
        OSError: If move fails
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    # If dst is a directory, use source filename
    if dst_path.is_dir():
        dst_path = dst_path / src_path.name
    
    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Move file
    shutil.move(str(src_path), str(dst_path))
    return dst_path


def safe_copy_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """Safely copy file to destination.
    
    Args:
        src: Source file path
        dst: Destination path (file or directory)
        
    Returns:
        Path to copied file
        
    Raises:
        FileNotFoundError: If source doesn't exist
        OSError: If copy fails
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    # If dst is a directory, use source filename
    if dst_path.is_dir():
        dst_path = dst_path / src_path.name
    
    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy file
    shutil.copy2(str(src_path), str(dst_path))
    return dst_path


def safe_delete_file(path: Union[str, Path]) -> bool:
    """Safely delete file if it exists.
    
    Args:
        path: File path
        
    Returns:
        True if file was deleted, False if it didn't exist
    """
    path = Path(path)
    
    if path.exists():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        return True
    return False


def get_file_hash(
    path: Union[str, Path], 
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> str:
    """Calculate file hash.
    
    Args:
        path: File path
        algorithm: Hash algorithm (sha256, md5, etc.)
        chunk_size: Read chunk size
        
    Returns:
        Hex digest of file hash
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    hasher = hashlib.new(algorithm)
    
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def read_json_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Read JSON file safely.
    
    Args:
        path: JSON file path
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(
    path: Union[str, Path], 
    data: Dict[str, Any],
    indent: int = 2
) -> None:
    """Write JSON file safely.
    
    Args:
        path: JSON file path
        data: Data to write
        indent: JSON indentation
    """
    path = Path(path)
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file first
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=path.parent,
        delete=False
    ) as f:
        json.dump(data, f, indent=indent)
        temp_path = f.name
    
    # Move temp file to final location
    shutil.move(temp_path, str(path))


def get_directory_size(path: Union[str, Path]) -> int:
    """Get total size of directory in bytes.
    
    Args:
        path: Directory path
        
    Returns:
        Total size in bytes
    """
    path = Path(path)
    total = 0
    
    if path.is_file():
        return path.stat().st_size
    
    for entry in path.rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    
    return total


def find_files_by_extension(
    directory: Union[str, Path],
    extensions: Union[str, List[str]],
    recursive: bool = True
) -> List[Path]:
    """Find all files with given extensions.
    
    Args:
        directory: Directory to search
        extensions: File extension(s) to find
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    
    if isinstance(extensions, str):
        extensions = [extensions]
    
    # Ensure extensions start with dot
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    files = []
    pattern = "**/*" if recursive else "*"
    
    for item in directory.glob(pattern):
        if item.is_file() and item.suffix.lower() in extensions:
            files.append(item)
    
    return sorted(files)


def create_unique_filename(
    base_path: Union[str, Path],
    prefix: str = "",
    suffix: str = "",
    extension: str = ""
) -> Path:
    """Create a unique filename in directory.
    
    Args:
        base_path: Base directory path
        prefix: Filename prefix
        suffix: Filename suffix
        extension: File extension
        
    Returns:
        Unique file path
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    counter = 1
    while True:
        if counter == 1:
            filename = f"{prefix}{suffix}{extension}"
        else:
            filename = f"{prefix}{suffix}_{counter}{extension}"
        
        file_path = base_path / filename
        if not file_path.exists():
            return file_path
        
        counter += 1