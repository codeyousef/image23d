"""
Storage service for managing generated assets
"""

import os
import shutil
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import aiofiles
import hashlib

from core.config import OUTPUT_DIR

class StorageService:
    """
    Service for managing file storage and retrieval
    """
    
    def __init__(self):
        self.output_dir = OUTPUT_DIR
        self.temp_dir = OUTPUT_DIR / "temp"
        
    async def initialize(self):
        """Initialize storage service"""
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean old temp files
        await self._cleanup_temp_files()
        
    async def save_generated_file(
        self,
        file_data: bytes,
        user_id: str,
        job_id: str,
        file_type: str,
        file_extension: str
    ) -> str:
        """Save a generated file and return its URL"""
        # Create user directory
        user_dir = self.output_dir / user_id / file_type
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{file_type}_{timestamp}_{job_id[:8]}.{file_extension}"
        file_path = user_dir / filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_data)
            
        # Generate URL path
        url_path = f"/outputs/{user_id}/{file_type}/{filename}"
        
        return url_path
        
    async def save_temp_file(self, file_data: bytes, extension: str) -> Path:
        """Save a temporary file"""
        # Generate unique filename
        file_hash = hashlib.md5(file_data).hexdigest()[:8]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"temp_{timestamp}_{file_hash}.{extension}"
        file_path = self.temp_dir / filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_data)
            
        return file_path
        
    async def get_user_files(
        self,
        user_id: str,
        file_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get list of user's files"""
        user_dir = self.output_dir / user_id
        
        if not user_dir.exists():
            return []
            
        files = []
        
        # Get all files or specific type
        if file_type:
            search_dirs = [user_dir / file_type]
        else:
            search_dirs = [
                user_dir / "images",
                user_dir / "3d",
                user_dir / "videos"
            ]
            
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            for file_path in search_dir.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        "filename": file_path.name,
                        "path": str(file_path.relative_to(self.output_dir)),
                        "url": f"/outputs/{file_path.relative_to(self.output_dir)}",
                        "type": search_dir.name,
                        "size": stat.st_size,
                        "created_at": datetime.fromtimestamp(stat.st_ctime),
                        "modified_at": datetime.fromtimestamp(stat.st_mtime)
                    })
                    
        # Sort by created date (newest first)
        files.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply pagination
        return files[offset:offset + limit]
        
    async def delete_file(self, user_id: str, file_path: str) -> bool:
        """Delete a user's file"""
        # Validate path to prevent directory traversal
        full_path = self.output_dir / file_path
        user_dir = self.output_dir / user_id
        
        # Ensure file is within user's directory
        try:
            full_path.relative_to(user_dir)
        except ValueError:
            return False
            
        # Delete file
        if full_path.exists() and full_path.is_file():
            full_path.unlink()
            return True
            
        return False
        
    async def get_storage_stats(self, user_id: str) -> Dict[str, Any]:
        """Get storage statistics for a user"""
        user_dir = self.output_dir / user_id
        
        if not user_dir.exists():
            return {
                "total_files": 0,
                "total_size": 0,
                "by_type": {}
            }
            
        stats = {
            "total_files": 0,
            "total_size": 0,
            "by_type": {}
        }
        
        for type_dir in ["images", "3d", "videos"]:
            dir_path = user_dir / type_dir
            if not dir_path.exists():
                continue
                
            type_stats = {
                "count": 0,
                "size": 0
            }
            
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    type_stats["count"] += 1
                    type_stats["size"] += file_path.stat().st_size
                    
            stats["by_type"][type_dir] = type_stats
            stats["total_files"] += type_stats["count"]
            stats["total_size"] += type_stats["size"]
            
        return stats
        
    async def create_zip_archive(
        self,
        user_id: str,
        file_paths: List[str]
    ) -> Path:
        """Create a zip archive of multiple files"""
        import zipfile
        
        # Create temp zip file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        zip_name = f"export_{user_id}_{timestamp}.zip"
        zip_path = self.temp_dir / zip_name
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in file_paths:
                full_path = self.output_dir / file_path
                
                # Validate path
                try:
                    full_path.relative_to(self.output_dir / user_id)
                except ValueError:
                    continue
                    
                if full_path.exists() and full_path.is_file():
                    arcname = full_path.relative_to(self.output_dir / user_id)
                    zipf.write(full_path, arcname)
                    
        return zip_path
        
    async def _cleanup_temp_files(self):
        """Clean up old temporary files"""
        if not self.temp_dir.exists():
            return
            
        # Delete files older than 24 hours
        cutoff_time = datetime.utcnow().timestamp() - (24 * 60 * 60)
        
        for file_path in self.temp_dir.iterdir():
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()