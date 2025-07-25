"""Thumbnail management for generation history"""

import logging
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


class ThumbnailManager:
    """Manages thumbnail creation and storage"""
    
    def __init__(self, thumbnails_dir: Path):
        self.thumbnails_dir = Path(thumbnails_dir)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
    
    def create_thumbnail(
        self, 
        generation_id: str, 
        output_path: str, 
        size: Tuple[int, int] = (256, 256)
    ) -> Optional[Path]:
        """Create thumbnail for an output file
        
        Args:
            generation_id: Unique generation ID
            output_path: Path to output file
            size: Thumbnail size (width, height)
            
        Returns:
            Path to thumbnail or None if failed
        """
        try:
            output_path = Path(output_path)
            
            # Skip if not an image or doesn't exist
            if not output_path.exists() or output_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.webp']:
                return None
            
            # Create thumbnail filename
            thumb_name = f"{generation_id}_{output_path.stem}_thumb.jpg"
            thumb_path = self.thumbnails_dir / thumb_name
            
            # Skip if already exists
            if thumb_path.exists():
                return thumb_path
            
            # Load and create thumbnail
            with Image.open(output_path) as img:
                # Convert RGBA to RGB if needed
                if img.mode == 'RGBA':
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                
                # Create thumbnail maintaining aspect ratio
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Save thumbnail
                img.save(thumb_path, 'JPEG', quality=85, optimize=True)
                
            return thumb_path
            
        except Exception as e:
            logger.error(f"Failed to create thumbnail for {output_path}: {e}")
            return None
    
    def delete_thumbnails(self, generation_id: str) -> int:
        """Delete all thumbnails for a generation
        
        Args:
            generation_id: Generation ID
            
        Returns:
            Number of thumbnails deleted
        """
        deleted = 0
        pattern = f"{generation_id}_*_thumb.jpg"
        
        for thumb_path in self.thumbnails_dir.glob(pattern):
            try:
                thumb_path.unlink()
                deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete thumbnail {thumb_path}: {e}")
        
        return deleted
    
    def cleanup_orphaned_thumbnails(self, valid_generation_ids: set) -> int:
        """Clean up thumbnails without corresponding generations
        
        Args:
            valid_generation_ids: Set of valid generation IDs
            
        Returns:
            Number of thumbnails deleted
        """
        deleted = 0
        
        for thumb_path in self.thumbnails_dir.glob("*_thumb.jpg"):
            # Extract generation ID from filename
            generation_id = thumb_path.stem.split('_')[0]
            
            if generation_id not in valid_generation_ids:
                try:
                    thumb_path.unlink()
                    deleted += 1
                except Exception as e:
                    logger.error(f"Failed to delete orphaned thumbnail {thumb_path}: {e}")
        
        return deleted