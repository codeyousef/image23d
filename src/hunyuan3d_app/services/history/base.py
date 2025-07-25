"""Base History Manager Class

Core history management functionality.
"""

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import gradio as gr

from .models import GenerationRecord
from .database import DatabaseManager
from .thumbnail_manager import ThumbnailManager
from .statistics import StatisticsManager
from .export_import import ExportImportManager
from .ui_component import HistoryUIComponent

logger = logging.getLogger(__name__)


class HistoryManager:
    """Manages generation history with SQLite backend"""
    
    def __init__(self, db_path: Path, thumbnails_dir: Path):
        self.db_path = Path(db_path)
        self.thumbnails_dir = Path(thumbnails_dir)
        
        # Initialize components
        self.db = DatabaseManager(self.db_path)
        self.thumbnails = ThumbnailManager(self.thumbnails_dir)
        self.ui = HistoryUIComponent(self)
    
    def add_generation(
        self,
        generation_id: str,
        generation_type: str,
        model_name: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        output_paths: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Add a new generation to history
        
        Args:
            generation_id: Unique ID for this generation
            generation_type: Type of generation (image, 3d, full_pipeline)
            model_name: Name of the model used
            prompt: The prompt used
            negative_prompt: The negative prompt (optional)
            parameters: Generation parameters
            output_paths: List of output file paths
            metadata: Additional metadata
            tags: List of tags
            
        Returns:
            Success status
        """
        try:
            # Create thumbnails for output files
            thumbnails = []
            if output_paths:
                for output_path in output_paths:
                    thumb_path = self.thumbnails.create_thumbnail(generation_id, output_path)
                    if thumb_path:
                        thumbnails.append(str(thumb_path))
            
            # Create record
            record = GenerationRecord(
                id=generation_id,
                timestamp=time.time(),
                generation_type=generation_type,
                model_name=model_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                parameters=parameters or {},
                output_paths=output_paths or [],
                thumbnails=thumbnails,
                metadata=metadata or {},
                tags=tags or []
            )
            
            # Add to database
            return self.db.add_record(record)
            
        except Exception as e:
            logger.error(f"Failed to add generation to history: {e}")
            return False
    
    def get_history(
        self,
        limit: int = 50,
        offset: int = 0,
        generation_type: Optional[str] = None,
        model_name: Optional[str] = None,
        search_query: Optional[str] = None,
        favorites_only: bool = False,
        unviewed_only: bool = False,
        sort_by: str = "timestamp",
        sort_order: str = "DESC"
    ) -> List[GenerationRecord]:
        """Get generation history with filtering
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            generation_type: Filter by type
            model_name: Filter by model
            search_query: Search in prompts and tags
            favorites_only: Only return favorites
            unviewed_only: Only return unviewed items
            sort_by: Field to sort by
            sort_order: Sort order (ASC or DESC)
            
        Returns:
            List of generation records
        """
        return self.db.get_records(
            limit=limit,
            offset=offset,
            generation_type=generation_type,
            model_name=model_name,
            search_query=search_query,
            favorites_only=favorites_only,
            unviewed_only=unviewed_only,
            sort_by=sort_by,
            sort_order=sort_order
        )
    
    def get_record(self, generation_id: str) -> Optional[GenerationRecord]:
        """Get a single generation record
        
        Args:
            generation_id: ID of the generation
            
        Returns:
            Generation record or None
        """
        return self.db.get_record(generation_id)
    
    def update_record(self, generation_id: str, **kwargs) -> bool:
        """Update a generation record
        
        Args:
            generation_id: ID of the generation
            **kwargs: Fields to update
            
        Returns:
            Success status
        """
        return self.db.update_record(generation_id, **kwargs)
    
    def delete_record(self, generation_id: str, delete_files: bool = False) -> bool:
        """Delete a generation record
        
        Args:
            generation_id: ID of the generation
            delete_files: Whether to delete output files
            
        Returns:
            Success status
        """
        try:
            record = self.get_record(generation_id)
            if not record:
                return False
            
            # Delete files if requested
            if delete_files:
                for output_path in record.output_paths:
                    try:
                        Path(output_path).unlink()
                    except Exception as e:
                        logger.error(f"Failed to delete output file {output_path}: {e}")
            
            # Delete thumbnails
            self.thumbnails.delete_thumbnails(generation_id)
            
            # Delete from database
            return self.db.delete_record(generation_id)
            
        except Exception as e:
            logger.error(f"Failed to delete record {generation_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics
        
        Returns:
            Statistics dictionary
        """
        # Get basic stats from database
        db_stats = self.db.get_statistics()
        
        # Get recent records for additional analysis
        recent_records = self.get_history(limit=1000)
        
        # Calculate additional statistics
        if recent_records:
            detailed_stats = StatisticsManager.calculate_statistics(recent_records)
            usage_patterns = StatisticsManager.get_usage_patterns(recent_records)
            timeline = StatisticsManager.generate_timeline(recent_records, days=30)
            
            db_stats.update({
                "detailed": detailed_stats,
                "usage_patterns": usage_patterns,
                "timeline": timeline
            })
        
        return db_stats
    
    def export_history(self, output_path: Path, include_files: bool = False):
        """Export generation history
        
        Args:
            output_path: Path for export file
            include_files: Whether to include output files
        """
        records = self.get_history(limit=10000)  # Get all records
        
        # Determine base output directory
        base_output_dir = None
        if records and records[0].output_paths:
            first_path = Path(records[0].output_paths[0])
            # Try to find common base directory
            base_output_dir = first_path.parent.parent
        
        return ExportImportManager.export_history(
            records=records,
            output_path=output_path,
            include_files=include_files,
            base_output_dir=base_output_dir
        )
    
    def mark_as_viewed(self, generation_id: str) -> bool:
        """Mark a generation as viewed
        
        Args:
            generation_id: ID of the generation
            
        Returns:
            Success status
        """
        return self.update_record(
            generation_id,
            viewed=True,
            viewed_at=time.time()
        )
    
    def get_unviewed_count(self) -> int:
        """Get count of unviewed generations
        
        Returns:
            Number of unviewed generations
        """
        return self.db.get_count(unviewed_only=True)
    
    def get_history_count(
        self,
        generation_type: Optional[str] = None,
        favorites_only: bool = False
    ) -> int:
        """Get total count of generations
        
        Args:
            generation_type: Filter by type
            favorites_only: Only count favorites
            
        Returns:
            Total count
        """
        return self.db.get_count(
            generation_type=generation_type,
            favorites_only=favorites_only
        )
    
    def delete_generation(self, generation_id: str) -> bool:
        """Delete a generation (legacy compatibility)
        
        Args:
            generation_id: ID to delete
            
        Returns:
            Success status
        """
        return self.delete_record(generation_id, delete_files=True)
    
    def toggle_favorite(self, generation_id: str) -> bool:
        """Toggle favorite status
        
        Args:
            generation_id: ID of the generation
            
        Returns:
            New favorite status
        """
        record = self.get_record(generation_id)
        if record:
            new_status = not record.favorite
            self.update_record(generation_id, favorite=new_status)
            return new_status
        return False
    
    def scan_outputs_directory(self, outputs_dir: Path, import_untracked: bool = True) -> int:
        """Scan outputs directory for untracked generations
        
        Args:
            outputs_dir: Directory to scan
            import_untracked: Whether to import found generations
            
        Returns:
            Number of imported generations
        """
        imported = 0
        outputs_dir = Path(outputs_dir)
        
        if not outputs_dir.exists():
            return 0
        
        # Get existing generation IDs
        existing_ids = set(r.id for r in self.get_history(limit=10000))
        
        # Scan for metadata files
        for metadata_file in outputs_dir.rglob("metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                gen_id = metadata.get('generation_id')
                if not gen_id or gen_id in existing_ids:
                    continue
                
                # Find output files
                output_files = []
                gen_dir = metadata_file.parent
                for ext in ['.png', '.jpg', '.jpeg', '.webp', '.glb', '.obj', '.ply']:
                    output_files.extend(gen_dir.glob(f"*{ext}"))
                
                if output_files and import_untracked:
                    # Import the generation
                    self.add_generation(
                        generation_id=gen_id,
                        generation_type=metadata.get('type', 'unknown'),
                        model_name=metadata.get('model', 'unknown'),
                        prompt=metadata.get('prompt', ''),
                        negative_prompt=metadata.get('negative_prompt'),
                        parameters=metadata.get('parameters', {}),
                        output_paths=[str(f) for f in output_files],
                        metadata=metadata
                    )
                    imported += 1
                    
            except Exception as e:
                logger.error(f"Failed to import from {metadata_file}: {e}")
        
        return imported
    
    def create_ui_component(self) -> gr.Group:
        """Create Gradio UI component for history viewing
        
        Returns:
            Gradio Group component
        """
        return self.ui.create_ui_component()
    
    def cleanup_old_records(self, cutoff: datetime) -> int:
        """Clean up old records
        
        Args:
            cutoff: Delete records older than this
            
        Returns:
            Number of records deleted
        """
        cutoff_timestamp = cutoff.timestamp()
        
        # Get records to delete
        old_records = self.db.get_records(limit=10000)
        old_records = [r for r in old_records if r.timestamp < cutoff_timestamp and not r.favorite]
        
        # Delete thumbnails
        for record in old_records:
            self.thumbnails.delete_thumbnails(record.id)
        
        # Delete from database
        return self.db.cleanup_old_records(cutoff_timestamp)