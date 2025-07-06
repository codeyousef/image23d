"""Generation history and metadata tracking"""

import json
import logging
import shutil
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import gradio as gr
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GenerationRecord:
    """Record of a single generation"""
    id: str
    timestamp: float
    generation_type: str  # "image", "3d", "full_pipeline"
    model_name: str
    prompt: str
    negative_prompt: Optional[str]
    parameters: Dict[str, Any]
    output_paths: List[str]
    thumbnails: List[str]
    metadata: Dict[str, Any]
    tags: List[str] = None
    favorite: bool = False
    viewed: bool = False
    viewed_at: Optional[float] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationRecord':
        """Create from dictionary"""
        return cls(**data)


class HistoryManager:
    """Manages generation history with SQLite backend"""
    
    def __init__(self, db_path: Path, thumbnails_dir: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.thumbnails_dir = Path(thumbnails_dir)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            # Check if viewed columns exist (for migration)
            cursor = conn.execute("PRAGMA table_info(generations)")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Add viewed columns if they don't exist
            if 'viewed' not in columns:
                try:
                    conn.execute("ALTER TABLE generations ADD COLUMN viewed INTEGER DEFAULT 0")
                    conn.execute("ALTER TABLE generations ADD COLUMN viewed_at REAL")
                    conn.commit()
                    logger.info("Added viewed tracking columns to existing database")
                except sqlite3.OperationalError:
                    # Table doesn't exist yet, will be created below
                    pass
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generations (
                    id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    generation_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    negative_prompt TEXT,
                    parameters TEXT NOT NULL,
                    output_paths TEXT NOT NULL,
                    thumbnails TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    tags TEXT,
                    favorite INTEGER DEFAULT 0,
                    viewed INTEGER DEFAULT 0,
                    viewed_at REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON generations(timestamp DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_name 
                ON generations(model_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_generation_type 
                ON generations(generation_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_favorite 
                ON generations(favorite)
            """)
            
            conn.commit()
            
        logger.info(f"Initialized history database at {self.db_path}")
        
    def add_generation(
        self,
        generation_id: str,
        generation_type: str,
        model_name: str,
        prompt: str,
        negative_prompt: Optional[str],
        parameters: Dict[str, Any],
        output_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> GenerationRecord:
        """Add a new generation to history
        
        Args:
            generation_id: Unique ID for the generation
            generation_type: Type of generation
            model_name: Model used
            prompt: Generation prompt
            negative_prompt: Negative prompt if used
            parameters: Generation parameters
            output_paths: List of output file paths
            metadata: Additional metadata
            
        Returns:
            Created generation record
        """
        # Create thumbnails
        thumbnails = []
        for output_path in output_paths:
            try:
                thumbnail_path = self._create_thumbnail(generation_id, output_path)
                if thumbnail_path:
                    thumbnails.append(str(thumbnail_path))
            except Exception as e:
                logger.error(f"Error creating thumbnail: {e}")
                
        # Create record
        record = GenerationRecord(
            id=generation_id,
            timestamp=datetime.now().timestamp(),
            generation_type=generation_type,
            model_name=model_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            parameters=parameters,
            output_paths=output_paths,
            thumbnails=thumbnails,
            metadata=metadata or {}
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO generations (
                    id, timestamp, generation_type, model_name, 
                    prompt, negative_prompt, parameters, output_paths,
                    thumbnails, metadata, tags, favorite, viewed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id,
                record.timestamp,
                record.generation_type,
                record.model_name,
                record.prompt,
                record.negative_prompt,
                json.dumps(record.parameters),
                json.dumps(record.output_paths),
                json.dumps(record.thumbnails),
                json.dumps(record.metadata),
                json.dumps(record.tags),
                int(record.favorite),
                0  # New generations are unviewed by default
            ))
            conn.commit()
            
        logger.info(f"Added generation {generation_id} to history")
        return record
        
    def _create_thumbnail(self, generation_id: str, output_path: str, size: Tuple[int, int] = (256, 256)) -> Optional[Path]:
        """Create thumbnail for an output file
        
        Args:
            generation_id: Generation ID
            output_path: Path to output file
            size: Thumbnail size
            
        Returns:
            Path to thumbnail or None
        """
        output_path = Path(output_path)
        if not output_path.exists():
            return None
            
        # Handle different file types
        if output_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
            # Image file
            try:
                img = Image.open(output_path)
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                thumbnail_path = self.thumbnails_dir / f"{generation_id}_{output_path.stem}_thumb.png"
                img.save(thumbnail_path, "PNG")
                
                return thumbnail_path
            except Exception as e:
                logger.error(f"Error creating image thumbnail: {e}")
                return None
                
        elif output_path.suffix.lower() in ['.glb', '.obj', '.ply', '.stl']:
            # 3D file - would need special handling
            # For now, return a placeholder
            placeholder_path = self.thumbnails_dir / "3d_placeholder.png"
            if not placeholder_path.exists():
                # Create a simple placeholder
                img = Image.new('RGB', size, color=(128, 128, 128))
                img.save(placeholder_path)
            return placeholder_path
            
        return None
        
    def get_history(
        self,
        limit: int = 50,
        offset: int = 0,
        generation_type: Optional[str] = None,
        model_name: Optional[str] = None,
        search_query: Optional[str] = None,
        favorites_only: bool = False,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[GenerationRecord]:
        """Get generation history with filters
        
        Args:
            limit: Maximum number of records
            offset: Offset for pagination
            generation_type: Filter by type
            model_name: Filter by model
            search_query: Search in prompts
            favorites_only: Only show favorites
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of generation records
        """
        query = "SELECT * FROM generations WHERE 1=1"
        params = []
        
        if generation_type:
            query += " AND generation_type = ?"
            params.append(generation_type)
            
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
            
        if search_query:
            query += " AND (prompt LIKE ? OR negative_prompt LIKE ?)"
            search_pattern = f"%{search_query}%"
            params.extend([search_pattern, search_pattern])
            
        if favorites_only:
            query += " AND favorite = 1"
            
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.timestamp())
            
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.timestamp())
            
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        records = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            for row in cursor:
                # Check which columns are available
                columns = row.keys()
                
                record = GenerationRecord(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    generation_type=row['generation_type'],
                    model_name=row['model_name'],
                    prompt=row['prompt'],
                    negative_prompt=row['negative_prompt'],
                    parameters=json.loads(row['parameters']),
                    output_paths=json.loads(row['output_paths']),
                    thumbnails=json.loads(row['thumbnails']),
                    metadata=json.loads(row['metadata']),
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    favorite=bool(row['favorite']),
                    viewed=bool(row['viewed']) if 'viewed' in columns else False,
                    viewed_at=row['viewed_at'] if 'viewed_at' in columns else None
                )
                records.append(record)
                
        return records
        
    def get_record(self, generation_id: str) -> Optional[GenerationRecord]:
        """Get a specific generation record"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM generations WHERE id = ?",
                (generation_id,)
            )
            row = cursor.fetchone()
            
            if row:
                # Check which columns are available
                columns = row.keys()
                
                return GenerationRecord(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    generation_type=row['generation_type'],
                    model_name=row['model_name'],
                    prompt=row['prompt'],
                    negative_prompt=row['negative_prompt'],
                    parameters=json.loads(row['parameters']),
                    output_paths=json.loads(row['output_paths']),
                    thumbnails=json.loads(row['thumbnails']),
                    metadata=json.loads(row['metadata']),
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    favorite=bool(row['favorite']),
                    viewed=bool(row['viewed']) if 'viewed' in columns else False,
                    viewed_at=row['viewed_at'] if 'viewed_at' in columns else None
                )
                
        return None
        
    def update_record(
        self,
        generation_id: str,
        tags: Optional[List[str]] = None,
        favorite: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a generation record
        
        Args:
            generation_id: ID to update
            tags: New tags
            favorite: New favorite status
            metadata: Updated metadata
            
        Returns:
            Success status
        """
        record = self.get_record(generation_id)
        if not record:
            return False
            
        updates = []
        params = []
        
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
            
        if favorite is not None:
            updates.append("favorite = ?")
            params.append(int(favorite))
            
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))
            
        if not updates:
            return True
            
        query = f"UPDATE generations SET {', '.join(updates)} WHERE id = ?"
        params.append(generation_id)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(query, params)
            conn.commit()
            
        return True
        
    def delete_record(self, generation_id: str, delete_files: bool = False) -> bool:
        """Delete a generation record
        
        Args:
            generation_id: ID to delete
            delete_files: Whether to delete output files
            
        Returns:
            Success status
        """
        record = self.get_record(generation_id)
        if not record:
            return False
            
        # Delete files if requested
        if delete_files:
            for path in record.output_paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception as e:
                    logger.error(f"Error deleting file {path}: {e}")
                    
            for path in record.thumbnails:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception as e:
                    logger.error(f"Error deleting thumbnail {path}: {e}")
                    
        # Delete from database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM generations WHERE id = ?", (generation_id,))
            conn.commit()
            
        return True
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total generations
            total = conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
            
            # By type
            type_stats = {}
            for row in conn.execute("""
                SELECT generation_type, COUNT(*) as count 
                FROM generations 
                GROUP BY generation_type
            """):
                type_stats[row[0]] = row[1]
                
            # By model
            model_stats = {}
            for row in conn.execute("""
                SELECT model_name, COUNT(*) as count 
                FROM generations 
                GROUP BY model_name
                ORDER BY count DESC
                LIMIT 10
            """):
                model_stats[row[0]] = row[1]
                
            # Favorites
            favorites = conn.execute(
                "SELECT COUNT(*) FROM generations WHERE favorite = 1"
            ).fetchone()[0]
            
            # Unviewed count
            unviewed = conn.execute(
                "SELECT COUNT(*) FROM generations WHERE viewed = 0"
            ).fetchone()[0]
            
            # Date range
            date_range = conn.execute("""
                SELECT MIN(timestamp), MAX(timestamp) 
                FROM generations
            """).fetchone()
            
        return {
            "total_generations": total,
            "by_type": type_stats,
            "top_models": model_stats,
            "favorites": favorites,
            "unviewed": unviewed,
            "date_range": {
                "start": datetime.fromtimestamp(date_range[0]) if date_range[0] else None,
                "end": datetime.fromtimestamp(date_range[1]) if date_range[1] else None
            }
        }
        
    def export_history(self, output_path: Path, include_files: bool = False):
        """Export history to JSON file
        
        Args:
            output_path: Where to save export
            include_files: Whether to include output files
        """
        records = self.get_history(limit=10000)  # Get all records
        
        export_data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "records": [r.to_dict() for r in records]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if include_files:
            # Create a directory for the export
            export_dir = output_path.with_suffix('')
            export_dir.mkdir(exist_ok=True)
            
            # Copy files
            files_dir = export_dir / "files"
            files_dir.mkdir(exist_ok=True)
            
            for record in records:
                for i, path in enumerate(record.output_paths):
                    src = Path(path)
                    if src.exists():
                        dst = files_dir / f"{record.id}_{i}{src.suffix}"
                        shutil.copy2(src, dst)
                        # Update path in export
                        export_data["records"][records.index(record)]["output_paths"][i] = str(dst.relative_to(export_dir))
                        
            # Save JSON in export directory
            json_path = export_dir / "history.json"
        else:
            json_path = output_path
            
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported history to {json_path}")
        
    def mark_as_viewed(self, generation_id: str) -> bool:
        """Mark a generation as viewed
        
        Args:
            generation_id: ID of generation to mark as viewed
            
        Returns:
            Success status
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE generations 
                SET viewed = 1, viewed_at = ? 
                WHERE id = ? AND viewed = 0
            """, (datetime.now().timestamp(), generation_id))
            conn.commit()
            return conn.total_changes > 0
            
    def get_unviewed_count(self) -> int:
        """Get count of unviewed generations
        
        Returns:
            Number of unviewed generations
        """
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM generations WHERE viewed = 0"
            ).fetchone()[0]
        return count
        
    def get_history_count(
        self,
        generation_type: Optional[str] = None,
        model_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """Get count of history records with filters
        
        Args:
            generation_type: Filter by type
            model_name: Filter by model
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            Count of matching records
        """
        query = "SELECT COUNT(*) FROM generations WHERE 1=1"
        params = []
        
        if generation_type:
            query += " AND generation_type = ?"
            params.append(generation_type)
            
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
            
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.timestamp())
            
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.timestamp())
            
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(query, params).fetchone()[0]
            
        return count
        
    def delete_generation(self, generation_id: str) -> bool:
        """Delete a generation record and its files
        
        Args:
            generation_id: ID to delete
            
        Returns:
            Success status
        """
        return self.delete_record(generation_id, delete_files=True)
        
    def toggle_favorite(self, generation_id: str) -> bool:
        """Toggle favorite status of a generation
        
        Args:
            generation_id: ID to toggle
            
        Returns:
            Success status
        """
        record = self.get_record(generation_id)
        if record:
            return self.update_record(generation_id, favorite=not record.favorite)
        return False
        
    def scan_outputs_directory(self, outputs_dir: Path, import_untracked: bool = True) -> int:
        """Scan outputs directory and import untracked files
        
        Args:
            outputs_dir: Path to outputs directory
            import_untracked: Whether to import files not in database
            
        Returns:
            Number of files imported
        """
        outputs_dir = Path(outputs_dir)
        if not outputs_dir.exists():
            logger.warning(f"Outputs directory does not exist: {outputs_dir}")
            return 0
            
        imported_count = 0
        
        # Get all existing file paths from database
        existing_paths = set()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT output_paths FROM generations")
            for row in cursor:
                paths = json.loads(row[0])
                existing_paths.update(paths)
        
        # Scan for image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        model_extensions = {'.glb', '.obj', '.ply', '.stl', '.fbx', '.usdz'}
        video_extensions = {'.mp4', '.avi', '.mov', '.webm'}
        
        all_extensions = image_extensions | model_extensions | video_extensions
        
        for file_path in outputs_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in all_extensions:
                # Skip if already in database
                if str(file_path) in existing_paths:
                    continue
                    
                # Skip thumbnails
                if 'thumbnail' in file_path.name.lower() or '_thumb' in file_path.name:
                    continue
                    
                if import_untracked:
                    # Determine generation type
                    if file_path.suffix.lower() in image_extensions:
                        generation_type = "image"
                    elif file_path.suffix.lower() in model_extensions:
                        generation_type = "3d"
                    elif file_path.suffix.lower() in video_extensions:
                        generation_type = "video"
                    else:
                        continue
                    
                    # Extract info from filename if possible
                    file_name = file_path.stem
                    
                    # Try to extract prompt from filename (common patterns)
                    prompt = file_name
                    if '_' in file_name:
                        # Handle patterns like "prompt_timestamp" or "type_prompt_id"
                        parts = file_name.split('_')
                        if parts[0] in ['image', '3d', 'video', 'model']:
                            prompt = ' '.join(parts[1:])
                        else:
                            prompt = ' '.join(parts)
                    
                    # Generate ID based on file
                    import hashlib
                    file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
                    generation_id = f"imported_{file_hash}_{int(time.time())}"
                    
                    # Get file modification time as timestamp
                    timestamp = file_path.stat().st_mtime
                    
                    # Add to database
                    try:
                        self.add_generation(
                            generation_id=generation_id,
                            generation_type=generation_type,
                            model_name="Unknown",
                            prompt=prompt[:100],  # Limit prompt length
                            negative_prompt=None,
                            parameters={"imported": True, "original_path": str(file_path)},
                            output_paths=[str(file_path)],
                            metadata={
                                "imported_at": datetime.now().isoformat(),
                                "file_size": file_path.stat().st_size,
                                "original_name": file_path.name
                            }
                        )
                        
                        # Override timestamp with file modification time
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute(
                                "UPDATE generations SET timestamp = ? WHERE id = ?",
                                (timestamp, generation_id)
                            )
                            conn.commit()
                        
                        imported_count += 1
                        logger.info(f"Imported {file_path.name}")
                        
                    except Exception as e:
                        logger.error(f"Failed to import {file_path}: {e}")
        
        logger.info(f"Imported {imported_count} files from outputs directory")
        return imported_count
        
    def create_ui_component(self) -> gr.Group:
        """Create Gradio UI component for history browsing
        
        Returns:
            Gradio Group component
        """
        with gr.Group() as history_group:
            gr.Markdown("### 📚 Generation History")
            
            # Search and filters
            with gr.Row():
                search_box = gr.Textbox(
                    label="Search prompts",
                    placeholder="Search...",
                    scale=3
                )
                
                type_filter = gr.Dropdown(
                    choices=["All", "image", "3d", "full_pipeline"],
                    value="All",
                    label="Type",
                    scale=1
                )
                
                favorites_only = gr.Checkbox(
                    label="Favorites only",
                    value=False
                )
                
            # Gallery
            gallery = gr.Gallery(
                label="History",
                show_label=False,
                elem_id="history_gallery",
                columns=4,
                rows=3,
                object_fit="cover",
                height="auto"
            )
            
            # Selected item details
            with gr.Row():
                with gr.Column(scale=2):
                    selected_info = gr.JSON(
                        label="Generation Details",
                        value={}
                    )
                    
                with gr.Column(scale=1):
                    # Actions
                    favorite_btn = gr.Button("⭐ Toggle Favorite", size="sm")
                    regenerate_btn = gr.Button("🔄 Regenerate", size="sm")
                    delete_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                    
            # Pagination
            with gr.Row():
                prev_btn = gr.Button("◀ Previous", size="sm")
                page_info = gr.Textbox(
                    value="Page 1",
                    interactive=False,
                    show_label=False
                )
                next_btn = gr.Button("Next ▶", size="sm")
                
            # Hidden state
            current_page = gr.State(0)
            selected_id = gr.State(None)
            
            # Functions
            def load_gallery(search, type_filter, favorites, page):
                """Load gallery images"""
                limit = 12
                offset = page * limit
                
                records = self.get_history(
                    limit=limit,
                    offset=offset,
                    generation_type=type_filter if type_filter != "All" else None,
                    search_query=search if search else None,
                    favorites_only=favorites
                )
                
                images = []
                for record in records:
                    if record.thumbnails:
                        # Use first thumbnail
                        thumb_path = record.thumbnails[0]
                        if Path(thumb_path).exists():
                            images.append((thumb_path, record.prompt[:50] + "..."))
                            
                return images, f"Page {page + 1}"
                
            def select_item(evt: gr.SelectData, search, type_filter, favorites, page):
                """Handle gallery selection"""
                limit = 12
                offset = page * limit
                
                records = self.get_history(
                    limit=limit,
                    offset=offset,
                    generation_type=type_filter if type_filter != "All" else None,
                    search_query=search if search else None,
                    favorites_only=favorites
                )
                
                if evt.index < len(records):
                    record = records[evt.index]
                    return record.to_dict(), record.id
                    
                return {}, None
                
            def toggle_favorite(selected_id):
                """Toggle favorite status"""
                if selected_id:
                    record = self.get_record(selected_id)
                    if record:
                        self.update_record(selected_id, favorite=not record.favorite)
                        
            def delete_item(selected_id):
                """Delete selected item"""
                if selected_id:
                    self.delete_record(selected_id, delete_files=False)
                    
            # Wire up events
            search_box.change(
                lambda s, t, f: load_gallery(s, t, f, 0),
                inputs=[search_box, type_filter, favorites_only],
                outputs=[gallery, page_info]
            ).then(
                lambda: 0,
                outputs=[current_page]
            )
            
            type_filter.change(
                lambda s, t, f: load_gallery(s, t, f, 0),
                inputs=[search_box, type_filter, favorites_only],
                outputs=[gallery, page_info]
            ).then(
                lambda: 0,
                outputs=[current_page]
            )
            
            favorites_only.change(
                lambda s, t, f: load_gallery(s, t, f, 0),
                inputs=[search_box, type_filter, favorites_only],
                outputs=[gallery, page_info]
            ).then(
                lambda: 0,
                outputs=[current_page]
            )
            
            gallery.select(
                select_item,
                inputs=[search_box, type_filter, favorites_only, current_page],
                outputs=[selected_info, selected_id]
            )
            
            favorite_btn.click(
                toggle_favorite,
                inputs=[selected_id]
            ).then(
                lambda s, t, f, p: load_gallery(s, t, f, p),
                inputs=[search_box, type_filter, favorites_only, current_page],
                outputs=[gallery, page_info]
            )
            
            delete_btn.click(
                delete_item,
                inputs=[selected_id]
            ).then(
                lambda s, t, f, p: load_gallery(s, t, f, p),
                inputs=[search_box, type_filter, favorites_only, current_page],
                outputs=[gallery, page_info]
            )
            
            prev_btn.click(
                lambda p: max(0, p - 1),
                inputs=[current_page],
                outputs=[current_page]
            ).then(
                lambda s, t, f, p: load_gallery(s, t, f, p),
                inputs=[search_box, type_filter, favorites_only, current_page],
                outputs=[gallery, page_info]
            )
            
            next_btn.click(
                lambda p: p + 1,
                inputs=[current_page],
                outputs=[current_page]
            ).then(
                lambda s, t, f, p: load_gallery(s, t, f, p),
                inputs=[search_box, type_filter, favorites_only, current_page],
                outputs=[gallery, page_info]
            )
            
            # Initial load - set default values instead of using .load()
            # The gallery will be populated when the tab is first accessed
            
        return history_group