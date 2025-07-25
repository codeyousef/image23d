"""Database operations for history management"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import threading
from queue import Queue, Empty

from .models import GenerationRecord

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Handles database operations for history with connection pooling"""
    
    # Connection pool settings
    _pool_size = 5
    _pool_timeout = 30  # seconds
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection pool
        self._connection_pool = Queue(maxsize=self._pool_size)
        self._all_connections = []
        self._pool_lock = threading.Lock()
        
        # Initialize database schema
        self._init_database()
        
        # Pre-populate the connection pool
        for _ in range(min(3, self._pool_size)):  # Start with 3 connections
            self._create_connection()
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimized settings"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        
        # Optimize SQLite settings for better performance
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
        
        with self._pool_lock:
            self._all_connections.append(conn)
        
        return conn
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection from the pool"""
        conn = None
        try:
            # Try to get a connection from the pool
            try:
                conn = self._connection_pool.get(timeout=self._pool_timeout)
            except Empty:
                # Pool is empty, create a new connection if under limit
                with self._pool_lock:
                    if len(self._all_connections) < self._pool_size:
                        conn = self._create_connection()
                    else:
                        # Wait for a connection to be available
                        conn = self._connection_pool.get()
            
            yield conn
            
        finally:
            # Return connection to pool
            if conn:
                try:
                    # Check if connection is still valid
                    conn.execute("SELECT 1")
                    self._connection_pool.put(conn)
                except:
                    # Connection is broken, close it
                    try:
                        conn.close()
                    except:
                        pass
                    with self._pool_lock:
                        if conn in self._all_connections:
                            self._all_connections.remove(conn)
    
    def close_all_connections(self):
        """Close all pooled connections"""
        with self._pool_lock:
            # Close all connections
            for conn in self._all_connections:
                try:
                    conn.close()
                except:
                    pass
            
            # Clear the pool
            while not self._connection_pool.empty():
                try:
                    self._connection_pool.get_nowait()
                except:
                    pass
            
            self._all_connections.clear()
    
    def _init_database(self):
        """Initialize SQLite database"""
        # Use a direct connection for initialization (not from pool)
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
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON generations(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_generation_type ON generations(generation_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON generations(model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_favorite ON generations(favorite)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viewed ON generations(viewed)")
            
            # Add composite indexes for common query patterns
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type_timestamp ON generations(generation_type, timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_timestamp ON generations(model_name, timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_favorite_timestamp ON generations(favorite, timestamp DESC) WHERE favorite = 1")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viewed_timestamp ON generations(viewed, timestamp DESC) WHERE viewed = 0")
            
            conn.commit()
    
    def add_record(self, record: GenerationRecord) -> bool:
        """Add a generation record to database"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO generations (
                        id, timestamp, generation_type, model_name,
                        prompt, negative_prompt, parameters, output_paths,
                        thumbnails, metadata, tags, favorite, viewed, viewed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    int(record.viewed),
                    record.viewed_at
                ))
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to add generation record: {e}")
            return False
    
    def add_records_batch(self, records: List[GenerationRecord]) -> Tuple[int, int]:
        """Add multiple generation records in a single transaction.
        
        Args:
            records: List of generation records to add
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0
        
        try:
            with self._get_connection() as conn:
                # Prepare batch data
                batch_data = []
                for record in records:
                    batch_data.append((
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
                        int(record.viewed),
                        record.viewed_at
                    ))
                
                # Execute batch insert
                conn.executemany("""
                    INSERT INTO generations (
                        id, timestamp, generation_type, model_name,
                        prompt, negative_prompt, parameters, output_paths,
                        thumbnails, metadata, tags, favorite, viewed, viewed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch_data)
                
                conn.commit()
                successful = len(records)
                
        except Exception as e:
            logger.error(f"Failed to add records in batch: {e}")
            failed = len(records)
            
        return successful, failed
    
    def get_records(
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
        """Get generation records with filtering"""
        with self._get_connection() as conn:
            # Build query
            query = "SELECT * FROM generations WHERE 1=1"
            params = []
            
            if generation_type:
                query += " AND generation_type = ?"
                params.append(generation_type)
                
            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)
                
            if search_query:
                query += " AND (prompt LIKE ? OR negative_prompt LIKE ? OR tags LIKE ?)"
                search_param = f"%{search_query}%"
                params.extend([search_param, search_param, search_param])
                
            if favorites_only:
                query += " AND favorite = 1"
                
            if unviewed_only:
                query += " AND viewed = 0"
                
            # Add sorting
            valid_sort_columns = ["timestamp", "generation_type", "model_name", "favorite", "viewed_at"]
            if sort_by in valid_sort_columns:
                query += f" ORDER BY {sort_by} {sort_order}"
            else:
                query += " ORDER BY timestamp DESC"
                
            # Add pagination
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute query
            cursor = conn.execute(query, params)
            
            # Convert to records
            records = []
            for row in cursor:
                try:
                    record = GenerationRecord(
                        id=row[0],
                        timestamp=row[1],
                        generation_type=row[2],
                        model_name=row[3],
                        prompt=row[4],
                        negative_prompt=row[5],
                        parameters=json.loads(row[6]),
                        output_paths=json.loads(row[7]),
                        thumbnails=json.loads(row[8]),
                        metadata=json.loads(row[9]),
                        tags=json.loads(row[10]) if row[10] else [],
                        favorite=bool(row[11]),
                        viewed=bool(row[12]),
                        viewed_at=row[13] if len(row) > 13 else None
                    )
                    records.append(record)
                except Exception as e:
                    logger.error(f"Failed to parse record: {e}")
                    
            return records
    
    def get_record(self, generation_id: str) -> Optional[GenerationRecord]:
        """Get a single record by ID - optimized to use primary key index directly"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM generations WHERE id = ? LIMIT 1",
                (generation_id,)
            )
            row = cursor.fetchone()
            
            if row:
                try:
                    return GenerationRecord(
                        id=row[0],
                        timestamp=row[1],
                        generation_type=row[2],
                        model_name=row[3],
                        prompt=row[4],
                        negative_prompt=row[5],
                        parameters=json.loads(row[6]),
                        output_paths=json.loads(row[7]),
                        thumbnails=json.loads(row[8]),
                        metadata=json.loads(row[9]),
                        tags=json.loads(row[10]) if row[10] else [],
                        favorite=bool(row[11]),
                        viewed=bool(row[12]) if len(row) > 12 else False,
                        viewed_at=row[13] if len(row) > 13 else None
                    )
                except Exception as e:
                    logger.error(f"Failed to parse record {generation_id}: {e}")
        
        return None
    
    def update_record(self, generation_id: str, **kwargs) -> bool:
        """Update a record's fields"""
        if not kwargs:
            return True
            
        try:
            with self._get_connection() as conn:
                # Build update query
                set_clauses = []
                params = []
                
                for field, value in kwargs.items():
                    if field in {"parameters", "output_paths", "thumbnails", "metadata", "tags"}:
                        set_clauses.append(f"{field} = ?")
                        params.append(json.dumps(value))
                    elif field in {"favorite", "viewed"}:
                        set_clauses.append(f"{field} = ?")
                        params.append(int(value))
                    elif field in {"timestamp", "viewed_at"}:
                        set_clauses.append(f"{field} = ?")
                        params.append(float(value) if value is not None else None)
                    else:
                        set_clauses.append(f"{field} = ?")
                        params.append(value)
                
                if not set_clauses:
                    return True
                    
                query = f"UPDATE generations SET {', '.join(set_clauses)} WHERE id = ?"
                params.append(generation_id)
                
                conn.execute(query, params)
                conn.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to update record {generation_id}: {e}")
            return False
    
    def update_records_batch(self, updates: List[Tuple[str, Dict[str, Any]]]) -> Tuple[int, int]:
        """Update multiple records in a single transaction.
        
        Args:
            updates: List of tuples (generation_id, update_dict)
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0
        
        try:
            with self._get_connection() as conn:
                # Group updates by the fields being updated to minimize queries
                updates_by_fields = {}
                
                for gen_id, update_dict in updates:
                    fields_key = tuple(sorted(update_dict.keys()))
                    if fields_key not in updates_by_fields:
                        updates_by_fields[fields_key] = []
                    updates_by_fields[fields_key].append((gen_id, update_dict))
                
                # Execute batch updates for each field combination
                for fields, batch_updates in updates_by_fields.items():
                    # Build query for this field combination
                    set_clauses = []
                    for field in fields:
                        set_clauses.append(f"{field} = ?")
                    
                    query = f"UPDATE generations SET {', '.join(set_clauses)} WHERE id = ?"
                    
                    # Prepare batch data
                    batch_data = []
                    for gen_id, update_dict in batch_updates:
                        params = []
                        for field in fields:
                            value = update_dict[field]
                            if field in {"parameters", "output_paths", "thumbnails", "metadata", "tags"}:
                                params.append(json.dumps(value))
                            elif field in {"favorite", "viewed"}:
                                params.append(int(value))
                            elif field in {"timestamp", "viewed_at"}:
                                params.append(float(value) if value is not None else None)
                            else:
                                params.append(value)
                        params.append(gen_id)
                        batch_data.append(params)
                    
                    # Execute batch update
                    cursor = conn.executemany(query, batch_data)
                    successful += cursor.rowcount
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update records in batch: {e}")
            failed = len(updates) - successful
            
        return successful, failed
    
    def delete_record(self, generation_id: str) -> bool:
        """Delete a record from database"""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM generations WHERE id = ?", (generation_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to delete record {generation_id}: {e}")
            return False
    
    def get_count(
        self,
        generation_type: Optional[str] = None,
        favorites_only: bool = False,
        unviewed_only: bool = False
    ) -> int:
        """Get count of records"""
        with self._get_connection() as conn:
            query = "SELECT COUNT(*) FROM generations WHERE 1=1"
            params = []
            
            if generation_type:
                query += " AND generation_type = ?"
                params.append(generation_type)
                
            if favorites_only:
                query += " AND favorite = 1"
                
            if unviewed_only:
                query += " AND viewed = 0"
                
            cursor = conn.execute(query, params)
            return cursor.fetchone()[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics - optimized with single query"""
        with self._get_connection() as conn:
            week_ago = time.time() - (7 * 24 * 60 * 60)
            
            # Single comprehensive query for all statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_generations,
                    SUM(CASE WHEN favorite = 1 THEN 1 ELSE 0 END) as favorites,
                    SUM(CASE WHEN viewed = 0 THEN 1 ELSE 0 END) as unviewed,
                    SUM(CASE WHEN timestamp > ? THEN 1 ELSE 0 END) as last_week,
                    COUNT(DISTINCT id) as unique_generations,
                    SUM(LENGTH(output_paths)) as total_path_chars,
                    SUM(LENGTH(thumbnails)) as total_thumb_chars
                FROM generations
            """, (week_ago,))
            
            row = cursor.fetchone()
            
            stats = {
                "total_generations": row[0] or 0,
                "favorites": row[1] or 0,
                "unviewed": row[2] or 0,
                "last_week": row[3] or 0,
                "storage_estimate": {
                    "unique_generations": row[4] or 0,
                    "estimated_mb": ((row[5] or 0) + (row[6] or 0)) / (1024 * 1024)
                }
            }
            
            # Group by queries - these still need separate queries but can be combined
            cursor = conn.execute("""
                SELECT 
                    'type_' || generation_type as key,
                    COUNT(*) as count
                FROM generations 
                GROUP BY generation_type
                UNION ALL
                SELECT 
                    'model_' || model_name as key,
                    COUNT(*) as count
                FROM generations 
                GROUP BY model_name
            """)
            
            by_type = {}
            by_model = {}
            
            for key, count in cursor.fetchall():
                if key.startswith('type_'):
                    by_type[key[5:]] = count  # Remove 'type_' prefix
                elif key.startswith('model_'):
                    by_model[key[6:]] = count  # Remove 'model_' prefix
            
            stats["by_type"] = by_type
            stats["by_model"] = by_model
            
            return stats
    
    def cleanup_old_records(self, cutoff_timestamp: float) -> int:
        """Delete records older than cutoff timestamp"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM generations WHERE timestamp < ? AND favorite = 0",
                    (cutoff_timestamp,)
                )
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return 0