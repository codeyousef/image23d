"""Export and import functionality for generation history"""

import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .models import GenerationRecord

logger = logging.getLogger(__name__)


class ExportImportManager:
    """Manages export and import of generation history"""
    
    @staticmethod
    def export_history(
        records: List[GenerationRecord],
        output_path: Path,
        include_files: bool = False,
        base_output_dir: Optional[Path] = None
    ) -> bool:
        """Export history to JSON or ZIP file
        
        Args:
            records: List of generation records to export
            output_path: Output file path
            include_files: Whether to include output files
            base_output_dir: Base directory for output files
            
        Returns:
            Success status
        """
        try:
            output_path = Path(output_path)
            
            # Prepare export data
            export_data = {
                "version": "1.0",
                "export_date": datetime.now().isoformat(),
                "total_records": len(records),
                "records": [record.to_dict() for record in records]
            }
            
            if include_files and output_path.suffix.lower() == '.zip':
                # Export as ZIP with files
                return ExportImportManager._export_as_zip(
                    export_data, 
                    records, 
                    output_path,
                    base_output_dir
                )
            else:
                # Export as JSON only
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Exported {len(records)} records to {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to export history: {e}")
            return False
    
    @staticmethod
    def _export_as_zip(
        export_data: Dict[str, Any],
        records: List[GenerationRecord],
        output_path: Path,
        base_output_dir: Optional[Path]
    ) -> bool:
        """Export as ZIP file with included files"""
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Write metadata
                zf.writestr('history.json', json.dumps(export_data, indent=2))
                
                # Collect and copy files
                copied_files = set()
                file_mapping = {}
                
                for record in records:
                    # Copy output files
                    for output_path_str in record.output_paths:
                        output_file = Path(output_path_str)
                        
                        if output_file.exists() and str(output_file) not in copied_files:
                            # Create archive path
                            if base_output_dir and output_file.is_relative_to(base_output_dir):
                                archive_path = output_file.relative_to(base_output_dir)
                            else:
                                archive_path = Path(record.generation_type) / record.id / output_file.name
                            
                            zf.write(output_file, str(archive_path))
                            copied_files.add(str(output_file))
                            file_mapping[str(output_file)] = str(archive_path)
                    
                    # Copy thumbnails
                    for thumb_path_str in record.thumbnails:
                        thumb_file = Path(thumb_path_str)
                        
                        if thumb_file.exists() and str(thumb_file) not in copied_files:
                            archive_path = Path('thumbnails') / thumb_file.name
                            zf.write(thumb_file, str(archive_path))
                            copied_files.add(str(thumb_file))
                            file_mapping[str(thumb_file)] = str(archive_path)
                
                # Write file mapping
                zf.writestr('file_mapping.json', json.dumps(file_mapping, indent=2))
                
                logger.info(f"Exported {len(records)} records with {len(copied_files)} files to {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to export as ZIP: {e}")
            return False
    
    @staticmethod
    def import_history(
        import_path: Path,
        output_dir: Optional[Path] = None,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """Import history from JSON or ZIP file
        
        Args:
            import_path: Path to import file
            output_dir: Directory to extract files to
            skip_existing: Skip records that already exist
            
        Returns:
            Import results
        """
        results = {
            "success": False,
            "imported": 0,
            "skipped": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            import_path = Path(import_path)
            
            if not import_path.exists():
                results["errors"].append(f"Import file not found: {import_path}")
                return results
            
            if import_path.suffix.lower() == '.zip':
                # Import from ZIP
                return ExportImportManager._import_from_zip(
                    import_path, 
                    output_dir, 
                    skip_existing
                )
            else:
                # Import from JSON
                with open(import_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate version
                if data.get('version') != '1.0':
                    results["errors"].append(f"Unsupported version: {data.get('version')}")
                    return results
                
                # Parse records
                records = []
                for record_data in data.get('records', []):
                    try:
                        record = GenerationRecord.from_dict(record_data)
                        records.append(record)
                    except Exception as e:
                        results["failed"] += 1
                        results["errors"].append(f"Failed to parse record: {e}")
                
                results["success"] = True
                results["imported"] = len(records)
                results["records"] = records
                
                return results
                
        except Exception as e:
            results["errors"].append(f"Import failed: {e}")
            return results
    
    @staticmethod
    def _import_from_zip(
        import_path: Path,
        output_dir: Optional[Path],
        skip_existing: bool
    ) -> Dict[str, Any]:
        """Import from ZIP file"""
        results = {
            "success": False,
            "imported": 0,
            "skipped": 0,
            "failed": 0,
            "extracted_files": 0,
            "errors": []
        }
        
        try:
            with zipfile.ZipFile(import_path, 'r') as zf:
                # Read metadata
                if 'history.json' not in zf.namelist():
                    results["errors"].append("No history.json found in ZIP")
                    return results
                
                with zf.open('history.json') as f:
                    data = json.load(f)
                
                # Read file mapping if available
                file_mapping = {}
                if 'file_mapping.json' in zf.namelist():
                    with zf.open('file_mapping.json') as f:
                        file_mapping = json.load(f)
                
                # Extract files if output directory provided
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Extract all files except metadata
                    for file_info in zf.filelist:
                        if file_info.filename not in ['history.json', 'file_mapping.json']:
                            try:
                                zf.extract(file_info, output_dir)
                                results["extracted_files"] += 1
                            except Exception as e:
                                results["errors"].append(f"Failed to extract {file_info.filename}: {e}")
                
                # Process records
                records = []
                reverse_mapping = {v: k for k, v in file_mapping.items()}
                
                for record_data in data.get('records', []):
                    try:
                        # Update paths if files were extracted
                        if output_dir and file_mapping:
                            # Update output paths
                            updated_paths = []
                            for path in record_data.get('output_paths', []):
                                if path in file_mapping:
                                    new_path = output_dir / file_mapping[path]
                                    updated_paths.append(str(new_path))
                                else:
                                    updated_paths.append(path)
                            record_data['output_paths'] = updated_paths
                            
                            # Update thumbnail paths
                            updated_thumbs = []
                            for path in record_data.get('thumbnails', []):
                                if path in file_mapping:
                                    new_path = output_dir / file_mapping[path]
                                    updated_thumbs.append(str(new_path))
                                else:
                                    updated_thumbs.append(path)
                            record_data['thumbnails'] = updated_thumbs
                        
                        record = GenerationRecord.from_dict(record_data)
                        records.append(record)
                        results["imported"] += 1
                        
                    except Exception as e:
                        results["failed"] += 1
                        results["errors"].append(f"Failed to process record: {e}")
                
                results["success"] = True
                results["records"] = records
                
                return results
                
        except Exception as e:
            results["errors"].append(f"Failed to import from ZIP: {e}")
            return results