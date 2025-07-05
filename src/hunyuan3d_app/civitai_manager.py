"""Civitai API integration for model discovery and download"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from urllib.parse import urlparse, parse_qs

import aiohttp
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class CivitaiModelInfo:
    """Information about a Civitai model"""
    id: int
    name: str
    type: str  # Checkpoint, LORA, TextualInversion, etc.
    base_model: str  # SD 1.5, SDXL, FLUX.1, etc.
    version_id: int
    version_name: str
    download_url: str
    file_size_mb: float
    images: List[str]
    description: str
    tags: List[str]
    creator: str
    stats: Dict[str, int]  # downloads, likes, etc.
    nsfw: bool = False
    early_access: bool = False
    
    @property
    def compatible_with_flux(self) -> bool:
        """Check if model is compatible with FLUX"""
        return self.base_model in ["FLUX.1", "FLUX.1-dev", "FLUX.1-schnell"]
    
    @property
    def compatible_with_sdxl(self) -> bool:
        """Check if model is compatible with SDXL"""
        return self.base_model in ["SDXL 1.0", "SDXL Turbo", "SDXL"]


class CivitaiManager:
    """Manages Civitai model discovery, search, and download"""
    
    API_BASE = "https://civitai.com/api/v1"
    MODELS_ENDPOINT = f"{API_BASE}/models"
    MODEL_VERSIONS_ENDPOINT = f"{API_BASE}/model-versions"
    
    SUPPORTED_TYPES = ["Checkpoint", "LORA", "TextualInversion", "ControlNet", "VAE"]
    SUPPORTED_BASE_MODELS = ["FLUX.1", "FLUX.1-dev", "FLUX.1-schnell", "SDXL 1.0", "SDXL", "SD 1.5"]
    
    def __init__(self, cache_dir: Path, api_key: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.session = None
        
        # Cache for model metadata
        self.metadata_cache_file = self.cache_dir / "civitai_metadata.json"
        self.metadata_cache = self._load_metadata_cache()
        
    def _load_metadata_cache(self) -> Dict:
        """Load cached metadata"""
        if self.metadata_cache_file.exists():
            try:
                with open(self.metadata_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata cache: {e}")
        return {}
    
    def _save_metadata_cache(self):
        """Save metadata cache"""
        try:
            with open(self.metadata_cache_file, 'w') as f:
                json.dump(self.metadata_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata cache: {e}")
    
    def search_models(
        self,
        query: Optional[str] = None,
        types: Optional[List[str]] = None,
        base_models: Optional[List[str]] = None,
        sort: str = "Most Downloaded",
        limit: int = 20,
        nsfw: bool = False,
        early_access: bool = False
    ) -> List[CivitaiModelInfo]:
        """Search for models on Civitai
        
        Args:
            query: Search query string
            types: Model types to filter (Checkpoint, LORA, etc.)
            base_models: Base models to filter (FLUX.1, SDXL, etc.)
            sort: Sort order (Most Downloaded, Newest, Most Liked)
            limit: Maximum number of results
            nsfw: Include NSFW content
            early_access: Include early access content
            
        Returns:
            List of model information
        """
        params = {
            "limit": limit,
            "sort": sort,
            "nsfw": nsfw,
            "earlyAccess": early_access
        }
        
        if query:
            params["query"] = query
            
        if types:
            # Filter to supported types
            filtered_types = [t for t in types if t in self.SUPPORTED_TYPES]
            if filtered_types:
                params["types"] = ",".join(filtered_types)
                
        if base_models:
            # Filter to supported base models
            filtered_bases = [b for b in base_models if b in self.SUPPORTED_BASE_MODELS]
            if filtered_bases:
                params["baseModels"] = ",".join(filtered_bases)
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        try:
            response = requests.get(
                self.MODELS_ENDPOINT,
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for item in data.get("items", []):
                try:
                    model_info = self._parse_model_data(item)
                    if model_info:
                        models.append(model_info)
                        # Cache metadata
                        self.metadata_cache[str(model_info.id)] = item
                except Exception as e:
                    logger.error(f"Error parsing model {item.get('id')}: {e}")
                    continue
                    
            self._save_metadata_cache()
            return models
            
        except Exception as e:
            logger.error(f"Error searching Civitai models: {e}")
            return []
    
    def _parse_model_data(self, data: Dict) -> Optional[CivitaiModelInfo]:
        """Parse model data from API response"""
        try:
            # Get the latest version
            versions = data.get("modelVersions", [])
            if not versions:
                return None
                
            latest_version = versions[0]
            
            # Find download URL
            files = latest_version.get("files", [])
            if not files:
                return None
                
            # Prefer primary file
            primary_file = next((f for f in files if f.get("primary", False)), files[0])
            
            # Get images
            images = []
            for img in latest_version.get("images", []):
                if img.get("url"):
                    images.append(img["url"])
                    
            # Calculate file size in MB
            file_size_bytes = primary_file.get("sizeKB", 0) * 1024
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            return CivitaiModelInfo(
                id=data["id"],
                name=data["name"],
                type=data["type"],
                base_model=latest_version.get("baseModel", "Unknown"),
                version_id=latest_version["id"],
                version_name=latest_version["name"],
                download_url=primary_file.get("downloadUrl", ""),
                file_size_mb=file_size_mb,
                images=images[:4],  # Limit to 4 images
                description=data.get("description", ""),
                tags=data.get("tags", []),
                creator=data.get("creator", {}).get("username", "Unknown"),
                stats={
                    "downloads": latest_version.get("stats", {}).get("downloadCount", 0),
                    "likes": data.get("stats", {}).get("thumbsUpCount", 0),
                    "rating": data.get("stats", {}).get("rating", 0)
                },
                nsfw=data.get("nsfw", False),
                early_access=latest_version.get("earlyAccessTimeFrame", 0) > 0
            )
        except Exception as e:
            logger.error(f"Error parsing model data: {e}")
            return None
    
    def get_model_by_id(self, model_id: int) -> Optional[CivitaiModelInfo]:
        """Get specific model information by ID"""
        # Check cache first
        if str(model_id) in self.metadata_cache:
            return self._parse_model_data(self.metadata_cache[str(model_id)])
            
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        try:
            response = requests.get(
                f"{self.MODELS_ENDPOINT}/{model_id}",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            model_info = self._parse_model_data(data)
            
            if model_info:
                # Cache metadata
                self.metadata_cache[str(model_id)] = data
                self._save_metadata_cache()
                
            return model_info
            
        except Exception as e:
            logger.error(f"Error getting model {model_id}: {e}")
            return None
    
    def get_model_from_url(self, url: str) -> Optional[CivitaiModelInfo]:
        """Extract model info from a Civitai URL"""
        try:
            parsed = urlparse(url)
            
            # Extract model ID from URL patterns:
            # https://civitai.com/models/123456
            # https://civitai.com/models/123456/model-name
            # https://civitai.com/models/123456?modelVersionId=789
            
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 2 and path_parts[0] == "models":
                model_id = int(path_parts[1])
                return self.get_model_by_id(model_id)
                
        except Exception as e:
            logger.error(f"Error parsing Civitai URL {url}: {e}")
            
        return None
    
    def download_model(
        self,
        model_info: CivitaiModelInfo,
        destination_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        use_aria2: bool = True
    ) -> Tuple[bool, str]:
        """Download a model from Civitai
        
        Args:
            model_info: Model information
            destination_dir: Where to save the model (defaults to cache_dir/type)
            progress_callback: Callback for progress updates
            use_aria2: Use aria2c for faster downloads if available
            
        Returns:
            Tuple of (success, file_path or error_message)
        """
        if not model_info.download_url:
            return False, "No download URL available"
            
        # Determine destination
        if destination_dir is None:
            type_dir = model_info.type.lower().replace(" ", "_")
            destination_dir = self.cache_dir / type_dir / model_info.base_model.replace(" ", "_")
            
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', model_info.name)
        filename = f"{safe_name}_{model_info.version_name}.safetensors"
        file_path = destination_dir / filename
        
        # Check if already downloaded
        if file_path.exists():
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if abs(file_size_mb - model_info.file_size_mb) < 1:  # Within 1MB tolerance
                logger.info(f"Model already downloaded: {file_path}")
                return True, str(file_path)
                
        # Add API key to download URL if available
        download_url = model_info.download_url
        if self.api_key and "token=" not in download_url:
            separator = "&" if "?" in download_url else "?"
            download_url += f"{separator}token={self.api_key}"
            
        # Try aria2c first if requested
        if use_aria2 and self._is_aria2_available():
            success, result = self._download_with_aria2(
                download_url, file_path, model_info.file_size_mb, progress_callback
            )
            if success:
                return True, result
            else:
                logger.warning(f"aria2c download failed, falling back to requests: {result}")
                
        # Fallback to requests
        return self._download_with_requests(
            download_url, file_path, model_info.file_size_mb, progress_callback
        )
    
    def _is_aria2_available(self) -> bool:
        """Check if aria2c is available"""
        try:
            import subprocess
            result = subprocess.run(["aria2c", "--version"], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    def _download_with_aria2(
        self,
        url: str,
        file_path: Path,
        expected_size_mb: float,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, str]:
        """Download using aria2c for faster multi-connection downloads"""
        import subprocess
        
        temp_path = file_path.with_suffix('.aria2')
        
        cmd = [
            "aria2c",
            "-x", "16",  # 16 connections
            "-s", "16",  # 16 splits
            "-k", "1M",  # 1MB chunk size
            "--file-allocation=none",
            "--continue=true",
            "--max-connection-per-server=16",
            "--min-split-size=1M",
            "--disable-ipv6=true",
            "--user-agent=CivitaiManager/1.0",
            "-d", str(file_path.parent),
            "-o", file_path.name,
            url
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output and progress_callback:
                    # Parse aria2c progress
                    match = re.search(r'\[#\w+\s+(\d+)%\]', output)
                    if match:
                        progress = int(match.group(1)) / 100.0
                        progress_callback(progress, f"Downloading: {progress*100:.1f}%")
                        
            if process.returncode == 0:
                return True, str(file_path)
            else:
                stderr = process.stderr.read()
                return False, f"aria2c failed: {stderr}"
                
        except Exception as e:
            return False, f"aria2c error: {str(e)}"
    
    def _download_with_requests(
        self,
        url: str,
        file_path: Path,
        expected_size_mb: float,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, str]:
        """Download using requests with progress tracking"""
        headers = {
            "User-Agent": "CivitaiManager/1.0"
        }
        
        try:
            # Support resumable downloads
            resume_pos = 0
            mode = 'wb'
            
            if file_path.exists():
                resume_pos = file_path.stat().st_size
                mode = 'ab'
                headers["Range"] = f"bytes={resume_pos}-"
                
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            if resume_pos > 0:
                total_size += resume_pos
                
            # Download with progress
            with open(file_path, mode) as f:
                downloaded = resume_pos
                chunk_size = 1024 * 1024  # 1MB chunks
                
                with tqdm(
                    total=total_size,
                    initial=resume_pos,
                    unit='B',
                    unit_scale=True,
                    desc=f"Downloading {file_path.name}"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
                            
                            if progress_callback:
                                progress = downloaded / total_size if total_size > 0 else 0
                                speed_mb = pbar.format_dict['rate'] / (1024 * 1024) if pbar.format_dict['rate'] else 0
                                progress_callback(
                                    progress,
                                    f"Downloading: {progress*100:.1f}% ({speed_mb:.1f} MB/s)"
                                )
                                
            # Verify size
            actual_size_mb = file_path.stat().st_size / (1024 * 1024)
            if abs(actual_size_mb - expected_size_mb) > 10:  # 10MB tolerance
                logger.warning(
                    f"Downloaded size ({actual_size_mb:.1f}MB) differs from "
                    f"expected ({expected_size_mb:.1f}MB)"
                )
                
            return True, str(file_path)
            
        except Exception as e:
            return False, f"Download error: {str(e)}"
    
    async def download_multiple_models(
        self,
        models: List[CivitaiModelInfo],
        destination_dir: Optional[Path] = None,
        max_concurrent: int = 3,
        progress_callback: Optional[Callable] = None
    ) -> Dict[int, Tuple[bool, str]]:
        """Download multiple models concurrently
        
        Args:
            models: List of models to download
            destination_dir: Where to save models
            max_concurrent: Maximum concurrent downloads
            progress_callback: Progress callback
            
        Returns:
            Dictionary mapping model ID to (success, path/error)
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}
        
        async def download_with_limit(model: CivitaiModelInfo):
            async with semaphore:
                # Run download in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                success, result = await loop.run_in_executor(
                    None,
                    self.download_model,
                    model,
                    destination_dir,
                    progress_callback
                )
                results[model.id] = (success, result)
                
        # Create tasks for all downloads
        tasks = [download_with_limit(model) for model in models]
        
        # Wait for all downloads to complete
        await asyncio.gather(*tasks)
        
        return results
    
    def get_recommended_models(
        self,
        base_model: str = "FLUX.1",
        model_type: str = "LORA",
        limit: int = 10
    ) -> List[CivitaiModelInfo]:
        """Get recommended models for a specific base model and type
        
        Args:
            base_model: Base model to filter by
            model_type: Type of model (LORA, Checkpoint, etc.)
            limit: Maximum number of results
            
        Returns:
            List of recommended models sorted by popularity
        """
        return self.search_models(
            types=[model_type],
            base_models=[base_model],
            sort="Most Downloaded",
            limit=limit,
            nsfw=False
        )
    
    def cleanup_cache(self, days_old: int = 30):
        """Clean up old cached files
        
        Args:
            days_old: Remove files older than this many days
        """
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        logger.info(f"Removed old cache file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing {file_path}: {e}")