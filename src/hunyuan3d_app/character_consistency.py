"""Character consistency module using IP-Adapter for universal character preservation"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import json
import time
from datetime import datetime
import uuid

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

logger = logging.getLogger(__name__)


@dataclass
class CharacterProfile:
    """A character profile with embeddings and metadata"""
    id: str
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Visual features
    face_embeddings: Optional[torch.Tensor] = None
    style_embeddings: Optional[torch.Tensor] = None
    full_embeddings: Optional[torch.Tensor] = None
    
    # Reference data
    reference_images: List[Path] = field(default_factory=list)
    anchor_frames: List[Dict[str, Any]] = field(default_factory=list)
    
    # Character attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    trigger_words: List[str] = field(default_factory=list)
    negative_triggers: List[str] = field(default_factory=list)
    
    # Control settings
    control_methods: List[str] = field(default_factory=lambda: ["ip_adapter", "controlnet"])
    consistency_strength: float = 0.8
    style_strength: float = 0.6
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "reference_images": [str(p) for p in self.reference_images],
            "attributes": self.attributes,
            "trigger_words": self.trigger_words,
            "negative_triggers": self.negative_triggers,
            "control_methods": self.control_methods,
            "consistency_strength": self.consistency_strength,
            "style_strength": self.style_strength
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embeddings_dir: Path) -> "CharacterProfile":
        """Load from dictionary"""
        profile = cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            reference_images=[Path(p) for p in data.get("reference_images", [])],
            attributes=data.get("attributes", {}),
            trigger_words=data.get("trigger_words", []),
            negative_triggers=data.get("negative_triggers", []),
            control_methods=data.get("control_methods", ["ip_adapter", "controlnet"]),
            consistency_strength=data.get("consistency_strength", 0.8),
            style_strength=data.get("style_strength", 0.6)
        )
        
        # Load embeddings if they exist
        embeddings_path = embeddings_dir / f"{profile.id}_embeddings.pt"
        if embeddings_path.exists():
            embeddings = torch.load(embeddings_path, map_location="cpu")
            profile.face_embeddings = embeddings.get("face")
            profile.style_embeddings = embeddings.get("style")
            profile.full_embeddings = embeddings.get("full")
            
        return profile


class CharacterConsistencyManager:
    """Manages character consistency across generations using IP-Adapter"""
    
    def __init__(
        self,
        profiles_dir: Optional[Path] = None,
        embeddings_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ):
        self.profiles_dir = profiles_dir or Path("./characters/profiles")
        self.embeddings_dir = embeddings_dir or Path("./characters/embeddings")
        self.cache_dir = cache_dir or Path("./cache/characters")
        
        # Create directories
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.clip_processor = None
        self.clip_model = None
        self.ip_adapter = None
        
        # Load existing profiles
        self.profiles: Dict[str, CharacterProfile] = self._load_profiles()
        
    def _load_profiles(self) -> Dict[str, CharacterProfile]:
        """Load all character profiles"""
        profiles = {}
        
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, "r") as f:
                    data = json.load(f)
                profile = CharacterProfile.from_dict(data, self.embeddings_dir)
                profiles[profile.id] = profile
            except Exception as e:
                logger.error(f"Failed to load profile {profile_file}: {e}")
                
        logger.info(f"Loaded {len(profiles)} character profiles")
        return profiles
        
    def initialize_models(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ) -> Tuple[bool, str]:
        """Initialize CLIP and IP-Adapter models
        
        Args:
            device: Device to load models on
            dtype: Model precision
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Initialize CLIP
            logger.info("Loading CLIP vision model...")
            self.clip_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=dtype
            ).to(device)
            
            # In a real implementation, we would load IP-Adapter here
            # For now, we'll use a placeholder
            self.ip_adapter = self._create_mock_ip_adapter(device, dtype)
            
            return True, "Character consistency models initialized"
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            return False, f"Failed to initialize: {str(e)}"
            
    def _create_mock_ip_adapter(self, device: str, dtype: torch.dtype) -> Any:
        """Create mock IP-Adapter for demonstration"""
        class MockIPAdapter:
            def __init__(self, device, dtype):
                self.device = device
                self.dtype = dtype
                
            def extract_features(self, images, feature_type="all"):
                # Mock feature extraction
                batch_size = len(images) if isinstance(images, list) else 1
                if feature_type == "face":
                    return torch.randn(batch_size, 512, dtype=dtype, device=device)
                elif feature_type == "style":
                    return torch.randn(batch_size, 768, dtype=dtype, device=device)
                else:
                    return torch.randn(batch_size, 1024, dtype=dtype, device=device)
                    
        return MockIPAdapter(device, dtype)
        
    def create_character(
        self,
        name: str,
        reference_images: List[Union[str, Path, Image.Image]],
        description: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        extract_style: bool = True,
        extract_face: bool = True
    ) -> Tuple[CharacterProfile, str]:
        """Create a new character profile from reference images
        
        Args:
            name: Character name
            reference_images: List of reference images
            description: Character description
            attributes: Additional attributes
            extract_style: Extract style embeddings
            extract_face: Extract face embeddings
            
        Returns:
            Tuple of (character profile, message)
        """
        try:
            # Generate unique ID
            character_id = str(uuid.uuid4())
            
            # Process reference images
            processed_images = []
            image_paths = []
            
            for img in reference_images:
                if isinstance(img, (str, Path)):
                    image = Image.open(img).convert("RGB")
                    image_paths.append(Path(img))
                else:
                    image = img.convert("RGB")
                    # Save image
                    img_path = self.cache_dir / f"{character_id}_{len(image_paths)}.png"
                    image.save(img_path)
                    image_paths.append(img_path)
                    
                processed_images.append(image)
                
            # Extract embeddings
            logger.info(f"Extracting embeddings for {name}...")
            
            embeddings = {}
            
            if self.clip_model and self.clip_processor:
                # Extract CLIP embeddings
                inputs = self.clip_processor(
                    images=processed_images,
                    return_tensors="pt"
                ).to(self.clip_model.device)
                
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    clip_embeddings = outputs.image_embeds
                    embeddings["clip"] = clip_embeddings.mean(dim=0)
                    
            if self.ip_adapter:
                # Extract IP-Adapter features
                if extract_face:
                    embeddings["face"] = self.ip_adapter.extract_features(
                        processed_images, "face"
                    ).mean(dim=0)
                    
                if extract_style:
                    embeddings["style"] = self.ip_adapter.extract_features(
                        processed_images, "style"
                    ).mean(dim=0)
                    
                embeddings["full"] = self.ip_adapter.extract_features(
                    processed_images, "all"
                ).mean(dim=0)
                
            # Generate anchor frames
            anchor_frames = self._generate_anchor_frames(processed_images)
            
            # Create character profile
            profile = CharacterProfile(
                id=character_id,
                name=name,
                description=description,
                face_embeddings=embeddings.get("face"),
                style_embeddings=embeddings.get("style"),
                full_embeddings=embeddings.get("full"),
                reference_images=image_paths,
                anchor_frames=anchor_frames,
                attributes=attributes or {}
            )
            
            # Auto-generate trigger words
            profile.trigger_words = self._generate_trigger_words(name, attributes)
            
            # Save profile
            self.save_character(profile)
            
            # Add to loaded profiles
            self.profiles[character_id] = profile
            
            return profile, f"Character '{name}' created successfully"
            
        except Exception as e:
            logger.error(f"Failed to create character: {e}")
            return None, f"Failed to create character: {str(e)}"
            
    def _generate_anchor_frames(
        self,
        images: List[Image.Image]
    ) -> List[Dict[str, Any]]:
        """Generate anchor frames for consistent generation"""
        anchor_frames = []
        
        for i, img in enumerate(images):
            # Extract key features
            # In a real implementation, this would extract pose, expression, etc.
            anchor = {
                "index": i,
                "image_size": img.size,
                "dominant_colors": self._extract_dominant_colors(img),
                "brightness": self._calculate_brightness(img),
                # These would be extracted using specialized models:
                # "pose": self._extract_pose(img),
                # "expression": self._extract_expression(img),
                # "lighting": self._extract_lighting(img)
            }
            anchor_frames.append(anchor)
            
        return anchor_frames
        
    def _extract_dominant_colors(
        self,
        image: Image.Image,
        num_colors: int = 5
    ) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image"""
        # Simplified implementation
        img_array = np.array(image.resize((50, 50)))
        pixels = img_array.reshape(-1, 3)
        
        # Use k-means to find dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        return [tuple(color) for color in colors]
        
    def _calculate_brightness(self, image: Image.Image) -> float:
        """Calculate average brightness"""
        grayscale = image.convert('L')
        histogram = grayscale.histogram()
        pixels = sum(histogram)
        brightness = sum(i * histogram[i] for i in range(256)) / pixels
        return brightness / 255.0
        
    def _generate_trigger_words(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate trigger words for the character"""
        triggers = [name.lower(), f"{name}character"]
        
        if attributes:
            # Add attribute-based triggers
            if "gender" in attributes:
                triggers.append(attributes["gender"])
            if "age" in attributes:
                triggers.append(f"{attributes['age']}yo")
            if "hair_color" in attributes:
                triggers.append(f"{attributes['hair_color']}hair")
            if "style" in attributes:
                triggers.extend(attributes["style"].split())
                
        return triggers
        
    def save_character(self, profile: CharacterProfile):
        """Save character profile and embeddings"""
        # Save profile JSON
        profile_path = self.profiles_dir / f"{profile.id}.json"
        with open(profile_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
            
        # Save embeddings
        embeddings_path = self.embeddings_dir / f"{profile.id}_embeddings.pt"
        embeddings = {}
        if profile.face_embeddings is not None:
            embeddings["face"] = profile.face_embeddings
        if profile.style_embeddings is not None:
            embeddings["style"] = profile.style_embeddings
        if profile.full_embeddings is not None:
            embeddings["full"] = profile.full_embeddings
            
        if embeddings:
            torch.save(embeddings, embeddings_path)
            
    def get_character(self, character_id: str) -> Optional[CharacterProfile]:
        """Get a character profile by ID"""
        return self.profiles.get(character_id)
        
    def list_characters(self) -> List[CharacterProfile]:
        """List all character profiles"""
        return list(self.profiles.values())
        
    def search_characters(
        self,
        query: str = "",
        attributes: Optional[Dict[str, Any]] = None
    ) -> List[CharacterProfile]:
        """Search for characters"""
        results = []
        
        for profile in self.profiles.values():
            # Text search
            if query:
                query_lower = query.lower()
                if (query_lower in profile.name.lower() or
                    query_lower in profile.description.lower() or
                    any(query_lower in word for word in profile.trigger_words)):
                    results.append(profile)
                    continue
                    
            # Attribute search
            if attributes:
                match = True
                for key, value in attributes.items():
                    if profile.attributes.get(key) != value:
                        match = False
                        break
                if match:
                    results.append(profile)
                    
        return results
        
    def apply_character_consistency(
        self,
        pipeline: Any,
        character_id: str,
        prompt: str,
        strength: Optional[float] = None,
        style_weight: Optional[float] = None
    ) -> Tuple[Any, str]:
        """Apply character consistency to a pipeline
        
        Args:
            pipeline: Diffusion pipeline
            character_id: Character ID
            prompt: Generation prompt
            strength: Consistency strength override
            style_weight: Style weight override
            
        Returns:
            Tuple of (modified pipeline, modified prompt)
        """
        character = self.get_character(character_id)
        if not character:
            logger.error(f"Character {character_id} not found")
            return pipeline, prompt
            
        # Add trigger words to prompt
        trigger_words = " ".join(character.trigger_words)
        modified_prompt = f"{trigger_words}, {prompt}"
        
        # Apply IP-Adapter embeddings
        if hasattr(pipeline, "set_ip_adapter_scale"):
            scale = strength or character.consistency_strength
            pipeline.set_ip_adapter_scale(scale)
            
        if character.full_embeddings is not None:
            # In real implementation, this would set the embeddings
            # pipeline.set_ip_adapter_image_embeds(character.full_embeddings)
            pass
            
        # Apply style if available
        if character.style_embeddings is not None and style_weight:
            # This would apply style embeddings
            pass
            
        return pipeline, modified_prompt
        
    def inject_character_tokens(
        self,
        prompt: str,
        character: CharacterProfile
    ) -> str:
        """Inject character tokens into prompt"""
        # Add trigger words
        triggers = " ".join(character.trigger_words)
        
        # Add negative triggers to negative prompt if needed
        negative = " ".join(character.negative_triggers) if character.negative_triggers else ""
        
        # Combine with original prompt
        enhanced_prompt = f"{triggers}, {prompt}"
        
        return enhanced_prompt
        
    def create_character_lora(
        self,
        character_id: str,
        output_path: Path,
        training_images: Optional[List[Path]] = None,
        steps: int = 1000
    ) -> Tuple[bool, str]:
        """Create a LoRA for a specific character
        
        Args:
            character_id: Character ID
            output_path: Output path for LoRA
            training_images: Additional training images
            steps: Training steps
            
        Returns:
            Tuple of (success, message)
        """
        character = self.get_character(character_id)
        if not character:
            return False, f"Character {character_id} not found"
            
        # This would implement LoRA training
        # For now, it's a placeholder
        logger.info(f"Would train LoRA for {character.name} with {steps} steps")
        
        return False, "LoRA training not yet implemented"
        
    def blend_characters(
        self,
        character_ids: List[str],
        weights: Optional[List[float]] = None,
        name: str = "Blended Character"
    ) -> Tuple[Optional[CharacterProfile], str]:
        """Blend multiple characters into one
        
        Args:
            character_ids: List of character IDs to blend
            weights: Weights for each character
            name: Name for blended character
            
        Returns:
            Tuple of (blended profile, message)
        """
        if not character_ids:
            return None, "No characters provided"
            
        characters = [self.get_character(cid) for cid in character_ids]
        characters = [c for c in characters if c is not None]
        
        if not characters:
            return None, "No valid characters found"
            
        if weights is None:
            weights = [1.0 / len(characters)] * len(characters)
        elif len(weights) != len(characters):
            return None, "Number of weights must match number of characters"
            
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Blend embeddings
        blended_embeddings = {}
        
        for emb_type in ["face", "style", "full"]:
            embeddings = []
            for char, weight in zip(characters, weights):
                emb = getattr(char, f"{emb_type}_embeddings")
                if emb is not None:
                    embeddings.append(emb * weight)
                    
            if embeddings:
                blended_embeddings[emb_type] = sum(embeddings)
                
        # Create blended character
        blended_id = str(uuid.uuid4())
        blended = CharacterProfile(
            id=blended_id,
            name=name,
            description=f"Blended from: {', '.join(c.name for c in characters)}",
            face_embeddings=blended_embeddings.get("face"),
            style_embeddings=blended_embeddings.get("style"),
            full_embeddings=blended_embeddings.get("full"),
            attributes={"blended": True, "source_characters": character_ids}
        )
        
        # Merge trigger words
        all_triggers = []
        for char in characters:
            all_triggers.extend(char.trigger_words)
        blended.trigger_words = list(set(all_triggers))
        
        # Save blended character
        self.save_character(blended)
        self.profiles[blended_id] = blended
        
        return blended, f"Created blended character: {name}"
        
    def export_character(
        self,
        character_id: str,
        output_path: Path,
        include_images: bool = True
    ) -> Tuple[bool, str]:
        """Export a character for sharing
        
        Args:
            character_id: Character ID
            output_path: Output file path
            include_images: Include reference images
            
        Returns:
            Tuple of (success, message)
        """
        import zipfile
        
        character = self.get_character(character_id)
        if not character:
            return False, f"Character {character_id} not found"
            
        try:
            with zipfile.ZipFile(output_path, 'w') as zf:
                # Add profile JSON
                profile_data = character.to_dict()
                zf.writestr(f"{character_id}/profile.json", json.dumps(profile_data, indent=2))
                
                # Add embeddings
                embeddings_path = self.embeddings_dir / f"{character_id}_embeddings.pt"
                if embeddings_path.exists():
                    zf.write(embeddings_path, f"{character_id}/embeddings.pt")
                    
                # Add reference images
                if include_images:
                    for i, img_path in enumerate(character.reference_images):
                        if img_path.exists():
                            zf.write(img_path, f"{character_id}/images/{img_path.name}")
                            
            return True, f"Exported character to {output_path}"
            
        except Exception as e:
            logger.error(f"Failed to export character: {e}")
            return False, f"Export failed: {str(e)}"