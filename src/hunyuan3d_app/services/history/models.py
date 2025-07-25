"""Data models for generation history"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


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