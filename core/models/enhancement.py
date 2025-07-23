"""
Models for prompt enhancement fields and configuration
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union, List
from enum import Enum

class ModelType(str, Enum):
    """Supported model types for enhancement"""
    FLUX_1_DEV = "flux_1_dev"
    FLUX_1_SCHNELL = "flux_1_schnell"
    HUNYUAN_3D_21 = "hunyuan_3d_21"
    HUNYUAN_3D_20 = "hunyuan_3d_20"
    HUNYUAN_3D_MINI = "hunyuan_3d_mini"
    HI3DGEN = "hi3dgen"
    SPARC3D = "sparc3d"
    SDXL = "sdxl"
    SD15 = "sd15"

class FieldType(str, Enum):
    """UI field types"""
    DROPDOWN = "dropdown"
    MULTI_CHECKBOX = "multi_checkbox"
    SLIDER = "slider"
    TEXT = "text"

class EnhancementField(BaseModel):
    """Individual enhancement field configuration"""
    label: str = Field(..., description="Display label with emoji")
    type: FieldType = Field(FieldType.DROPDOWN, description="Field type")
    options: Optional[Dict[str, str]] = Field(None, description="Option value to description mapping")
    min_value: Optional[float] = Field(None, description="Minimum value for sliders")
    max_value: Optional[float] = Field(None, description="Maximum value for sliders")
    default_value: Optional[Union[str, float, List[str]]] = Field(None, description="Default value")
    description: Optional[str] = Field(None, description="Field description/tooltip")

class EnhancementFields(BaseModel):
    """Collection of enhancement fields for a model type"""
    model_type: ModelType = Field(..., description="Model type these fields apply to")
    fields: Dict[str, EnhancementField] = Field(..., description="Field ID to field configuration")
    
    def get_field_values(self, selections: Dict[str, Any]) -> Dict[str, str]:
        """Convert user selections to prompt additions"""
        prompt_additions = {}
        
        for field_id, value in selections.items():
            if field_id not in self.fields:
                continue
                
            field = self.fields[field_id]
            
            if field.type == FieldType.DROPDOWN and field.options and value in field.options:
                prompt_additions[field_id] = field.options[value]
            elif field.type == FieldType.MULTI_CHECKBOX and field.options:
                selected_prompts = []
                for option_key in value:
                    if option_key in field.options:
                        selected_prompts.append(field.options[option_key])
                if selected_prompts:
                    prompt_additions[field_id] = ", ".join(selected_prompts)
            elif field.type in [FieldType.SLIDER, FieldType.TEXT]:
                prompt_additions[field_id] = str(value)
                
        return prompt_additions

class EnhancementTemplate(BaseModel):
    """LLM prompt template for enhancement"""
    model_type: ModelType = Field(..., description="Model type this template is for")
    system_prompt: str = Field(..., description="System prompt for LLM")
    user_prompt_template: str = Field(..., description="User prompt template with {user_prompt} placeholder")
    
class EnhancementConfig(BaseModel):
    """Complete enhancement configuration"""
    fields: Dict[ModelType, EnhancementFields] = Field(..., description="Fields per model type")
    templates: Dict[ModelType, EnhancementTemplate] = Field(..., description="LLM templates per model type")
    llm_model: str = Field("mistral:latest", description="Ollama model to use")
    cache_enhanced_prompts: bool = Field(True, description="Whether to cache enhanced prompts")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")