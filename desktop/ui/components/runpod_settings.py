"""
RunPod settings component for desktop app
"""

from typing import Optional, Callable
from nicegui import ui
import os
from pathlib import Path

class RunPodSettings:
    """RunPod configuration settings component"""
    
    def __init__(self, on_save: Optional[Callable] = None):
        self.on_save = on_save
        self.config_file = Path.home() / ".neuralforge" / "runpod_config.json"
        self.api_key = ""
        self.enabled = False
        self.execution_mode = "auto"
        self._load_config()
        
    def _load_config(self):
        """Load configuration from file"""
        if self.config_file.exists():
            import json
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.api_key = config.get("api_key", "")
                    self.enabled = config.get("enabled", False)
                    self.execution_mode = config.get("execution_mode", "auto")
            except:
                pass
                
    def _save_config(self):
        """Save configuration to file"""
        import json
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            "api_key": self.api_key,
            "enabled": self.enabled,
            "execution_mode": self.execution_mode
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        if self.on_save:
            self.on_save(config)
            
        ui.notify('RunPod settings saved!', type='positive')
        
    def render(self):
        """Render the RunPod settings UI"""
        with ui.column().classes('w-full gap-4'):
            ui.label('RunPod Configuration').classes('text-xl font-bold mb-2')
            
            # Enable/Disable toggle
            with ui.row().classes('items-center gap-4 mb-4'):
                self.enabled_switch = ui.switch(
                    'Enable RunPod GPU',
                    value=self.enabled
                ).on('update:model-value', lambda e: setattr(self, 'enabled', e.args))
                
                ui.label('Use cloud GPUs for heavy workloads').classes('text-sm text-gray-400')
                
            # API Key input
            with ui.column().classes('w-full gap-2').bind_visibility_from(self.enabled_switch, 'value'):
                ui.label('API Key').classes('text-sm font-medium')
                self.api_key_input = ui.input(
                    placeholder='Enter your RunPod API key',
                    value=self.api_key,
                    password=True
                ).classes('w-full').on('update:model-value', lambda e: setattr(self, 'api_key', e.args))
                
                with ui.row().classes('gap-2'):
                    ui.link(
                        'Get API Key',
                        'https://www.runpod.io/console/api-keys',
                        new_tab=True
                    ).classes('text-sm text-purple-400')
                    
                    ui.link(
                        'Pricing Info',
                        'https://www.runpod.io/pricing',
                        new_tab=True
                    ).classes('text-sm text-purple-400')
                    
            # Execution mode
            with ui.column().classes('w-full gap-2 mt-4').bind_visibility_from(self.enabled_switch, 'value'):
                ui.label('Execution Mode').classes('text-sm font-medium')
                self.mode_select = ui.select(
                    options={
                        'auto': 'Auto (Recommended)',
                        'local': 'Always Local',
                        'serverless': 'Always Serverless'
                    },
                    value=self.execution_mode
                ).classes('w-full').on('update:model-value', lambda e: setattr(self, 'execution_mode', e.args))
                
                # Mode descriptions
                with ui.column().classes('mt-2 p-3 bg-gray-800 rounded'):
                    ui.label('• Auto: Use local GPU when available, serverless for heavy tasks').classes('text-xs')
                    ui.label('• Local: Always use local GPU (may fail if insufficient VRAM)').classes('text-xs')
                    ui.label('• Serverless: Always use RunPod (incurs costs)').classes('text-xs')
                    
            # Cost estimates
            with ui.column().classes('w-full mt-4').bind_visibility_from(self.enabled_switch, 'value'):
                ui.label('Estimated Costs').classes('text-sm font-medium mb-2')
                
                with ui.grid(columns=2).classes('w-full gap-2'):
                    # Image generation
                    ui.label('Image Generation:').classes('text-sm')
                    ui.label('~$0.005 per image').classes('text-sm text-gray-400')
                    
                    # 3D conversion
                    ui.label('3D Conversion:').classes('text-sm')
                    ui.label('~$0.045 per model').classes('text-sm text-gray-400')
                    
                    # Face swap
                    ui.label('Face Swap:').classes('text-sm')
                    ui.label('~$0.008 per swap').classes('text-sm text-gray-400')
                    
                    # Video generation
                    ui.label('Video Generation:').classes('text-sm')
                    ui.label('~$0.030 per second').classes('text-sm text-gray-400')
                    
            # GPU availability status
            with ui.column().classes('w-full mt-4').bind_visibility_from(self.enabled_switch, 'value'):
                ui.label('GPU Availability').classes('text-sm font-medium mb-2')
                
                # This would be populated with actual data
                self.gpu_status = ui.column().classes('w-full gap-1')
                self._update_gpu_status()
                
            # Save button
            ui.button(
                'Save Settings',
                icon='save',
                on_click=self._save_config
            ).props('unelevated').classes('mt-4')
            
    def _update_gpu_status(self):
        """Update GPU availability status"""
        self.gpu_status.clear()
        
        with self.gpu_status:
            # Mock data - would be real API call
            gpus = [
                {"name": "RTX 3090", "available": True, "cost": "$0.25/hr"},
                {"name": "RTX 4090", "available": True, "cost": "$0.44/hr"},
                {"name": "A100 40GB", "available": True, "cost": "$0.90/hr"},
                {"name": "A100 80GB", "available": False, "cost": "$1.50/hr"}
            ]
            
            for gpu in gpus:
                with ui.row().classes('items-center gap-2'):
                    if gpu["available"]:
                        ui.icon('check_circle', size='sm').classes('text-green-500')
                    else:
                        ui.icon('cancel', size='sm').classes('text-red-500')
                        
                    ui.label(f"{gpu['name']} - {gpu['cost']}").classes('text-xs')
                    
    def get_config(self) -> dict:
        """Get current configuration"""
        return {
            "api_key": self.api_key,
            "enabled": self.enabled,
            "execution_mode": self.execution_mode
        }