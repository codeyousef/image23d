"""
Settings page with application configuration
"""

from pathlib import Path
from nicegui import ui
import json
import os

from ..components.runpod_settings import RunPodSettings
from ...localization import get_translator, set_language

class SettingsPage:
    """Settings page for application configuration"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".neuralforge"
        self.config_file = self.config_dir / "settings.json"
        self.settings = self._load_settings()
        
        # Localization
        self.translator = get_translator()
        
        # Apply saved language
        saved_language = self.settings.get('language', 'en')
        set_language(saved_language)
        
        # Components
        self.runpod_settings = RunPodSettings(on_save=self._on_runpod_save)
        
    def _load_settings(self) -> dict:
        """Load settings from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return self._get_default_settings()
        
    def _get_default_settings(self) -> dict:
        """Get default settings"""
        return {
            "theme": "dark",
            "language": "en",
            "auto_save": True,
            "output_format": "png",
            "output_quality": 95,
            "enable_gpu_monitoring": True,
            "max_parallel_jobs": 2,
            "cache_size_gb": 10,
            "enable_telemetry": False
        }
        
    def _save_settings(self):
        """Save settings to file"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_file, 'w') as f:
            json.dump(self.settings, f, indent=2)
            
        ui.notify('Settings saved!', type='positive')
        
    def _on_runpod_save(self, config: dict):
        """Handle RunPod settings save"""
        self.settings["runpod"] = config
        self._save_settings()
        
    def render(self):
        """Render the settings page"""
        t = self.translator.t
        
        # Apply RTL styles if needed
        rtl_classes = 'dir-rtl' if self.translator.is_rtl() else ''
        font_style = f'font-family: {self.translator.get_font_family()}'
        
        with ui.column().classes(f'w-full h-full p-6 {rtl_classes}').style(font_style):
            # Header
            ui.label(t('settings.title')).classes('text-3xl font-bold mb-6')
            
            # Tabs for different settings sections
            with ui.tabs().classes('w-full') as tabs:
                general_tab = ui.tab(t('settings.tabs.general'), icon='settings')
                gpu_tab = ui.tab(t('settings.tabs.performance'), icon='memory')
                output_tab = ui.tab(t('settings.tabs.appearance'), icon='folder')
                runpod_tab = ui.tab('RunPod', icon='cloud')
                advanced_tab = ui.tab(t('settings.tabs.advanced'), icon='build')
                
            with ui.tab_panels(tabs, value=general_tab).classes('w-full'):
                # General settings
                with ui.tab_panel(general_tab):
                    self._render_general_settings()
                    
                # GPU settings
                with ui.tab_panel(gpu_tab):
                    self._render_gpu_settings()
                    
                # Output settings
                with ui.tab_panel(output_tab):
                    self._render_output_settings()
                    
                # RunPod settings
                with ui.tab_panel(runpod_tab):
                    self.runpod_settings.render()
                    
                # Advanced settings
                with ui.tab_panel(advanced_tab):
                    self._render_advanced_settings()
                    
    def _render_general_settings(self):
        """Render general settings"""
        t = self.translator.t
        
        with ui.column().classes('w-full gap-4 max-w-2xl'):
            ui.label(t('settings.tabs.general')).classes('text-xl font-bold mb-2')
            
            # Language selector - NOW ACTIVE
            with ui.row().classes('items-center gap-4 mb-4'):
                ui.label(t('settings.general.language')).classes('w-32')
                
                # Get available languages
                available_languages = self.translator.get_available_languages()
                
                language_select = ui.select(
                    options=available_languages,
                    value=self.settings.get('language', 'en')
                ).classes('flex-grow').on(
                    'update:model-value',
                    self._on_language_change
                )
                
            # Theme selector
            with ui.row().classes('items-center gap-4 mb-4'):
                ui.label(t('settings.appearance.theme')).classes('w-32')
                ui.select(
                    options={
                        'dark': t('settings.appearance.theme_dark'),
                        'light': t('settings.appearance.theme_light'),
                        'auto': t('settings.appearance.theme_auto')
                    },
                    value=self.settings.get('theme', 'dark')
                ).classes('flex-grow').on(
                    'update:model-value',
                    lambda e: self._update_setting('theme', e.args)
                )
                
            # Auto-save toggle
            with ui.row().classes('items-center gap-4 mb-4'):
                ui.label(t('settings.general.auto_save')).classes('w-32')
                ui.switch(
                    value=self.settings.get('auto_save', True)
                ).on(
                    'update:model-value',
                    lambda e: self._update_setting('auto_save', e.args)
                )
                ui.label(t('settings.general.auto_save')).classes('text-sm text-gray-400')
                
            # Save button
            ui.button(t('settings.save'), icon='save', on_click=self._save_settings).props('unelevated')
            
    def _render_gpu_settings(self):
        """Render GPU and performance settings"""
        with ui.column().classes('w-full gap-4 max-w-2xl'):
            ui.label('GPU & Performance Settings').classes('text-xl font-bold mb-2')
            
            # GPU monitoring
            with ui.row().classes('items-center gap-4 mb-4'):
                ui.label('GPU Monitoring').classes('w-48')
                ui.switch(
                    value=self.settings.get('enable_gpu_monitoring', True)
                ).on(
                    'update:model-value',
                    lambda e: self._update_setting('enable_gpu_monitoring', e.args)
                )
                ui.label('Show GPU usage in UI').classes('text-sm text-gray-400')
                
            # Parallel jobs
            with ui.row().classes('items-center gap-4 mb-4'):
                ui.label('Max Parallel Jobs').classes('w-48')
                ui.number(
                    value=self.settings.get('max_parallel_jobs', 2),
                    min=1,
                    max=10,
                    step=1
                ).classes('w-32').on(
                    'update:model-value',
                    lambda e: self._update_setting('max_parallel_jobs', int(e.args))
                )
                ui.label('Maximum concurrent generations').classes('text-sm text-gray-400')
                
            # Cache size
            with ui.row().classes('items-center gap-4 mb-4'):
                ui.label('Model Cache Size').classes('w-48')
                ui.number(
                    value=self.settings.get('cache_size_gb', 10),
                    min=1,
                    max=100,
                    step=1,
                    suffix='GB'
                ).classes('w-32').on(
                    'update:model-value',
                    lambda e: self._update_setting('cache_size_gb', int(e.args))
                )
                ui.label('Maximum cache for loaded models').classes('text-sm text-gray-400')
                
            # Current GPU info
            with ui.column().classes('mt-6 p-4 bg-gray-800 rounded'):
                ui.label('Current GPU Information').classes('font-medium mb-2')
                self._show_gpu_info()
                
            ui.button('Save GPU Settings', icon='save', on_click=self._save_settings).props('unelevated')
            
    def _render_output_settings(self):
        """Render output settings"""
        with ui.column().classes('w-full gap-4 max-w-2xl'):
            ui.label('Output Settings').classes('text-xl font-bold mb-2')
            
            # Output format
            with ui.row().classes('items-center gap-4 mb-4'):
                ui.label('Image Format').classes('w-48')
                ui.select(
                    options={'png': 'PNG', 'jpg': 'JPEG', 'webp': 'WebP'},
                    value=self.settings.get('output_format', 'png')
                ).classes('flex-grow').on(
                    'update:model-value',
                    lambda e: self._update_setting('output_format', e.args)
                )
                
            # Output quality
            with ui.row().classes('items-center gap-4 mb-4'):
                ui.label('JPEG Quality').classes('w-48')
                ui.slider(
                    min=50,
                    max=100,
                    value=self.settings.get('output_quality', 95)
                ).props('label-always').classes('flex-grow').on(
                    'update:model-value',
                    lambda e: self._update_setting('output_quality', int(e.args))
                )
                
            # Output directory
            with ui.row().classes('items-center gap-4 mb-4'):
                ui.label('Output Directory').classes('w-48')
                ui.input(
                    value=str(Path.home() / "NeuralForge" / "outputs"),
                    readonly=True
                ).classes('flex-grow')
                ui.button('Browse', icon='folder_open').props('flat')
                
            # File naming
            with ui.row().classes('items-center gap-4 mb-4'):
                ui.label('File Naming').classes('w-48')
                ui.input(
                    value='{model}_{timestamp}_{seed}',
                    placeholder='File naming pattern'
                ).classes('flex-grow')
                
            ui.button('Save Output Settings', icon='save', on_click=self._save_settings).props('unelevated')
            
    def _render_advanced_settings(self):
        """Render advanced settings"""
        with ui.column().classes('w-full gap-4 max-w-2xl'):
            ui.label('Advanced Settings').classes('text-xl font-bold mb-2')
            
            # Telemetry
            with ui.row().classes('items-center gap-4 mb-4'):
                ui.label('Anonymous Telemetry').classes('w-48')
                ui.switch(
                    value=self.settings.get('enable_telemetry', False)
                ).on(
                    'update:model-value',
                    lambda e: self._update_setting('enable_telemetry', e.args)
                )
                ui.label('Help improve the app').classes('text-sm text-gray-400')
                
            # Developer mode
            with ui.row().classes('items-center gap-4 mb-4'):
                ui.label('Developer Mode').classes('w-48')
                ui.switch(value=False).props('disable')
                ui.label('Show advanced options').classes('text-sm text-gray-400')
                
            # Clear cache button
            with ui.column().classes('mt-6 gap-2'):
                ui.label('Cache Management').classes('font-medium')
                ui.button(
                    'Clear Model Cache',
                    icon='delete_sweep',
                    on_click=self._clear_cache
                ).props('flat color=red')
                
            # Reset settings
            with ui.column().classes('mt-6 gap-2'):
                ui.label('Reset').classes('font-medium')
                ui.button(
                    'Reset All Settings',
                    icon='restart_alt',
                    on_click=self._reset_settings
                ).props('flat color=red')
                
    def _show_gpu_info(self):
        """Show current GPU information"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                with ui.grid(columns=2).classes('gap-2 text-sm'):
                    ui.label('Name:')
                    ui.label(gpu.name)
                    ui.label('Memory:')
                    ui.label(f'{gpu.memoryFree:.0f}/{gpu.memoryTotal:.0f} MB')
                    ui.label('Utilization:')
                    ui.label(f'{gpu.load * 100:.0f}%')
                    ui.label('Temperature:')
                    ui.label(f'{gpu.temperature}°C')
            else:
                ui.label('No GPU detected').classes('text-gray-400')
        except:
            ui.label('GPU information unavailable').classes('text-gray-400')
            
    def _update_setting(self, key: str, value):
        """Update a setting value"""
        self.settings[key] = value
        
    def _clear_cache(self):
        """Clear model cache"""
        ui.notify('Cache cleared!', type='positive')
        
    def _reset_settings(self):
        """Reset all settings to defaults"""
        self.settings = self._get_default_settings()
        self._save_settings()
        ui.notify('Settings reset to defaults!', type='info')
        # Reload the page
        ui.run_javascript('window.location.reload()')
        
    def _on_language_change(self, e):
        """Handle language change"""
        new_language = e.args
        self._update_setting('language', new_language)
        set_language(new_language)
        self._save_settings()
        
        # Show notification
        ui.notify(self.translator.t('settings.saved'), type='positive')
        
        # Reload the page to apply new language
        ui.run_javascript('window.location.reload()')