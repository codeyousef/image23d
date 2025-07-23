"""
Translation system for NeuralForge Studio
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Translator:
    """Handles translation of UI text"""
    
    def __init__(self):
        self.current_language = 'en'
        self.translations: Dict[str, Dict[str, str]] = {}
        self.fallback_language = 'en'
        
        # Load translations
        self._load_translations()
        
    def _load_translations(self):
        """Load all available translations"""
        localization_dir = Path(__file__).parent / 'translations'
        
        # Create translations directory if it doesn't exist
        if not localization_dir.exists():
            logger.warning(f"Translations directory not found at {localization_dir}")
            localization_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # Load all JSON files
        json_files = list(localization_dir.glob('*.json'))
        if not json_files:
            logger.warning(f"No translation files found in {localization_dir}")
            return
            
        for lang_file in json_files:
            lang_code = lang_file.stem
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
                logger.info(f"Loaded translation: {lang_code}")
            except Exception as e:
                logger.error(f"Failed to load translation {lang_code}: {e}")
                
        # Log summary
        logger.info(f"Loaded {len(self.translations)} translations: {list(self.translations.keys())}")
                
    def set_language(self, language: str):
        """Set the current language"""
        if language in self.translations:
            self.current_language = language
            logger.info(f"Language set to: {language}")
        else:
            logger.warning(f"Language {language} not available, keeping {self.current_language}")
            
    def get_available_languages(self) -> Dict[str, str]:
        """Get list of available languages"""
        languages = {}
        for lang_code in self.translations:
            lang_info = self.translations[lang_code].get('_metadata', {})
            languages[lang_code] = lang_info.get('name', lang_code)
        return languages
        
    def t(self, key: str, **kwargs) -> str:
        """Translate a key with optional formatting parameters"""
        # Try current language first
        if self.current_language in self.translations:
            text = self._get_nested_value(self.translations[self.current_language], key)
            if text is not None:
                # Apply formatting if parameters provided
                if kwargs:
                    try:
                        return text.format(**kwargs)
                    except Exception as e:
                        logger.warning(f"Formatting error for key {key}: {e}")
                        return text
                return text
                
        # Fallback to English
        if self.fallback_language in self.translations:
            text = self._get_nested_value(self.translations[self.fallback_language], key)
            if text is not None:
                if kwargs:
                    try:
                        return text.format(**kwargs)
                    except:
                        pass
                return text
                
        # Return key if no translation found
        logger.debug(f"No translation found for key: {key}")
        return key
        
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Optional[str]:
        """Get value from nested dictionary using dot notation"""
        keys = key.split('.')
        value = data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
                
        return str(value) if value is not None else None
        
    def is_rtl(self) -> bool:
        """Check if current language is right-to-left"""
        rtl_languages = ['ar', 'he', 'fa', 'ur']
        return self.current_language in rtl_languages
        
    def get_font_family(self) -> str:
        """Get appropriate font family for current language"""
        if self.current_language == 'ar':
            return "'Noto Sans Arabic', 'Cairo', 'Arial', sans-serif"
        elif self.current_language in ['zh', 'zh-CN']:
            return "'Noto Sans SC', 'Microsoft YaHei', sans-serif"
        elif self.current_language in ['ja']:
            return "'Noto Sans JP', 'Hiragino Sans', sans-serif"
        elif self.current_language in ['ko']:
            return "'Noto Sans KR', 'Malgun Gothic', sans-serif"
        else:
            return "'Inter', 'Roboto', 'Arial', sans-serif"

# Global translator instance
_translator = None

def get_translator() -> Translator:
    """Get the global translator instance"""
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator

def set_language(language: str):
    """Set the global language"""
    translator = get_translator()
    translator.set_language(language)
    
def t(key: str, **kwargs) -> str:
    """Translate a key (convenience function)"""
    translator = get_translator()
    return translator.t(key, **kwargs)