"""Intelligent LoRA auto-suggestion system with NLP analysis"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
import json
import re
import time
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class LoRASuggestion:
    """A suggested LoRA for a prompt"""
    lora_id: str
    name: str
    model_type: str
    base_model: str
    relevance_score: float
    matched_keywords: List[str]
    trigger_words: List[str]
    download_url: Optional[str] = None
    file_size: Optional[int] = None
    preview_images: List[str] = field(default_factory=list)
    description: str = ""
    download_count: int = 0
    rating: float = 0.0
    nsfw: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "lora_id": self.lora_id,
            "name": self.name,
            "model_type": self.model_type,
            "base_model": self.base_model,
            "relevance_score": self.relevance_score,
            "matched_keywords": self.matched_keywords,
            "trigger_words": self.trigger_words,
            "download_url": self.download_url,
            "file_size": self.file_size,
            "preview_images": self.preview_images,
            "description": self.description,
            "download_count": self.download_count,
            "rating": self.rating,
            "nsfw": self.nsfw
        }


@dataclass
class PromptAnalysis:
    """Analysis results for a prompt"""
    concepts: List[str]
    styles: List[str]
    subjects: List[str]
    attributes: Dict[str, List[str]]
    sentiment: str
    complexity: float
    
    
class LoRASuggestionEngine:
    """Intelligent LoRA suggestion engine using NLP"""
    
    # Common concept mappings
    CONCEPT_KEYWORDS = {
        "portrait": ["face", "headshot", "person", "character", "portrait"],
        "landscape": ["scenery", "nature", "outdoor", "environment", "vista"],
        "anime": ["anime", "manga", "kawaii", "chibi", "waifu"],
        "realistic": ["photorealistic", "realistic", "photograph", "real", "lifelike"],
        "fantasy": ["fantasy", "magical", "mythical", "dragon", "wizard", "elf"],
        "scifi": ["sci-fi", "futuristic", "cyberpunk", "space", "robot", "tech"],
        "horror": ["horror", "scary", "dark", "creepy", "gothic", "nightmare"],
        "cartoon": ["cartoon", "animated", "pixar", "disney", "3d render"],
        "abstract": ["abstract", "surreal", "artistic", "conceptual", "modern art"],
        "vintage": ["vintage", "retro", "classic", "old", "nostalgic", "antique"]
    }
    
    # Style keywords
    STYLE_KEYWORDS = {
        "artistic": ["painting", "artwork", "artistic", "brushstrokes", "canvas"],
        "digital": ["digital art", "cgi", "render", "3d", "computer generated"],
        "traditional": ["oil painting", "watercolor", "pencil", "charcoal", "sketch"],
        "photography": ["photograph", "photo", "shot", "lens", "camera", "exposure"],
        "illustration": ["illustration", "drawing", "artwork", "design", "graphic"]
    }
    
    def __init__(
        self,
        civitai_manager: Optional[Any] = None,
        cache_dir: Optional[Path] = None
    ):
        self.civitai_manager = civitai_manager
        self.cache_dir = cache_dir or Path("./cache/lora_suggestions")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # NLP components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # User preference tracking
        self.user_preferences = self._load_preferences()
        
        # LoRA metadata cache
        self.lora_metadata_cache = self._load_metadata_cache()
        
    def _load_preferences(self) -> Dict[str, Any]:
        """Load user preferences"""
        pref_file = self.cache_dir / "user_preferences.json"
        if pref_file.exists():
            with open(pref_file, "r") as f:
                return json.load(f)
        return {
            "accepted_loras": [],
            "rejected_loras": [],
            "favorite_styles": [],
            "blacklisted_tags": []
        }
        
    def _save_preferences(self):
        """Save user preferences"""
        pref_file = self.cache_dir / "user_preferences.json"
        with open(pref_file, "w") as f:
            json.dump(self.user_preferences, f, indent=2)
            
    def _load_metadata_cache(self) -> Dict[str, Any]:
        """Load LoRA metadata cache"""
        cache_file = self.cache_dir / "lora_metadata.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)
        return {}
        
    def _save_metadata_cache(self):
        """Save LoRA metadata cache"""
        cache_file = self.cache_dir / "lora_metadata.json"
        with open(cache_file, "w") as f:
            json.dump(self.lora_metadata_cache, f, indent=2)
            
    def analyze_prompt(
        self,
        prompt: str,
        negative_prompt: str = ""
    ) -> PromptAnalysis:
        """Analyze a prompt to extract concepts and attributes
        
        Args:
            prompt: The generation prompt
            negative_prompt: The negative prompt
            
        Returns:
            PromptAnalysis object
        """
        # Clean and tokenize
        prompt_lower = prompt.lower()
        tokens = re.findall(r'\b\w+\b', prompt_lower)
        
        # Extract concepts
        concepts = []
        for concept, keywords in self.CONCEPT_KEYWORDS.items():
            if any(kw in prompt_lower for kw in keywords):
                concepts.append(concept)
                
        # Extract styles
        styles = []
        for style, keywords in self.STYLE_KEYWORDS.items():
            if any(kw in prompt_lower for kw in keywords):
                styles.append(style)
                
        # Extract subjects (nouns)
        subjects = self._extract_subjects(prompt)
        
        # Extract attributes
        attributes = {
            "colors": self._extract_colors(prompt),
            "lighting": self._extract_lighting(prompt),
            "mood": self._extract_mood(prompt),
            "quality": self._extract_quality_terms(prompt)
        }
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(prompt)
        
        # Calculate complexity
        complexity = len(tokens) / 10.0  # Simple metric
        
        return PromptAnalysis(
            concepts=concepts,
            styles=styles,
            subjects=subjects,
            attributes=attributes,
            sentiment=sentiment,
            complexity=min(1.0, complexity)
        )
        
    def _extract_subjects(self, prompt: str) -> List[str]:
        """Extract subject nouns from prompt"""
        # Simple pattern matching for common subjects
        subjects = []
        
        subject_patterns = [
            r'\b(man|woman|person|people|girl|boy|child)\b',
            r'\b(cat|dog|animal|bird|dragon|creature)\b',
            r'\b(car|vehicle|building|city|house|castle)\b',
            r'\b(tree|flower|mountain|ocean|forest|landscape)\b'
        ]
        
        for pattern in subject_patterns:
            matches = re.findall(pattern, prompt.lower())
            subjects.extend(matches)
            
        return list(set(subjects))
        
    def _extract_colors(self, prompt: str) -> List[str]:
        """Extract color terms"""
        color_pattern = r'\b(red|blue|green|yellow|orange|purple|pink|black|white|gray|brown|gold|silver)\b'
        return list(set(re.findall(color_pattern, prompt.lower())))
        
    def _extract_lighting(self, prompt: str) -> List[str]:
        """Extract lighting terms"""
        lighting_terms = [
            "sunlight", "moonlight", "candlelight", "neon", "dramatic lighting",
            "soft lighting", "harsh lighting", "backlit", "rim lighting", "ambient"
        ]
        found = []
        prompt_lower = prompt.lower()
        for term in lighting_terms:
            if term in prompt_lower:
                found.append(term)
        return found
        
    def _extract_mood(self, prompt: str) -> List[str]:
        """Extract mood/atmosphere terms"""
        mood_terms = [
            "happy", "sad", "angry", "peaceful", "dramatic", "mysterious",
            "romantic", "epic", "serene", "chaotic", "melancholic", "joyful"
        ]
        found = []
        prompt_lower = prompt.lower()
        for term in mood_terms:
            if term in prompt_lower:
                found.append(term)
        return found
        
    def _extract_quality_terms(self, prompt: str) -> List[str]:
        """Extract quality-related terms"""
        quality_pattern = r'\b(masterpiece|best quality|high quality|detailed|intricate|professional|award.?winning)\b'
        return list(set(re.findall(quality_pattern, prompt.lower())))
        
    def _analyze_sentiment(self, prompt: str) -> str:
        """Simple sentiment analysis"""
        positive_words = ["beautiful", "amazing", "wonderful", "perfect", "excellent"]
        negative_words = ["dark", "evil", "scary", "horrible", "terrible"]
        
        prompt_lower = prompt.lower()
        pos_count = sum(1 for word in positive_words if word in prompt_lower)
        neg_count = sum(1 for word in negative_words if word in prompt_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
            
    async def suggest_loras(
        self,
        prompt: str,
        negative_prompt: str = "",
        base_model: str = "FLUX.1",
        max_suggestions: int = 5,
        include_nsfw: bool = False
    ) -> List[LoRASuggestion]:
        """Suggest LoRAs based on prompt analysis
        
        Args:
            prompt: Generation prompt
            negative_prompt: Negative prompt
            base_model: Base model compatibility
            max_suggestions: Maximum suggestions to return
            include_nsfw: Include NSFW suggestions
            
        Returns:
            List of LoRA suggestions
        """
        # Analyze prompt
        analysis = self.analyze_prompt(prompt, negative_prompt)
        
        # Build search queries
        search_queries = self._build_search_queries(analysis)
        
        # Search for LoRAs
        all_suggestions = []
        
        if self.civitai_manager:
            for query in search_queries[:3]:  # Limit searches
                try:
                    results = await self._search_civitai(
                        query,
                        base_model,
                        include_nsfw
                    )
                    all_suggestions.extend(results)
                except Exception as e:
                    logger.error(f"Search failed for '{query}': {e}")
                    
        # Score and rank suggestions
        scored_suggestions = self._score_suggestions(
            all_suggestions,
            analysis,
            prompt
        )
        
        # Apply user preferences
        filtered_suggestions = self._apply_user_preferences(scored_suggestions)
        
        # Sort by score and return top N
        filtered_suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return filtered_suggestions[:max_suggestions]
        
    def _build_search_queries(self, analysis: PromptAnalysis) -> List[str]:
        """Build search queries from prompt analysis"""
        queries = []
        
        # Concept-based queries
        for concept in analysis.concepts[:2]:
            queries.append(concept)
            
        # Style-based queries
        for style in analysis.styles[:1]:
            queries.append(style)
            
        # Subject-based queries
        for subject in analysis.subjects[:2]:
            queries.append(subject)
            
        # Combined queries
        if analysis.concepts and analysis.styles:
            queries.append(f"{analysis.concepts[0]} {analysis.styles[0]}")
            
        return queries
        
    async def _search_civitai(
        self,
        query: str,
        base_model: str,
        include_nsfw: bool
    ) -> List[LoRASuggestion]:
        """Search Civitai for LoRAs"""
        suggestions = []
        
        try:
            # Search using Civitai manager
            models = self.civitai_manager.search_models(
                query=query,
                types=["LORA"],
                base_models=[base_model],
                nsfw=include_nsfw,
                limit=10
            )
            
            for model in models:
                # Get or cache metadata
                metadata = await self._get_lora_metadata(model.id)
                
                suggestion = LoRASuggestion(
                    lora_id=str(model.id),
                    name=model.name,
                    model_type="LORA",
                    base_model=base_model,
                    relevance_score=0.0,  # Will be calculated
                    matched_keywords=[query],
                    trigger_words=metadata.get("trigger_words", []),
                    download_url=model.download_url,
                    file_size=model.file_size,
                    preview_images=model.images[:3],
                    description=model.description,
                    download_count=model.stats.get("downloads", 0),
                    rating=model.stats.get("rating", 0.0),
                    nsfw=model.nsfw
                )
                
                suggestions.append(suggestion)
                
        except Exception as e:
            logger.error(f"Civitai search failed: {e}")
            
        return suggestions
        
    async def _get_lora_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get or fetch LoRA metadata"""
        if model_id in self.lora_metadata_cache:
            return self.lora_metadata_cache[model_id]
            
        # Fetch from Civitai
        try:
            metadata = await self.civitai_manager.get_model_metadata(model_id)
            
            # Extract trigger words from training data
            trigger_words = []
            if "trainedWords" in metadata:
                trigger_words = metadata["trainedWords"]
            elif "trigger" in metadata:
                trigger_words = [metadata["trigger"]]
                
            self.lora_metadata_cache[model_id] = {
                "trigger_words": trigger_words,
                "training_tags": metadata.get("tags", []),
                "base_model": metadata.get("baseModel", ""),
                "version": metadata.get("version", "")
            }
            
            self._save_metadata_cache()
            
        except Exception as e:
            logger.error(f"Failed to fetch metadata for {model_id}: {e}")
            self.lora_metadata_cache[model_id] = {
                "trigger_words": [],
                "training_tags": [],
                "base_model": "",
                "version": ""
            }
            
        return self.lora_metadata_cache[model_id]
        
    def _score_suggestions(
        self,
        suggestions: List[LoRASuggestion],
        analysis: PromptAnalysis,
        prompt: str
    ) -> List[LoRASuggestion]:
        """Score suggestions based on relevance"""
        
        # Build prompt vector
        prompt_tokens = " ".join([
            prompt,
            " ".join(analysis.concepts),
            " ".join(analysis.styles),
            " ".join(analysis.subjects)
        ])
        
        for suggestion in suggestions:
            score = 0.0
            
            # Concept matching
            suggestion_text = " ".join([
                suggestion.name,
                suggestion.description,
                " ".join(suggestion.trigger_words)
            ]).lower()
            
            # Count keyword matches
            matches = []
            for concept in analysis.concepts:
                if concept in suggestion_text:
                    score += 0.3
                    matches.append(concept)
                    
            for style in analysis.styles:
                if style in suggestion_text:
                    score += 0.2
                    matches.append(style)
                    
            for subject in analysis.subjects:
                if subject in suggestion_text:
                    score += 0.1
                    matches.append(subject)
                    
            # Popularity boost
            if suggestion.download_count > 10000:
                score += 0.1
            elif suggestion.download_count > 1000:
                score += 0.05
                
            # Rating boost
            if suggestion.rating > 4.5:
                score += 0.1
            elif suggestion.rating > 4.0:
                score += 0.05
                
            # User preference boost
            if suggestion.lora_id in self.user_preferences.get("accepted_loras", []):
                score += 0.3
            elif suggestion.lora_id in self.user_preferences.get("rejected_loras", []):
                score -= 0.5
                
            suggestion.relevance_score = min(1.0, score)
            suggestion.matched_keywords = list(set(matches))
            
        return suggestions
        
    def _apply_user_preferences(
        self,
        suggestions: List[LoRASuggestion]
    ) -> List[LoRASuggestion]:
        """Filter suggestions based on user preferences"""
        filtered = []
        
        blacklisted_tags = self.user_preferences.get("blacklisted_tags", [])
        
        for suggestion in suggestions:
            # Skip if blacklisted
            if suggestion.lora_id in self.user_preferences.get("rejected_loras", []):
                continue
                
            # Skip if contains blacklisted tags
            suggestion_tags = suggestion.description.lower().split()
            if any(tag in suggestion_tags for tag in blacklisted_tags):
                continue
                
            filtered.append(suggestion)
            
        return filtered
        
    def record_user_action(
        self,
        lora_id: str,
        action: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record user action for learning
        
        Args:
            lora_id: LoRA ID
            action: Action taken (accepted, rejected, downloaded)
            metadata: Additional metadata
        """
        if action == "accepted":
            if lora_id not in self.user_preferences["accepted_loras"]:
                self.user_preferences["accepted_loras"].append(lora_id)
                # Remove from rejected if present
                if lora_id in self.user_preferences["rejected_loras"]:
                    self.user_preferences["rejected_loras"].remove(lora_id)
                    
        elif action == "rejected":
            if lora_id not in self.user_preferences["rejected_loras"]:
                self.user_preferences["rejected_loras"].append(lora_id)
                # Remove from accepted if present
                if lora_id in self.user_preferences["accepted_loras"]:
                    self.user_preferences["accepted_loras"].remove(lora_id)
                    
        self._save_preferences()
        
    def get_trigger_words(self, lora_id: str) -> List[str]:
        """Get trigger words for a LoRA"""
        metadata = self.lora_metadata_cache.get(lora_id, {})
        return metadata.get("trigger_words", [])
        
    def update_prompt_with_triggers(
        self,
        prompt: str,
        lora_suggestions: List[LoRASuggestion]
    ) -> str:
        """Update prompt with trigger words from accepted LoRAs
        
        Args:
            prompt: Original prompt
            lora_suggestions: Accepted LoRA suggestions
            
        Returns:
            Updated prompt with trigger words
        """
        trigger_words = []
        
        for suggestion in lora_suggestions:
            triggers = suggestion.trigger_words
            if triggers:
                # Add unique trigger words
                for trigger in triggers:
                    if trigger.lower() not in prompt.lower():
                        trigger_words.append(trigger)
                        
        if trigger_words:
            # Insert trigger words at the beginning
            updated_prompt = f"{', '.join(trigger_words)}, {prompt}"
        else:
            updated_prompt = prompt
            
        return updated_prompt
        
    def export_statistics(self) -> Dict[str, Any]:
        """Export usage statistics"""
        return {
            "total_accepted": len(self.user_preferences["accepted_loras"]),
            "total_rejected": len(self.user_preferences["rejected_loras"]),
            "favorite_styles": Counter(self.user_preferences.get("favorite_styles", [])).most_common(5),
            "metadata_cache_size": len(self.lora_metadata_cache),
            "last_updated": datetime.now().isoformat()
        }