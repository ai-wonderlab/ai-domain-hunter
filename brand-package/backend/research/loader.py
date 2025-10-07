"""
Research Pattern Loader
"""
import os
from pathlib import Path
from typing import Dict, Optional, List, Any
import json
import hashlib
from datetime import datetime, timedelta
import logging

from config.settings import settings
from core.exceptions import FileNotFoundError

logger = logging.getLogger(__name__)


class ResearchPatternLoader:
    """Load and manage research patterns for generation"""
    
    _instance = None
    _cache: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self):
        """Initialize research loader"""
        self.research_dir = Path(__file__).parent / "documents"
        self.cache_dir = Path(__file__).parent / "cache"
        
        # Ensure directories exist
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration
        self.cache_ttl = timedelta(hours=24)  # Cache for 24 hours
        
        logger.info(f"✅ Research loader initialized")
        logger.info(f"   Documents: {self.research_dir}")
        logger.info(f"   Cache: {self.cache_dir}")
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_pattern(
        self,
        pattern_type: str,
        force_reload: bool = False
    ) -> str:
        """Load research pattern
        
        Args:
            pattern_type: Type of pattern (name, logo, color, tagline)
            force_reload: Force reload from disk
            
        Returns:
            Pattern content
        """
        # Check memory cache
        if not force_reload and pattern_type in self._cache:
            cache_data = self._cache[pattern_type]
            if self._is_cache_valid(cache_data):
                logger.debug(f"Using cached {pattern_type} pattern")
                return cache_data["content"]
        
        # Check file cache
        cache_file = self.cache_dir / f"{pattern_type}_cache.json"
        if not force_reload and cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    if self._is_cache_valid(cache_data):
                        logger.info(f"Loaded {pattern_type} from file cache")
                        self._cache[pattern_type] = cache_data
                        return cache_data["content"]
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        # Load from source
        content = self._load_from_file(pattern_type)
        
        # Process and enhance content
        enhanced_content = self._enhance_pattern(content, pattern_type)
        
        # Save to cache
        self._save_to_cache(pattern_type, enhanced_content)
        
        return enhanced_content
    
    def _load_from_file(self, pattern_type: str) -> str:
        """Load pattern from file
        
        Args:
            pattern_type: Pattern type
            
        Returns:
            File content
        """
        file_path = self.research_dir / f"{pattern_type}_intelligence.txt"
        
        if not file_path.exists():
            logger.warning(f"Pattern file not found: {file_path}")
            return self._get_default_pattern(pattern_type)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded {pattern_type} pattern from file")
            return content
        except Exception as e:
            logger.error(f"Failed to load pattern: {e}")
            return self._get_default_pattern(pattern_type)
    
    def _enhance_pattern(self, content: str, pattern_type: str) -> str:
        """Enhance pattern with additional intelligence
        
        Args:
            content: Original content
            pattern_type: Pattern type
            
        Returns:
            Enhanced content
        """
        # Add metadata header
        header = f"""# {pattern_type.upper()} GENERATION INTELLIGENCE
# Version: {__version__}
# Updated: {datetime.now().isoformat()}
# Environment: {settings.app_env}
# -----------------------------------

"""
        
        # Add type-specific enhancements
        enhancements = self._get_enhancements(pattern_type)
        
        # Combine
        enhanced = header + content
        if enhancements:
            enhanced += f"\n\n# ADDITIONAL PATTERNS\n\n{enhancements}"
        
        return enhanced
    
    def _get_enhancements(self, pattern_type: str) -> str:
        """Get type-specific enhancements
        
        Args:
            pattern_type: Pattern type
            
        Returns:
            Enhancement text
        """
        enhancements = {
            "name": self._get_name_enhancements(),
            "logo": self._get_logo_enhancements(),
            "color": self._get_color_enhancements(),
            "tagline": self._get_tagline_enhancements()
        }
        
        return enhancements.get(pattern_type, "")
    
    def _get_name_enhancements(self) -> str:
        """Get name generation enhancements"""
        return """## Modern Naming Trends (2024-2025)

### Tech Industry Patterns:
- Single syllable power words: Flux, Nexus, Core, Spark
- Compound innovations: DataForge, CloudNative, EdgeWise
- AI-focused: Neural*, Deep*, Smart*, Auto*
- Developer-friendly: Build*, Code*, Dev*, Stack*

### Domain Availability Strategies:
- Use creative TLDs: .ai, .io, .dev, .app, .cloud
- Add prefixes: get*, try*, use*, hello*
- Add suffixes: *hq, *labs, *studio, *works
- Invented words: Unique but pronounceable

### Cultural Considerations:
- Avoid unintended meanings in major languages
- Check pronunciation across cultures
- Consider trademark availability globally
- Test memorability with target audience"""
    
    def _get_logo_enhancements(self) -> str:
        """Get logo generation enhancements"""
        return """## Visual Design Principles

### Modern Logo Trends:
- Geometric minimalism with depth
- Dynamic gradients (used sparingly)
- Adaptive/responsive design systems
- Motion-ready designs for digital
- Accessibility-first approach

### Technical Requirements:
- Vector-first design approach
- Minimum size: 16x16px readable
- Maximum complexity: 3 distinct elements
- Color variations: Full, mono, reversed
- File formats: SVG primary, PNG/JPG secondary

### Industry-Specific Styles:
- Tech: Abstract geometric, circuit patterns
- Finance: Stable shapes, upward trends
- Healthcare: Organic curves, caring symbols
- Education: Growth symbols, book metaphors
- Gaming: Bold, dynamic, energetic"""
    
    def _get_color_enhancements(self) -> str:
        """Get color generation enhancements"""
        return """## Color Psychology & Strategy

### Emotional Associations:
- Trust & Security: Blue (#0066CC - #004499)
- Growth & Nature: Green (#00AA44 - #008833)
- Energy & Urgency: Red (#FF3333 - #CC0000)
- Optimism & Clarity: Yellow (#FFBB33 - #FFAA00)
- Luxury & Creativity: Purple (#8855FF - #6633CC)
- Balance & Premium: Gray (#666666 - #999999)

### Accessibility Standards:
- WCAG AA Minimum: 4.5:1 for normal text
- WCAG AA Large Text: 3:1 for 18pt+ or 14pt+ bold
- WCAG AAA Enhanced: 7:1 for critical UI
- Color blind safe combinations
- High contrast mode support

### Color System Building:
- Primary: Brand recognition (60% usage)
- Secondary: Support and depth (30% usage)
- Accent: CTAs and highlights (10% usage)
- Semantic: Success, warning, error, info
- Neutral: Text, backgrounds, borders"""
    
    def _get_tagline_enhancements(self) -> str:
        """Get tagline generation enhancements"""
        return """## Tagline Effectiveness Formula

### Proven Structures:
1. Benefit Statement: "Get [benefit] without [pain]"
2. Transformation: "From [current] to [desired]"
3. Unique Value: "The only [category] that [differentiator]"
4. Action Call: "[Verb] your [object]"
5. Question Hook: "What if [possibility]?"
6. Social Proof: "Trusted by [audience]"

### Power Words by Category:
- Innovation: Revolutionary, Breakthrough, Pioneer, Transform
- Trust: Proven, Trusted, Reliable, Secure, Guaranteed
- Speed: Instant, Quick, Fast, Rapid, Immediate
- Simplicity: Simple, Easy, Effortless, Intuitive
- Growth: Amplify, Boost, Accelerate, Scale, Expand

### Length Guidelines:
- Micro (2-3 words): Maximum punch, memorable
- Short (4-6 words): Clear value, versatile
- Medium (7-10 words): Complete thought, descriptive
- Long (11-15 words): Full story, SEO-friendly"""
    
    def _save_to_cache(self, pattern_type: str, content: str):
        """Save pattern to cache
        
        Args:
            pattern_type: Pattern type
            content: Pattern content
        """
        cache_data = {
            "pattern_type": pattern_type,
            "content": content,
            "hash": hashlib.md5(content.encode()).hexdigest(),
            "cached_at": datetime.now().isoformat(),
            "ttl_seconds": int(self.cache_ttl.total_seconds())
        }
        
        # Save to memory
        self._cache[pattern_type] = cache_data
        
        # Save to file
        cache_file = self.cache_dir / f"{pattern_type}_cache.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"Cached {pattern_type} pattern")
        except Exception as e:
            logger.error(f"Cache save failed: {e}")
    
    def _is_cache_valid(self, cache_data: Dict[str, Any]) -> bool:
        """Check if cache is still valid
        
        Args:
            cache_data: Cached data
            
        Returns:
            True if valid
        """
        cached_at_str = cache_data.get("cached_at")
        if not cached_at_str:
            return False
        
        cached_at = datetime.fromisoformat(cached_at_str)
        age = datetime.now() - cached_at
        
        return age < self.cache_ttl
    
    def _get_default_pattern(self, pattern_type: str) -> str:
        """Get default pattern if file not found
        
        Args:
            pattern_type: Pattern type
            
        Returns:
            Default pattern content
        """
        defaults = {
            "name": """Generate creative, memorable business names that are:
- Short and easy to pronounce (5-10 characters ideal)
- Unique and brandable
- Available as domains
- Appropriate for the industry
- Globally accessible""",
            
            "logo": """Design professional logos that are:
- Simple and scalable
- Memorable and unique
- Versatile across mediums
- Timeless, not trendy
- Accessible and inclusive""",
            
            "color": """Create color palettes that:
- Reflect brand personality
- Meet accessibility standards
- Work across all mediums
- Create emotional connection
- Support brand recognition""",
            
            "tagline": """Write compelling taglines that are:
- Clear and concise
- Memorable and unique
- Benefit-focused
- Emotionally resonant
- Action-oriented"""
        }
        
        return defaults.get(pattern_type, "Generate appropriate content for user needs.")
    
    def reload_all(self):
        """Reload all patterns"""
        pattern_types = ["name", "logo", "color", "tagline"]
        
        for pattern_type in pattern_types:
            self.load_pattern(pattern_type, force_reload=True)
        
        logger.info(f"✅ Reloaded {len(pattern_types)} patterns")
    
    def clear_cache(self):
        """Clear all cached patterns"""
        self._cache.clear()
        
        # Delete cache files
        for cache_file in self.cache_dir.glob("*_cache.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Failed to delete cache file: {e}")
        
        logger.info("✅ Cache cleared")
    
    def get_all_patterns(self) -> Dict[str, str]:
        """Get all patterns
        
        Returns:
            Dictionary of pattern type to content
        """
        patterns = {}
        for pattern_type in ["name", "logo", "color", "tagline"]:
            patterns[pattern_type] = self.load_pattern(pattern_type)
        return patterns