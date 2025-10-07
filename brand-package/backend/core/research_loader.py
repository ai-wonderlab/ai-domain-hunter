"""
Research Pattern Loader and Cache Manager
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging

from config.settings import settings
from config.constants import RESEARCH_CACHE_TTL

logger = logging.getLogger(__name__)


class ResearchLoader:
    """
    Loads and caches research patterns for intelligent generation
    """
    
    _instance = None
    _patterns: Dict[str, str] = {}
    _cache_metadata: Dict[str, Dict] = {}
    
    def __init__(self):
        """Initialize research loader"""
        self.research_dir = settings.research_dir
        self.cache_dir = settings.cache_dir
        
        # Ensure directories exist
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Research loader initialized")
        logger.info(f"   Research dir: {self.research_dir}")
        logger.info(f"   Cache dir: {self.cache_dir}")
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def load_all_patterns(cls):
        """Load all research patterns on startup"""
        instance = cls.get_instance()
        
        pattern_files = {
            "name": "name_intelligence.txt",
            "logo": "logo_intelligence.txt",
            "color": "color_intelligence.txt",
            "tagline": "tagline_intelligence.txt"
        }
        
        loaded = 0
        for pattern_type, filename in pattern_files.items():
            try:
                instance.load_pattern(pattern_type)
                loaded += 1
            except Exception as e:
                logger.warning(f"Could not load {pattern_type} patterns: {e}")
        
        logger.info(f"✅ Loaded {loaded}/{len(pattern_files)} research patterns")
        return instance
    
    def load_pattern(self, pattern_type: str, force_reload: bool = False) -> str:
        """
        Load research pattern for a specific generation type
        
        Args:
            pattern_type: Type of pattern (name, logo, color, tagline)
            force_reload: Force reload from disk even if cached
        
        Returns:
            Pattern content
        """
        # Check in-memory cache first
        if not force_reload and pattern_type in self._patterns:
            if self._is_cache_valid(pattern_type):
                logger.debug(f"Using cached {pattern_type} patterns")
                return self._patterns[pattern_type]
        
        # Check file cache
        cache_file = self.cache_dir / f"{pattern_type}_cache.json"
        if not force_reload and cache_file.exists():
            try:
                cached_data = self._load_from_cache(cache_file)
                if cached_data and self._is_file_cache_valid(cached_data):
                    logger.info(f"Loaded {pattern_type} patterns from cache")
                    self._patterns[pattern_type] = cached_data["content"]
                    self._cache_metadata[pattern_type] = cached_data["metadata"]
                    return cached_data["content"]
            except Exception as e:
                logger.warning(f"Cache read failed for {pattern_type}: {e}")
        
        # Load from source file
        source_file = self.research_dir / f"{pattern_type}_intelligence.txt"
        if not source_file.exists():
            logger.warning(f"Research file not found: {source_file}")
            return self._get_default_pattern(pattern_type)
        
        try:
            content = source_file.read_text(encoding="utf-8")
            
            # Process and enhance the content
            processed_content = self._process_pattern(content, pattern_type)
            
            # Save to cache
            self._save_to_cache(pattern_type, processed_content, cache_file)
            
            # Store in memory
            self._patterns[pattern_type] = processed_content
            self._cache_metadata[pattern_type] = {
                "loaded_at": datetime.now(),
                "source_file": str(source_file),
                "size": len(processed_content)
            }
            
            logger.info(f"✅ Loaded {pattern_type} patterns from source ({len(processed_content)} chars)")
            return processed_content
            
        except Exception as e:
            logger.error(f"Failed to load {pattern_type} patterns: {e}")
            return self._get_default_pattern(pattern_type)
    
    def _process_pattern(self, content: str, pattern_type: str) -> str:
        """
        Process and enhance pattern content
        Can add AI analysis here if needed
        """
        # Add metadata header
        header = f"""# {pattern_type.upper()} GENERATION PATTERNS
# Loaded: {datetime.now().isoformat()}
# Type: {pattern_type}
# ---

"""
        
        # Clean and format content
        content = content.strip()
        
        # Add type-specific enhancements
        if pattern_type == "name":
            content = self._enhance_name_patterns(content)
        elif pattern_type == "logo":
            content = self._enhance_logo_patterns(content)
        elif pattern_type == "color":
            content = self._enhance_color_patterns(content)
        elif pattern_type == "tagline":
            content = self._enhance_tagline_patterns(content)
        
        return header + content
    
    def _enhance_name_patterns(self, content: str) -> str:
        """Enhance name generation patterns"""
        additions = """
## ADDITIONAL PATTERNS FOR NAME GENERATION

### Successful .AI Domain Patterns:
- Single meaningful word: genius.ai, spark.ai
- Compound words: dataflow.ai, mindforge.ai
- Action + Object: buildbot.ai, learnhub.ai
- Adjective + Noun: smartcode.ai, deepwork.ai

### Avoid:
- Numbers (unless meaningful)
- Hyphens (reduce memorability)
- Generic terms (ai, ml, tech alone)
- Trademark conflicts
"""
        return content + "\n\n" + additions
    
    def _enhance_logo_patterns(self, content: str) -> str:
        """Enhance logo generation patterns"""
        additions = """
## LOGO GENERATION GUIDELINES

### Visual Hierarchy:
- Primary element should be instantly recognizable
- Secondary elements support, not compete
- Negative space is powerful

### Scalability Requirements:
- Must work at 16px favicon size
- Must work at billboard size
- Test at 32px, 64px, 128px, 512px

### Modern Logo Trends:
- Geometric simplification
- Gradient meshes (used sparingly)
- Variable width strokes
- Asymmetric balance
"""
        return content + "\n\n" + additions
    
    def _enhance_color_patterns(self, content: str) -> str:
        """Enhance color generation patterns"""
        additions = """
## COLOR PSYCHOLOGY & ACCESSIBILITY

### Industry Standards:
- FinTech: Blue (#0066CC), Green (#00AA44)
- Healthcare: Teal (#17A2B8), Green (#28A745)
- Education: Blue (#4A90E2), Orange (#FF6B35)
- Gaming: Purple (#8B5CF6), Neon (#39FF14)

### Accessibility Requirements:
- WCAG AA: 4.5:1 contrast ratio for normal text
- WCAG AAA: 7:1 contrast ratio for enhanced accessibility
- Test with color blindness simulators

### Emotional Associations:
- Trust: Blue tones
- Growth: Green tones
- Energy: Orange/Red tones
- Luxury: Purple/Gold tones
- Innovation: Purple/Teal tones
"""
        return content + "\n\n" + additions
    
    def _enhance_tagline_patterns(self, content: str) -> str:
        """Enhance tagline generation patterns"""
        additions = """
## TAGLINE FORMULAS

### Classic Structures:
1. Verb + Benefit: "Build better products"
2. Question: "What will you create?"
3. Declaration: "The future is here"
4. Invitation: "Join the revolution"
5. Promise: "Always connected, always secure"

### Length Guidelines:
- Short (2-4 words): Maximum impact, memorability
- Medium (5-8 words): Balance of clarity and detail
- Long (9-12 words): Full value proposition

### Power Words:
- Transform, Accelerate, Empower
- Simple, Smart, Seamless
- Future, Next, Beyond
- Together, Connected, United
"""
        return content + "\n\n" + additions
    
    def _save_to_cache(self, pattern_type: str, content: str, cache_file: Path):
        """Save processed pattern to cache"""
        cache_data = {
            "pattern_type": pattern_type,
            "content": content,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "content_hash": hashlib.md5(content.encode()).hexdigest(),
                "size": len(content),
                "ttl_seconds": RESEARCH_CACHE_TTL
            }
        }
        
        try:
            cache_file.write_text(json.dumps(cache_data, indent=2))
            logger.debug(f"Saved {pattern_type} to cache: {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache for {pattern_type}: {e}")
    
    def _load_from_cache(self, cache_file: Path) -> Optional[Dict]:
        """Load pattern from cache file"""
        try:
            content = cache_file.read_text()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_file}: {e}")
            return None
    
    def _is_cache_valid(self, pattern_type: str) -> bool:
        """Check if in-memory cache is still valid"""
        if pattern_type not in self._cache_metadata:
            return False
        
        metadata = self._cache_metadata[pattern_type]
        loaded_at = metadata.get("loaded_at")
        
        if not loaded_at:
            return False
        
        age = datetime.now() - loaded_at
        return age.total_seconds() < RESEARCH_CACHE_TTL
    
    def _is_file_cache_valid(self, cache_data: Dict) -> bool:
        """Check if file cache is still valid"""
        metadata = cache_data.get("metadata", {})
        created_at_str = metadata.get("created_at")
        
        if not created_at_str:
            return False
        
        created_at = datetime.fromisoformat(created_at_str)
        age = datetime.now() - created_at
        ttl = metadata.get("ttl_seconds", RESEARCH_CACHE_TTL)
        
        return age.total_seconds() < ttl
    
    def _get_default_pattern(self, pattern_type: str) -> str:
        """Get default pattern if file not found"""
        defaults = {
            "name": """Generate creative business names that are:
- Short and memorable (5-10 characters ideal)
- Easy to pronounce and spell
- Unique and brandable
- Domain-friendly (.com/.ai availability likely)
- Industry-appropriate""",
            
            "logo": """Create logo designs that are:
- Simple and scalable
- Memorable and unique
- Appropriate for the industry
- Work in black and white
- Timeless, not trendy""",
            
            "color": """Generate color palettes that:
- Reflect brand personality
- Are accessible (WCAG compliant)
- Work across media
- Include primary, secondary, and accent colors
- Have proper contrast ratios""",
            
            "tagline": """Create taglines that are:
- Clear and concise
- Memorable and unique
- Benefit-focused
- Emotionally resonant
- Action-oriented"""
        }
        
        return defaults.get(pattern_type, "Generate appropriate content for the user's needs.")
    
    def get_pattern(self, pattern_type: str) -> str:
        """Public method to get pattern"""
        return self.load_pattern(pattern_type)
    
    def clear_cache(self):
        """Clear all cached patterns"""
        self._patterns.clear()
        self._cache_metadata.clear()
        
        # Delete cache files
        for cache_file in self.cache_dir.glob("*_cache.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info("✅ Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get loader statistics"""
        stats = {
            "loaded_patterns": len(self._patterns),
            "patterns": {},
            "cache_dir": str(self.cache_dir),
            "research_dir": str(self.research_dir)
        }
        
        for pattern_type, metadata in self._cache_metadata.items():
            stats["patterns"][pattern_type] = {
                "size": metadata.get("size", 0),
                "loaded_at": metadata.get("loaded_at", "").isoformat() if isinstance(metadata.get("loaded_at"), datetime) else "",
                "source": metadata.get("source_file", "")
            }
        
        return stats