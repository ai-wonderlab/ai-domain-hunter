"""
Color Palette Generation Service
"""
from typing import Dict, List, Optional, Any, Tuple
import logging
import re
import colorsys
from datetime import datetime
import uuid

from services.base_service import BaseService
from core.exceptions import GenerationError, ValidationError
from config.constants import (
    MAX_COLORS_PER_PALETTE,
    MAX_PALETTES_PER_GENERATION,
    COLOR_THEMES,
    WCAG_CONTRAST_RATIO_AA,
    WCAG_CONTRAST_RATIO_AAA
)

logger = logging.getLogger(__name__)


class ColorService(BaseService):
    """Service for generating color palettes"""
    
    async def generate(
        self,
        description: str,
        user_id: str,
        business_name: Optional[str] = None,
        industry: Optional[str] = None,
        theme: Optional[str] = None,
        logo_colors: Optional[List[str]] = None,
        count: int = MAX_PALETTES_PER_GENERATION,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate color palettes
        
        Args:
            description: Business description
            user_id: User ID
            business_name: Business name
            industry: Industry type
            theme: Color theme preference
            logo_colors: Colors from logo to incorporate
            count: Number of palettes to generate
            
        Returns:
            Generated color palettes
        """
        # Validate theme
        if theme and theme not in COLOR_THEMES:
            theme = None
        
        if count > MAX_PALETTES_PER_GENERATION:
            count = MAX_PALETTES_PER_GENERATION
        
        # Get color patterns
        patterns = self.get_patterns("color")
        
        # Generate palettes
        palettes = await self._generate_palettes(
            description, patterns, business_name, industry, theme, logo_colors, count
        )
        
        # Validate and enhance palettes
        validated_palettes = []
        
        for i, palette in enumerate(palettes[:count]):
            try:
                # Validate colors
                validated_colors = self._validate_colors(palette.get('colors', []))
                
                if not validated_colors:
                    continue
                
                # Calculate contrast ratios
                contrasts = self._calculate_contrasts(validated_colors)
                
                # Check accessibility
                accessibility = self._check_accessibility(validated_colors)
                
                validated_palettes.append({
                    "id": str(uuid.uuid4()),
                    "name": palette.get('name', f'Palette {i+1}'),
                    "description": palette.get('description', ''),
                    "theme": palette.get('theme', theme),
                    "colors": validated_colors,
                    "primary_color": validated_colors[0] if validated_colors else None,
                    "secondary_color": validated_colors[1] if len(validated_colors) > 1 else None,
                    "accent_color": validated_colors[2] if len(validated_colors) > 2 else None,
                    "psychology": palette.get('psychology', ''),
                    "use_cases": palette.get('use_cases', []),
                    "contrasts": contrasts,
                    "accessibility": accessibility
                })
                
                logger.info(f"✅ Generated color palette {i+1}")
                
            except Exception as e:
                logger.error(f"Failed to process palette {i+1}: {e}")
                continue
        
        if not validated_palettes:
            raise GenerationError("Failed to generate valid color palettes")
        
        # Save generation
        generation_id = await self.save_generation(
            user_id=user_id,
            generation_type="color",
            input_data={
                "description": description,
                "business_name": business_name,
                "industry": industry,
                "theme": theme,
                "logo_colors": logo_colors,
                "count": count
            },
            output_data={"palettes": validated_palettes},
            cost=0.01
        )
        
        return {
            "palettes": validated_palettes,
            "generation_id": generation_id,
            "count": len(validated_palettes)
        }
    
    async def regenerate(
        self,
        generation_id: str,
        feedback: str,
        user_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Regenerate color palettes with feedback
        """
        # Get original generation
        original = await self.get_generation(generation_id, user_id)
        input_data = original['input_data']
        
        # Generate new palettes based on feedback
        prompt = f"""Based on this feedback: {feedback}

Create 3 new color palettes for this business:
{input_data['description']}

Previous theme: {input_data.get('theme', 'not specified')}

The new palettes should specifically address the user's feedback.

FORMAT AS JSON:
[
    {{
        "name": "Palette Name",
        "colors": [
            {{"hex": "#HEX", "name": "Primary", "rgb": "r,g,b"}},
            ...
        ],
        "description": "Why this palette works",
        "improvement": "How this addresses feedback"
    }}
]"""
        
        response = await self.ai.generate_text(prompt, temperature=0.8)
        palettes = self._extract_json_from_response(response) or []
        
        # Process new palettes
        validated_palettes = []
        for i, palette in enumerate(palettes[:3]):
            colors = self._validate_colors(palette.get('colors', []))
            if colors:
                validated_palettes.append({
                    "id": str(uuid.uuid4()),
                    "name": palette.get('name', f'Improved Palette {i+1}'),
                    "colors": colors,
                    "description": palette.get('description', ''),
                    "improvement": palette.get('improvement', feedback)
                })
        
        # Save new generation
        generation_id = await self.save_generation(
            user_id=user_id,
            generation_type="color",
            input_data={
                **input_data,
                "regeneration": True,
                "feedback": feedback,
                "original_id": generation_id
            },
            output_data={"palettes": validated_palettes},
            cost=0.01
        )
        
        return {
            "palettes": validated_palettes,
            "generation_id": generation_id
        }
    
    async def _generate_palettes(
        self,
        description: str,
        patterns: str,
        business_name: Optional[str],
        industry: Optional[str],
        theme: Optional[str],
        logo_colors: Optional[List[str]],
        count: int
    ) -> List[Dict]:
        """Generate color palettes using AI"""
        
        prompt = f"""{patterns}

Create {count} professional color palettes for:
{f"Business: {business_name}" if business_name else ""}
Description: {description}
"""
        
        if industry:
            prompt += f"Industry: {industry}\n"
        
        if theme:
            prompt += f"Theme Preference: {theme}\n"
        
        if logo_colors:
            prompt += f"Incorporate these colors: {', '.join(logo_colors)}\n"
        
        prompt += f"""
Each palette should include:
- Primary color (main brand color)
- Secondary color (supporting color)
- Accent color (calls-to-action)
- Background color (usually light/white or dark)
- Text color (high contrast with background)
- {MAX_COLORS_PER_PALETTE} colors maximum

Consider:
- Color psychology for the industry
- Accessibility (WCAG contrast ratios)
- Cultural implications
- Print and digital use

FORMAT AS JSON:
[
    {{
        "name": "Palette Name",
        "theme": "vibrant/pastel/monochrome/etc",
        "colors": [
            {{"hex": "#HEX", "name": "Primary", "rgb": "r,g,b", "role": "primary"}},
            {{"hex": "#HEX", "name": "Secondary", "rgb": "r,g,b", "role": "secondary"}},
            {{"hex": "#HEX", "name": "Accent", "rgb": "r,g,b", "role": "accent"}},
            {{"hex": "#HEX", "name": "Background", "rgb": "r,g,b", "role": "background"}},
            {{"hex": "#HEX", "name": "Text", "rgb": "r,g,b", "role": "text"}}
        ],
        "description": "Why this palette works",
        "psychology": "Emotional impact and associations",
        "use_cases": ["website", "print", "merchandise"]
    }}
]

Generate exactly {count} palettes."""
        
        response = await self.ai.generate_text(
            prompt,
            temperature=0.7,
            max_tokens=3000
        )
        
        palettes = self._extract_json_from_response(response)
        
        if not palettes or not isinstance(palettes, list):
            # Generate fallback palettes
            palettes = self._generate_fallback_palettes(count, theme)
        
        return palettes
    
    def _validate_colors(self, colors: List[Any]) -> List[Dict]:
        """Validate and normalize color data"""
        
        validated = []
        
        for color in colors[:MAX_COLORS_PER_PALETTE]:
            if isinstance(color, dict):
                hex_color = color.get('hex', '')
            else:
                hex_color = str(color)
            
            # Validate hex format
            hex_match = re.match(r'^#?([A-Fa-f0-9]{6})$', hex_color)
            if not hex_match:
                continue
            
            # Normalize hex
            hex_color = f"#{hex_match.group(1).upper()}"
            
            # Convert to RGB
            rgb = self._hex_to_rgb(hex_color)
            
            # Convert to HSL
            hsl = self._rgb_to_hsl(*rgb)
            
            validated.append({
                "hex": hex_color,
                "rgb": f"{rgb[0]},{rgb[1]},{rgb[2]}",
                "hsl": f"{hsl[0]},{hsl[1]}%,{hsl[2]}%",
                "name": color.get('name', '') if isinstance(color, dict) else f"Color {len(validated)+1}",
                "role": color.get('role', '') if isinstance(color, dict) else self._guess_role(len(validated))
            })
        
        return validated
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex to RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _rgb_to_hsl(self, r: int, g: int, b: int) -> Tuple[int, int, int]:
        """Convert RGB to HSL"""
        r, g, b = r/255.0, g/255.0, b/255.0
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return int(h*360), int(s*100), int(l*100)
    
    def _calculate_contrasts(self, colors: List[Dict]) -> Dict[str, float]:
        """Calculate contrast ratios between colors"""
        
        contrasts = {}
        
        for i, color1 in enumerate(colors):
            for j, color2 in enumerate(colors):
                if i >= j:
                    continue
                
                ratio = self._contrast_ratio(color1['hex'], color2['hex'])
                key = f"{color1['name']}_{color2['name']}"
                contrasts[key] = round(ratio, 2)
        
        return contrasts
    
    def _contrast_ratio(self, hex1: str, hex2: str) -> float:
        """Calculate WCAG contrast ratio between two colors"""
        
        def relative_luminance(hex_color: str) -> float:
            rgb = self._hex_to_rgb(hex_color)
            
            def channel_luminance(value: int) -> float:
                value = value / 255.0
                if value <= 0.03928:
                    return value / 12.92
                return ((value + 0.055) / 1.055) ** 2.4
            
            r, g, b = [channel_luminance(c) for c in rgb]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        lum1 = relative_luminance(hex1)
        lum2 = relative_luminance(hex2)
        
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        
        return (lighter + 0.05) / (darker + 0.05)
    
    def _check_accessibility(self, colors: List[Dict]) -> Dict[str, str]:
        """Check WCAG accessibility standards"""
        
        results = {}
        
        # Find background and text colors
        bg_color = None
        text_color = None
        
        for color in colors:
            if color['role'] == 'background':
                bg_color = color
            elif color['role'] == 'text':
                text_color = color
        
        if bg_color and text_color:
            ratio = self._contrast_ratio(bg_color['hex'], text_color['hex'])
            
            if ratio >= WCAG_CONTRAST_RATIO_AAA:
                results['text_contrast'] = 'AAA'
            elif ratio >= WCAG_CONTRAST_RATIO_AA:
                results['text_contrast'] = 'AA'
            else:
                results['text_contrast'] = 'Fail'
            
            results['contrast_ratio'] = round(ratio, 2)
        
        return results
    
    def _guess_role(self, index: int) -> str:
        """Guess color role based on position"""
        roles = ['primary', 'secondary', 'accent', 'background', 'text', 'neutral']
        return roles[index] if index < len(roles) else 'additional'
    
    def _generate_fallback_palettes(self, count: int, theme: Optional[str]) -> List[Dict]:
        """Generate fallback palettes if AI fails"""
        
        fallback_schemes = [
            {
                "name": "Professional Blue",
                "theme": "corporate",
                "colors": [
                    {"hex": "#0066CC", "name": "Primary Blue", "role": "primary"},
                    {"hex": "#4D94FF", "name": "Light Blue", "role": "secondary"},
                    {"hex": "#FF6B35", "name": "Orange Accent", "role": "accent"},
                    {"hex": "#FFFFFF", "name": "White", "role": "background"},
                    {"hex": "#333333", "name": "Dark Gray", "role": "text"}
                ]
            },
            {
                "name": "Modern Purple",
                "theme": "vibrant",
                "colors": [
                    {"hex": "#8B5CF6", "name": "Purple", "role": "primary"},
                    {"hex": "#A78BFA", "name": "Light Purple", "role": "secondary"},
                    {"hex": "#10B981", "name": "Green Accent", "role": "accent"},
                    {"hex": "#FAFAFA", "name": "Off White", "role": "background"},
                    {"hex": "#1F2937", "name": "Dark", "role": "text"}
                ]
            },
            {
                "name": "Earth Tones",
                "theme": "earth",
                "colors": [
                    {"hex": "#8B4513", "name": "Brown", "role": "primary"},
                    {"hex": "#DEB887", "name": "Tan", "role": "secondary"},
                    {"hex": "#228B22", "name": "Forest Green", "role": "accent"},
                    {"hex": "#FAF0E6", "name": "Linen", "role": "background"},
                    {"hex": "#2F1B14", "name": "Dark Brown", "role": "text"}
                ]
            }
        ]
        
        return fallback_schemes[:count]