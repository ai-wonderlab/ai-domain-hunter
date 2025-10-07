"""
Tagline Generation Service
"""
from typing import Dict, List, Optional, Any
import logging
import re
from datetime import datetime
import uuid

from services.base_service import BaseService
from core.exceptions import GenerationError, ValidationError
from config.constants import (
    MAX_TAGLINES_PER_GENERATION,
    MAX_TAGLINE_LENGTH,
    TAGLINE_TONES
)

logger = logging.getLogger(__name__)


class TaglineService(BaseService):
    """Service for generating taglines"""
    
    async def generate(
        self,
        description: str,
        user_id: str,
        business_name: str,
        tone: Optional[str] = None,
        industry: Optional[str] = None,
        target_audience: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        count: int = MAX_TAGLINES_PER_GENERATION,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate taglines
        
        Args:
            description: Business description
            user_id: User ID
            business_name: Business name (required)
            tone: Desired tone
            industry: Industry type
            target_audience: Target audience description
            keywords: Keywords to consider
            count: Number of taglines
            
        Returns:
            Generated taglines
        """
        # Validate input
        if not business_name:
            raise ValidationError("Business name is required for tagline generation")
        
        # Validate tone
        if tone and tone not in TAGLINE_TONES:
            tone = "professional"  # Default
        
        if count > MAX_TAGLINES_PER_GENERATION:
            count = MAX_TAGLINES_PER_GENERATION
        
        # Get tagline patterns
        patterns = self.get_patterns("tagline")
        
        # Generate taglines
        taglines = await self._generate_taglines(
            business_name, description, patterns, tone, 
            industry, target_audience, keywords, count
        )
        
        # Process and validate taglines
        processed_taglines = []
        
        for i, tagline in enumerate(taglines[:count]):
            try:
                # Validate tagline
                text = tagline.get('text', '')
                if not self._validate_tagline(text):
                    continue
                
                # Analyze tagline
                analysis = self._analyze_tagline(text)
                
                processed_taglines.append({
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "tone": tagline.get('tone', tone),
                    "style": tagline.get('style', ''),
                    "reasoning": tagline.get('reasoning', ''),
                    "target_emotion": tagline.get('emotion', ''),
                    "call_to_action": tagline.get('cta', False),
                    "word_count": analysis['word_count'],
                    "character_count": analysis['character_count'],
                    "readability": analysis['readability'],
                    "use_cases": tagline.get('use_cases', [])
                })
                
                logger.info(f"✅ Generated tagline {i+1}: {text[:30]}...")
                
            except Exception as e:
                logger.error(f"Failed to process tagline {i+1}: {e}")
                continue
        
        if not processed_taglines:
            raise GenerationError("Failed to generate valid taglines")
        
        # Save generation
        generation_id = await self.save_generation(
            user_id=user_id,
            generation_type="tagline",
            input_data={
                "business_name": business_name,
                "description": description,
                "tone": tone,
                "industry": industry,
                "target_audience": target_audience,
                "keywords": keywords,
                "count": count
            },
            output_data={"taglines": processed_taglines},
            cost=0.01
        )
        
        return {
            "taglines": processed_taglines,
            "generation_id": generation_id,
            "business_name": business_name,
            "count": len(processed_taglines)
        }
    
    async def regenerate(
        self,
        generation_id: str,
        feedback: str,
        user_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Regenerate taglines with feedback
        """
        # Get original generation
        original = await self.get_generation(generation_id, user_id)
        input_data = original['input_data']
        
        # Generate new taglines based on feedback
        prompt = f"""Based on this feedback: {feedback}

Create 5 new taglines for {input_data['business_name']}.
Business description: {input_data['description']}
Previous tone: {input_data.get('tone', 'not specified')}

The new taglines should specifically address the user's feedback.

FORMAT AS JSON:
[
    {{
        "text": "New tagline here",
        "tone": "professional/friendly/bold",
        "reasoning": "Why this addresses the feedback",
        "improvement": "What makes this better"
    }}
]"""
        
        response = await self.ai.generate_text(prompt, temperature=0.9)
        taglines = self._extract_json_from_response(response) or []
        
        # Process new taglines
        processed_taglines = []
        for tagline in taglines[:5]:
            text = tagline.get('text', '')
            if self._validate_tagline(text):
                processed_taglines.append({
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "tone": tagline.get('tone', input_data.get('tone', 'professional')),
                    "reasoning": tagline.get('reasoning', ''),
                    "improvement": tagline.get('improvement', feedback)
                })
        
        # Save new generation
        generation_id = await self.save_generation(
            user_id=user_id,
            generation_type="tagline",
            input_data={
                **input_data,
                "regeneration": True,
                "feedback": feedback,
                "original_id": generation_id
            },
            output_data={"taglines": processed_taglines},
            cost=0.01
        )
        
        return {
            "taglines": processed_taglines,
            "generation_id": generation_id,
            "business_name": input_data['business_name']
        }
    
    async def _generate_taglines(
        self,
        business_name: str,
        description: str,
        patterns: str,
        tone: Optional[str],
        industry: Optional[str],
        target_audience: Optional[str],
        keywords: Optional[List[str]],
        count: int
    ) -> List[Dict]:
        """Generate taglines using AI"""
        
        prompt = f"""{patterns}

Create {count} taglines for {business_name}.

BUSINESS DETAILS:
Description: {description}
"""
        
        if industry:
            prompt += f"Industry: {industry}\n"
        
        if tone:
            prompt += f"Desired Tone: {tone}\n"
        else:
            prompt += "Tone: Professional but approachable\n"
        
        if target_audience:
            prompt += f"Target Audience: {target_audience}\n"
        
        if keywords:
            prompt += f"Key Concepts: {', '.join(keywords)}\n"
        
        prompt += f"""
REQUIREMENTS:
- Maximum length: {MAX_TAGLINE_LENGTH} characters
- Clear value proposition
- Memorable and unique
- Appropriate for the brand
- Emotionally resonant

Create diverse taglines using these structures:
1. Value-focused: Emphasize the benefit
2. Action-oriented: Start with a verb
3. Question-based: Engage curiosity
4. Descriptive: Explain what you do
5. Aspirational: Paint a vision

FORMAT AS JSON:
[
    {{
        "text": "Your tagline here",
        "tone": "professional/friendly/bold/inspirational",
        "style": "value/action/question/descriptive/aspirational",
        "reasoning": "Why this works for the brand",
        "emotion": "Primary emotion it evokes",
        "cta": true/false,
        "use_cases": ["website header", "business card", "ad campaign"]
    }}
]

Generate exactly {count} unique taglines."""
        
        response = await self.ai.generate_text(
            prompt,
            temperature=0.9,  # Higher for creativity
            max_tokens=2000
        )
        
        taglines = self._extract_json_from_response(response)
        
        if not taglines or not isinstance(taglines, list):
            # Generate fallback taglines
            taglines = self._generate_fallback_taglines(
                business_name, description, count
            )
        
        return taglines
    
    def _validate_tagline(self, text: str) -> bool:
        """Validate tagline text"""
        
        if not text:
            return False
        
        # Check length
        if len(text) > MAX_TAGLINE_LENGTH:
            return False
        
        # Check for minimum length
        if len(text) < 5:
            return False
        
        # Check for inappropriate content (basic filter)
        inappropriate_patterns = [
            r'\b(xxx|porn|drug)\b',
            r'[^\w\s\.\,\!\?\-\']',  # Allow only standard punctuation
        ]
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        return True
    
    def _analyze_tagline(self, text: str) -> Dict:
        """Analyze tagline characteristics"""
        
        words = text.split()
        
        # Basic readability (Flesch Reading Ease approximation)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Simple readability score
        if avg_word_length <= 4:
            readability = "Very Easy"
        elif avg_word_length <= 5:
            readability = "Easy"
        elif avg_word_length <= 6:
            readability = "Moderate"
        else:
            readability = "Complex"
        
        return {
            "word_count": len(words),
            "character_count": len(text),
            "readability": readability,
            "has_punctuation": bool(re.search(r'[.!?]', text)),
            "starts_with_verb": self._starts_with_verb(text),
            "is_question": text.strip().endswith('?')
        }
    
    def _starts_with_verb(self, text: str) -> bool:
        """Check if tagline starts with a verb (simple heuristic)"""
        
        common_verbs = [
            'be', 'build', 'create', 'discover', 'empower', 'find',
            'get', 'grow', 'imagine', 'join', 'learn', 'make',
            'transform', 'unlock', 'achieve', 'connect', 'explore'
        ]
        
        first_word = text.split()[0].lower() if text.split() else ""
        return first_word in common_verbs
    
    def _generate_fallback_taglines(
        self,
        business_name: str,
        description: str,
        count: int
    ) -> List[Dict]:
        """Generate fallback taglines if AI fails"""
        
        templates = [
            {
                "template": f"{business_name} - Your partner in success",
                "style": "descriptive",
                "tone": "professional"
            },
            {
                "template": f"Empowering your journey with {business_name}",
                "style": "aspirational",
                "tone": "inspirational"
            },
            {
                "template": f"Where innovation meets excellence",
                "style": "value",
                "tone": "professional"
            },
            {
                "template": f"Transform your tomorrow, today",
                "style": "action",
                "tone": "bold"
            },
            {
                "template": f"Solutions that work for you",
                "style": "value",
                "tone": "friendly"
            }
        ]
        
        taglines = []
        for i, template in enumerate(templates[:count]):
            taglines.append({
                "text": template["template"],
                "style": template["style"],
                "tone": template["tone"],
                "reasoning": "Fallback tagline",
                "emotion": "trust",
                "cta": False,
                "use_cases": ["general"]
            })
        
        return taglines