"""
Business Name Generation Service
"""
from typing import Dict, List, Optional, Any
import logging
import re
from datetime import datetime

from services.base_service import BaseService
from core.exceptions import GenerationError, ValidationError
from config.constants import (
    MAX_NAMES_PER_GENERATION,
    BLOCKED_NAME_PATTERNS,
    MAX_NAME_LENGTH
)

logger = logging.getLogger(__name__)


class NameService(BaseService):
    """Service for generating business names"""
    
    async def generate(
        self,
        description: str,
        user_id: str,
        industry: Optional[str] = None,
        style: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        count: int = MAX_NAMES_PER_GENERATION,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate business names
        
        Args:
            description: Business description
            user_id: User ID
            industry: Industry type
            style: Naming style (modern, playful, etc)
            keywords: Keywords to consider
            count: Number of names to generate
            
        Returns:
            Generated names with metadata
        """
        # Validate input
        if not description or len(description) < 10:
            raise ValidationError("Description must be at least 10 characters")
        
        if count > MAX_NAMES_PER_GENERATION:
            count = MAX_NAMES_PER_GENERATION
        
        # Get naming patterns
        patterns = self.get_patterns("name")
        
        # Build prompt
        prompt = self._build_prompt(
            description, patterns, industry, style, keywords, count
        )
        
        # Generate names
        try:
            # Use parallel generation for variety
            prompts = [
                (prompt, "model_1"),
                (prompt, "model_2") if len(self.ai.text_models) > 1 else (prompt, "model_1")
            ]
            
            responses = await self.ai.generate_parallel(prompts, temperature=0.8)
            
            # Parse and combine names
            all_names = []
            for response in responses.values():
                if response:
                    parsed_names = self._parse_names(response)
                    all_names.extend(parsed_names)
            
            # Deduplicate and validate
            unique_names = self._deduplicate_names(all_names)
            valid_names = self._validate_names(unique_names)
            
            # Score and sort
            scored_names = self._score_names(valid_names, description)
            scored_names.sort(key=lambda x: x['score'], reverse=True)
            
            # Take top N
            final_names = scored_names[:count]
            
            # Save generation
            generation_id = await self.save_generation(
                user_id=user_id,
                generation_type="name",
                input_data={
                    "description": description,
                    "industry": industry,
                    "style": style,
                    "keywords": keywords,
                    "count": count
                },
                output_data={"names": final_names},
                cost=0.01
            )
            
            logger.info(f"✅ Generated {len(final_names)} names for user {user_id}")
            
            return {
                "names": final_names,
                "generation_id": generation_id,
                "count": len(final_names)
            }
            
        except Exception as e:
            logger.error(f"Name generation failed: {e}")
            raise GenerationError(f"Failed to generate names: {str(e)}")
    
    async def regenerate(
        self,
        generation_id: str,
        feedback: str,
        user_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Regenerate names with feedback
        
        Args:
            generation_id: Original generation ID
            feedback: User feedback
            user_id: User ID
            
        Returns:
            New generation results
        """
        # Get original generation
        original = await self.get_generation(generation_id, user_id)
        input_data = original['input_data']
        
        # Add feedback to prompt
        prompt = self._build_regeneration_prompt(
            input_data['description'],
            original['output_data']['names'],
            feedback
        )
        
        # Generate new names
        try:
            response = await self.ai.generate_text(
                prompt,
                temperature=0.9,
                max_tokens=1500
            )
            
            # Parse names
            new_names = self._parse_names(response)
            valid_names = self._validate_names(new_names)
            
            # Save new generation
            generation_id = await self.save_generation(
                user_id=user_id,
                generation_type="name",
                input_data={
                    **input_data,
                    "regeneration": True,
                    "feedback": feedback,
                    "original_id": generation_id
                },
                output_data={"names": valid_names},
                cost=0.01
            )
            
            return {
                "names": valid_names,
                "generation_id": generation_id,
                "count": len(valid_names)
            }
            
        except Exception as e:
            logger.error(f"Name regeneration failed: {e}")
            raise GenerationError(f"Failed to regenerate names: {str(e)}")
    
    def _build_prompt(
        self,
        description: str,
        patterns: str,
        industry: Optional[str],
        style: Optional[str],
        keywords: Optional[List[str]],
        count: int
    ) -> str:
        """Build generation prompt"""
        
        prompt = f"""{patterns}

Generate {count} creative business names for:

BUSINESS DESCRIPTION:
{description}

REQUIREMENTS:
- Names must be unique and brandable
- Length: 3-{MAX_NAME_LENGTH} characters
- Easy to pronounce and remember
- Check for potential trademark issues
- Consider .com and .ai domain availability

"""
        
        if industry:
            prompt += f"INDUSTRY: {industry}\n"
        
        if style:
            prompt += f"STYLE: {style} (modern, playful, professional, etc)\n"
        
        if keywords:
            prompt += f"KEYWORDS TO CONSIDER: {', '.join(keywords)}\n"
        
        prompt += """
FORMAT YOUR RESPONSE AS JSON:
[
    {
        "name": "BusinessName",
        "reasoning": "Why this name works",
        "style": "modern/playful/professional",
        "length": 10,
        "memorability": 8,
        "pronounceability": 9,
        "uniqueness": 8
    },
    ...
]

Generate exactly """ + str(count) + """ names."""
        
        return prompt
    
    def _build_regeneration_prompt(
        self,
        description: str,
        original_names: List[Dict],
        feedback: str
    ) -> str:
        """Build regeneration prompt with feedback"""
        
        original_list = ", ".join([n['name'] for n in original_names[:5]])
        
        prompt = f"""You previously generated these names:
{original_list}

User feedback: {feedback}

Generate 10 NEW business names based on this feedback for:
{description}

Make sure the new names address the user's feedback.
Avoid names similar to the original ones unless specifically requested.

FORMAT AS JSON:
[
    {{
        "name": "NewName",
        "reasoning": "How this addresses the feedback",
        "improvement": "What's better about this name"
    }},
    ...
]
"""
        
        return prompt
    
    def _parse_names(self, response: str) -> List[Dict]:
        """Parse names from AI response"""
        
        # Try JSON parsing first
        parsed = self._extract_json_from_response(response)
        
        if parsed and isinstance(parsed, list):
            return parsed
        
        # Fallback: manual parsing
        names = []
        
        # Look for name patterns
        name_pattern = r'"name"\s*:\s*"([^"]+)"'
        matches = re.findall(name_pattern, response)
        
        for match in matches:
            names.append({
                "name": match.strip(),
                "reasoning": "Extracted from response",
                "score": 5.0
            })
        
        # If still no names, try line-by-line
        if not names:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) <= MAX_NAME_LENGTH:
                    # Check if it looks like a name
                    if re.match(r'^[A-Za-z][A-Za-z0-9\s]{2,}$', line):
                        names.append({
                            "name": line,
                            "reasoning": "Extracted from response",
                            "score": 5.0
                        })
        
        return names
    
    def _validate_names(self, names: List[Dict]) -> List[Dict]:
        """Validate and filter names"""
        
        valid_names = []
        
        for name_data in names:
            name = name_data.get('name', '')
            
            # Clean the name
            name = re.sub(r'[^\w\s-]', '', name).strip()
            
            # Check length
            if len(name) < 3 or len(name) > MAX_NAME_LENGTH:
                continue
            
            # Check blocked patterns
            name_lower = name.lower()
            if any(blocked in name_lower for blocked in BLOCKED_NAME_PATTERNS):
                continue
            
            # Check for numbers at start
            if re.match(r'^\d', name):
                continue
            
            # Update name in dict
            name_data['name'] = name
            valid_names.append(name_data)
        
        return valid_names
    
    def _deduplicate_names(self, names: List[Dict]) -> List[Dict]:
        """Remove duplicate names"""
        
        seen = set()
        unique = []
        
        for name_data in names:
            name = name_data.get('name', '').lower()
            if name and name not in seen:
                seen.add(name)
                unique.append(name_data)
        
        return unique
    
    def _score_names(self, names: List[Dict], description: str) -> List[Dict]:
        """Score names based on various criteria"""
        
        for name_data in names:
            name = name_data.get('name', '')
            score = 5.0  # Base score
            
            # Length score (shorter is better for domains)
            if len(name) <= 8:
                score += 2.0
            elif len(name) <= 12:
                score += 1.0
            
            # Single word bonus
            if ' ' not in name:
                score += 1.5
            
            # Pronounceability (simple heuristic)
            vowels = len(re.findall(r'[aeiouAEIOU]', name))
            consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]', name))
            if vowels > 0 and consonants > 0:
                ratio = vowels / (vowels + consonants)
                if 0.3 <= ratio <= 0.5:  # Good vowel ratio
                    score += 1.0
            
            # Memorability (no complex patterns)
            if not re.search(r'(.)\1{2,}', name):  # No triple letters
                score += 0.5
            
            # AI suffix penalty (too generic)
            if name.lower().endswith('ai'):
                score -= 1.0
            
            # Update score
            name_data['score'] = min(10.0, max(0.0, score))
        
        return names