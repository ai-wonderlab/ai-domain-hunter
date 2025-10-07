"""
Logo Generation Service
"""
from typing import Dict, List, Optional, Any
import logging
import base64
from io import BytesIO
from PIL import Image
import uuid
from datetime import datetime

from services.base_service import BaseService
from core.exceptions import GenerationError, ValidationError, ImageGenerationError
from database.client import SupabaseStorage
from config.constants import MAX_LOGOS_PER_GENERATION, LOGO_STYLES
from config.settings import settings

logger = logging.getLogger(__name__)


class LogoService(BaseService):
    """Service for generating logo designs"""
    
    async def generate(
        self,
        description: str,
        user_id: str,
        business_name: str,
        style: Optional[str] = None,
        colors: Optional[List[str]] = None,
        industry: Optional[str] = None,
        count: int = MAX_LOGOS_PER_GENERATION,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate logo concepts
        
        Args:
            description: Business description
            user_id: User ID
            business_name: Name of the business
            style: Design style
            colors: Preferred colors
            industry: Industry type
            count: Number of logos to generate
            
        Returns:
            Generated logos with URLs
        """
        # Validate input
        if not business_name:
            # Try to extract from description
            business_name = await self._extract_business_name(description)
            if not business_name:
                raise ValidationError("Business name is required for logo generation")
        
        if count > MAX_LOGOS_PER_GENERATION:
            count = MAX_LOGOS_PER_GENERATION
        
        # Validate style
        if style and style not in LOGO_STYLES:
            style = "modern"  # Default
        
        # Get logo patterns
        patterns = self.get_patterns("logo")
        
        # Generate logo concepts
        concepts = await self._generate_concepts(
            business_name, description, patterns, style, colors, industry, count
        )
        
        # Generate images for each concept
        generated_logos = []
        
        for i, concept in enumerate(concepts[:count]):
            try:
                # Generate image
                image_bytes = await self._generate_logo_image(
                    business_name, concept, style
                )
                
                # Save to storage
                urls = await self._save_logo_assets(
                    image_bytes,
                    user_id,
                    business_name,
                    f"concept_{i+1}"
                )
                
                generated_logos.append({
                    "id": str(uuid.uuid4()),
                    "concept_name": concept.get('name', f'Concept {i+1}'),
                    "description": concept.get('description', ''),
                    "style": concept.get('style', style),
                    "colors": concept.get('colors', colors or []),
                    "rationale": concept.get('rationale', ''),
                    "urls": urls
                })
                
                logger.info(f"✅ Generated logo concept {i+1} for {business_name}")
                
            except Exception as e:
                logger.error(f"Failed to generate logo {i+1}: {e}")
                continue
        
        if not generated_logos:
            raise GenerationError("Failed to generate any logos")
        
        # Save generation
        generation_id = await self.save_generation(
            user_id=user_id,
            generation_type="logo",
            input_data={
                "business_name": business_name,
                "description": description,
                "style": style,
                "colors": colors,
                "industry": industry,
                "count": count
            },
            output_data={"logos": generated_logos},
            cost=0.15 * len(generated_logos)
        )
        
        return {
            "logos": generated_logos,
            "generation_id": generation_id,
            "business_name": business_name,
            "count": len(generated_logos)
        }
    
    async def regenerate(
        self,
        generation_id: str,
        feedback: str,
        user_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Regenerate logos with feedback
        
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
        
        # Generate new concepts based on feedback
        prompt = f"""Based on this feedback: {feedback}

Create 3 new logo concepts for {input_data['business_name']}.
Previous description: {input_data['description']}

The new designs should specifically address the user's feedback.

Format as JSON:
[
    {{
        "name": "Improved Concept",
        "description": "Detailed visual description",
        "rationale": "How this addresses the feedback",
        "style": "style type",
        "colors": ["#HEX1", "#HEX2"]
    }}
]"""
        
        response = await self.ai.generate_text(prompt, temperature=0.8)
        concepts = self._extract_json_from_response(response) or []
        
        # Generate images for new concepts
        generated_logos = []
        
        for i, concept in enumerate(concepts[:3]):
            try:
                # Generate image
                image_bytes = await self._generate_logo_image(
                    input_data['business_name'],
                    concept,
                    input_data.get('style', 'modern')
                )
                
                # Save to storage
                urls = await self._save_logo_assets(
                    image_bytes,
                    user_id,
                    input_data['business_name'],
                    f"regen_{i+1}"
                )
                
                generated_logos.append({
                    "id": str(uuid.uuid4()),
                    "concept_name": concept.get('name', f'Regenerated {i+1}'),
                    "description": concept.get('description', ''),
                    "urls": urls,
                    "improvement": concept.get('rationale', '')
                })
                
            except Exception as e:
                logger.error(f"Failed to regenerate logo {i+1}: {e}")
                continue
        
        # Save new generation
        generation_id = await self.save_generation(
            user_id=user_id,
            generation_type="logo",
            input_data={
                **input_data,
                "regeneration": True,
                "feedback": feedback,
                "original_id": generation_id
            },
            output_data={"logos": generated_logos},
            cost=0.15 * len(generated_logos)
        )
        
        return {
            "logos": generated_logos,
            "generation_id": generation_id,
            "business_name": input_data['business_name']
        }
    
    async def _extract_business_name(self, description: str) -> Optional[str]:
        """Extract business name from description using AI"""
        
        prompt = f"""Extract the business/company/product name from this description:
"{description}"

If no specific name is mentioned, respond with "NO_NAME_FOUND".
Otherwise respond with ONLY the name, nothing else."""
        
        response = await self.ai.generate_text(prompt, temperature=0.1, max_tokens=50)
        
        if "NO_NAME_FOUND" in response:
            return None
        
        # Clean the response
        name = response.strip().replace('"', '').replace("'", '')
        
        # Validate it looks like a name
        if len(name) > 50 or len(name) < 2:
            return None
        
        return name
    
    async def _generate_concepts(
        self,
        business_name: str,
        description: str,
        patterns: str,
        style: Optional[str],
        colors: Optional[List[str]],
        industry: Optional[str],
        count: int
    ) -> List[Dict]:
        """Generate logo concepts using AI"""
        
        prompt = f"""{patterns}

Create {count} distinct logo concepts for:
Business Name: {business_name}
Description: {description}
"""
        
        if industry:
            prompt += f"Industry: {industry}\n"
        
        if style:
            prompt += f"Preferred Style: {style}\n"
        
        if colors:
            prompt += f"Preferred Colors: {', '.join(colors)}\n"
        
        prompt += f"""
Each concept should be unique and appropriate for the brand.

FORMAT AS JSON:
[
    {{
        "name": "Concept Name",
        "description": "Detailed visual description for image generation",
        "style": "minimalist/modern/playful/etc",
        "colors": ["#primaryHex", "#secondaryHex"],
        "elements": ["icon type", "typography style", "layout"],
        "rationale": "Why this design works for the brand"
    }}
]

Generate exactly {count} concepts."""
        
        # Generate concepts
        response = await self.ai.generate_text(
            prompt,
            temperature=0.8,
            max_tokens=2000
        )
        
        # Parse response
        concepts = self._extract_json_from_response(response)
        
        if not concepts or not isinstance(concepts, list):
            # Fallback concepts
            concepts = [
                {
                    "name": f"Concept {i+1}",
                    "description": f"{style or 'modern'} logo for {business_name}",
                    "style": style or "modern",
                    "colors": colors or ["#000000", "#FFFFFF"]
                }
                for i in range(count)
            ]
        
        return concepts
    
    async def _generate_logo_image(
        self,
        business_name: str,
        concept: Dict,
        style: Optional[str]
    ) -> bytes:
        """Generate logo image from concept"""
        
        # Build image generation prompt
        description = concept.get('description', '')
        concept_style = concept.get('style', style or 'modern')
        
        image_prompt = f"""Professional logo design for "{business_name}":
{description}

Style: {concept_style}, clean, scalable, professional
Background: Pure white
Format: Centered, square composition
Quality: High resolution, vector-like clarity

IMPORTANT: This is a logo, not an illustration. Keep it simple and iconic."""
        
        # Add negative prompt for better results
        negative_prompt = """photograph, realistic, 3d render, complex background, 
        multiple logos, text outside logo, watermark, gradient background"""
        
        try:
            # Generate image
            image_bytes = await self.ai.generate_image(
                prompt=image_prompt,
                style=concept_style,
                negative_prompt=negative_prompt,
                width=512,
                height=512
            )
            
            return image_bytes
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise ImageGenerationError(f"Failed to generate logo image: {str(e)}")
    
    async def _save_logo_assets(
        self,
        image_bytes: bytes,
        user_id: str,
        business_name: str,
        concept_name: str
    ) -> Dict[str, str]:
        """Save logo in multiple formats"""
        
        # Open image with PIL
        image = Image.open(BytesIO(image_bytes))
        
        # Ensure RGBA for transparency support
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        urls = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = f"{user_id}/{business_name}/{concept_name}_{timestamp}"
        
        # Save PNG (with transparency)
        png_buffer = BytesIO()
        image.save(png_buffer, format='PNG', optimize=True)
        png_bytes = png_buffer.getvalue()
        
        png_path = f"{base_path}.png"
        urls['png'] = await SupabaseStorage.upload_file(
            settings.storage_bucket,
            png_path,
            png_bytes,
            "image/png"
        )
        
        # Save JPG (white background)
        jpg_image = Image.new('RGB', image.size, (255, 255, 255))
        jpg_image.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
        
        jpg_buffer = BytesIO()
        jpg_image.save(jpg_buffer, format='JPEG', quality=95)
        jpg_bytes = jpg_buffer.getvalue()
        
        jpg_path = f"{base_path}.jpg"
        urls['jpg'] = await SupabaseStorage.upload_file(
            settings.storage_bucket,
            jpg_path,
            jpg_bytes,
            "image/jpeg"
        )
        
        # TODO: Add SVG conversion using vectorization library
        # For now, we'll skip SVG
        
        return urls