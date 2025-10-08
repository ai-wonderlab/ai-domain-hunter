"""
Package Service - Orchestrates all generation services
"""
from typing import Dict, List, Optional, Any
import logging
import asyncio
from datetime import datetime
import uuid
import re

from services.base_service import BaseService
from services.name_service import NameService
from services.domain_service import DomainService
from services.logo_service import LogoService
from services.color_service import ColorService
from services.tagline_service import TaglineService
from core.exceptions import GenerationError, ValidationError, RateLimitExceeded

logger = logging.getLogger(__name__)


class PackageService(BaseService):
    """Service for generating complete brand packages"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize all services
        self.name_service = NameService()
        self.domain_service = DomainService()
        self.logo_service = LogoService()
        self.color_service = ColorService()
        self.tagline_service = TaglineService()
        
        logger.info("✅ Package Service initialized with all sub-services")
    
    async def generate(
        self,
        description: str,
        user_id: str,
        business_name: Optional[str] = None,
        industry: Optional[str] = None,
        style_preferences: Optional[Dict] = None,
        include_services: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate complete brand package
        
        Args:
            description: Business description
            user_id: User ID
            business_name: Optional pre-selected name
            industry: Industry type
            style_preferences: Style preferences for each component
            include_services: Which services to include
            
        Returns:
            Complete brand package
        """
        # Check usage limit
        if not await self.check_usage_limit(user_id):
            raise RateLimitExceeded("Generation limit exceeded")
        
        # Default to all services
        if include_services is None:
            include_services = ['name', 'domain', 'logo', 'color', 'tagline']
        
        # Parse style preferences
        styles = style_preferences or {}
        
        # Track what's been generated
        package_data = {
            "project_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "description": description,
            "industry": industry
        }
        
        errors = []
        total_cost = 0.0
        
        try:
            # Step 1: Generate or validate business name
            if not business_name and 'name' in include_services:
                logger.info("📝 Generating business names...")
                try:
                    name_result = await self.name_service.generate(
                        description=description,
                        user_id=user_id,
                        industry=industry,
                        style=styles.get('name_style'),
                        count=5
                    )
                    
                    # Use the top name
                    if name_result['names']:
                        business_name = name_result['names'][0]['name']
                        package_data['names'] = name_result['names']
                        package_data['selected_name'] = business_name
                        total_cost += 0.01
                    
                except Exception as e:
                    logger.error(f"Name generation failed: {e}")
                    errors.append({"service": "name", "error": str(e)})
            
            # Ensure we have a business name
            if not business_name:
                raise ValidationError("Business name is required to continue")
            
            package_data['business_name'] = business_name
            
            # Step 2: Check domain availability (parallel)
            domain_task = None
            if 'domain' in include_services:
                logger.info("🌐 Checking domain availability...")
                
                # Generate domain variations
                domains_to_check = self._generate_domain_variations(business_name)
                
                domain_task = asyncio.create_task(
                    self.domain_service.check_domains(
                        domains=domains_to_check,
                        user_id=user_id
                    )
                )
            
            # Step 3: Generate visual identity (parallel)
            logo_task = None
            color_task = None
            
            if 'logo' in include_services:
                logger.info("🎨 Generating logo concepts...")
                logo_task = asyncio.create_task(
                    self.logo_service.generate(
                        description=description,
                        user_id=user_id,
                        business_name=business_name,
                        industry=industry,
                        style=styles.get('logo_style'),
                        count=3
                    )
                )
            
            if 'color' in include_services:
                logger.info("🎨 Generating color palettes...")
                color_task = asyncio.create_task(
                    self.color_service.generate(
                        description=description,
                        user_id=user_id,
                        business_name=business_name,
                        industry=industry,
                        theme=styles.get('color_theme'),
                        count=3
                    )
                )
            
            # Step 4: Generate taglines
            tagline_task = None
            if 'tagline' in include_services:
                logger.info("💬 Generating taglines...")
                tagline_task = asyncio.create_task(
                    self.tagline_service.generate(
                        description=description,
                        user_id=user_id,
                        business_name=business_name,
                        industry=industry,
                        tone=styles.get('tagline_tone'),
                        count=5
                    )
                )
            
            # Wait for all parallel tasks
            tasks = [t for t in [domain_task, logo_task, color_task, tagline_task] if t]
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Task {i} failed: {result}")
                        errors.append({"service": "unknown", "error": str(result)})
                    elif isinstance(result, dict):
                        # Determine which service this is from
                        if 'results' in result and 'domain' in str(result):
                            package_data['domains'] = result.get('results', [])
                        elif 'logos' in result:
                            package_data['logos'] = result.get('logos', [])
                            total_cost += 0.15 * len(result.get('logos', []))
                        elif 'palettes' in result:
                            package_data['color_palettes'] = result.get('palettes', [])
                            total_cost += 0.01
                        elif 'taglines' in result:
                            package_data['taglines'] = result.get('taglines', [])
                            total_cost += 0.01
            
            # Create project record
            project_id = await self._create_project(
                user_id=user_id,
                package_data=package_data
            )
            
            package_data['project_id'] = project_id
            
            # Increment usage
            await self.increment_usage(user_id)
            
            # Save complete generation
            generation_id = await self.save_generation(
                user_id=user_id,
                generation_type="package",
                input_data={
                    "description": description,
                    "business_name": business_name,
                    "industry": industry,
                    "style_preferences": style_preferences,
                    "include_services": include_services
                },
                output_data=package_data,
                project_id=project_id,
                cost=total_cost
            )
            
            package_data['generation_id'] = generation_id
            
            # Add summary
            package_data['summary'] = self._generate_summary(package_data)
            package_data['errors'] = errors
            package_data['total_cost'] = total_cost
            
            logger.info(f"✅ Complete package generated for {business_name}")
            
            return package_data
            
        except Exception as e:
            logger.error(f"Package generation failed: {e}")
            raise GenerationError(f"Failed to generate package: {str(e)}")
    
    async def regenerate(
        self,
        generation_id: str,
        feedback: str,
        user_id: str,
        service_to_regenerate: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Regenerate specific parts of package with feedback
        
        Args:
            generation_id: Original generation ID
            feedback: User feedback
            user_id: User ID
            service_to_regenerate: Specific service to regenerate
            
        Returns:
            Updated package data
        """
        # Get original generation
        original = await self.get_generation(generation_id, user_id)
        
        if not service_to_regenerate:
            # Regenerate everything with feedback
            return await self.generate(
                description=original['input_data']['description'] + f"\n\nUser feedback: {feedback}",
                user_id=user_id,
                business_name=original['output_data'].get('business_name'),
                industry=original['input_data'].get('industry'),
                style_preferences=original['input_data'].get('style_preferences'),
                include_services=original['input_data'].get('include_services')
            )
        
        # Regenerate specific service
        regenerated = {}
        
        if service_to_regenerate == 'logo':
            regenerated = await self.logo_service.regenerate(
                generation_id=generation_id,
                feedback=feedback,
                user_id=user_id
            )
        elif service_to_regenerate == 'color':
            regenerated = await self.color_service.regenerate(
                generation_id=generation_id,
                feedback=feedback,
                user_id=user_id
            )
        elif service_to_regenerate == 'tagline':
            regenerated = await self.tagline_service.regenerate(
                generation_id=generation_id,
                feedback=feedback,
                user_id=user_id
            )
        elif service_to_regenerate == 'name':
            regenerated = await self.name_service.regenerate(
                generation_id=generation_id,
                feedback=feedback,
                user_id=user_id
            )
        
        # Update original package data
        updated_package = original['output_data'].copy()
        updated_package.update(regenerated)
        updated_package['regenerated_service'] = service_to_regenerate
        updated_package['regeneration_feedback'] = feedback
        
        return updated_package
    
    def _generate_domain_variations(self, business_name: str) -> List[str]:
        """Generate domain variations to check"""
        
        # Clean the name
        clean_name = re.sub(r'[^\w\s-]', '', business_name).lower()
        clean_name = re.sub(r'\s+', '', clean_name)
        
        variations = [
            f"{clean_name}.com",
            f"{clean_name}.ai",
            f"{clean_name}.io",
            f"get{clean_name}.com",
            f"{clean_name}app.com",
            f"{clean_name}hq.com",
            f"try{clean_name}.com",
            f"{clean_name}.co",
            f"{clean_name}.net",
            f"{clean_name}.org"
        ]
        
        # Add hyphenated version if multi-word
        if ' ' in business_name:
            hyphenated = business_name.lower().replace(' ', '-')
            variations.extend([
                f"{hyphenated}.com",
                f"{hyphenated}.ai"
            ])
        
        return variations[:20]  # Limit to 20 domains
    
    async def _create_project(
        self,
        user_id: str,
        package_data: Dict
    ) -> str:
        """Create project record in database"""
        
        try:
            project_id = package_data.get('project_id', str(uuid.uuid4()))
            
            result = self.db.table('projects').insert({
                'id': project_id,
                'user_id': user_id,
                'name': package_data.get('business_name', 'Untitled Project'),
                'description': package_data.get('description', ''),
                'status': 'completed',
                'created_at': datetime.now().isoformat()
            }).execute()
            
            return project_id
            
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return str(uuid.uuid4())
    
    def _generate_summary(self, package_data: Dict) -> Dict:
        """Generate package summary"""
        
        summary = {
            "business_name": package_data.get('business_name', 'Unknown'),
            "components_generated": [],
            "recommendations": []
        }
        
        # Check what was generated
        if 'names' in package_data:
            summary['components_generated'].append(f"{len(package_data['names'])} business names")
        
        if 'domains' in package_data:
            available = [d for d in package_data['domains'] if d.get('available')]
            summary['components_generated'].append(f"{len(available)} available domains")
            if available:
                summary['recommendations'].append(
                    f"Register {available[0]['domain']} soon"
                )
        
        if 'logos' in package_data:
            summary['components_generated'].append(f"{len(package_data['logos'])} logo concepts")
        
        if 'color_palettes' in package_data:
            summary['components_generated'].append(f"{len(package_data['color_palettes'])} color palettes")
        
        if 'taglines' in package_data:
            summary['components_generated'].append(f"{len(package_data['taglines'])} taglines")
            if package_data['taglines']:
                summary['featured_tagline'] = package_data['taglines'][0]['text']
        
        return summary