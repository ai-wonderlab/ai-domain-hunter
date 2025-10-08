"""
Domain Availability Checking Service
"""
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
import dns.resolver
import logging
from datetime import datetime

from services.base_service import BaseService
from core.exceptions import ValidationError, ServiceUnavailableError
from config.settings import settings
from config.constants import MAX_DOMAINS_TO_CHECK

logger = logging.getLogger(__name__)


class DomainService(BaseService):
    """Service for checking domain availability"""
    
    def __init__(self):
        super().__init__()
        self.domain_apis = settings.get_domain_apis()
        self.session = None

    async def generate(
        self,
        description: str,
        user_id: str,
        business_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate domain variations using AI and check until 10 available found
        
        Args:
            description: Business description (for context)
            user_id: User ID
            business_name: Base business name to generate variations from
            
        Returns:
            10 available domains (or best effort)
        """
        available = []
        checked_variations = set()
        round_num = 0
        max_rounds = 5
        all_checked_domains = []
        
        logger.info(f"🔍 Starting domain hunt for: {business_name}")
        
        # ✅ STEP 1: Check original name FIRST
        original_base = business_name.lower().replace(' ', '').replace('-', '')
        original_domains = [f"{original_base}.com", f"{original_base}.ai"]
        
        logger.info(f"🔎 Checking original name first: {original_base}")
        result = await self.check_domains(original_domains, user_id)
        
        for r in result['results']:
            if r['available'] and len(available) < 10:
                available.append(r)
        
        all_checked_domains.extend(original_domains)
        checked_variations.add(original_base)
        
        logger.info(f"✅ Original name check: {len(available)}/10 available")
        
        # ✅ STEP 2: Generate variations if needed
        while len(available) < 10 and round_num < max_rounds:
            round_num += 1
            logger.info(f"📍 Round {round_num}: Generating variations...")
            
            # Generate 10 new variations using AI
            variations = await self._generate_variations(
                business_name,
                description,
                checked_variations,
                count=10
            )
            
            # Build domains to check (.com + .ai for each variation)
            domains_to_check = []
            for v in variations:
                domains_to_check.append(f"{v}.com")
                domains_to_check.append(f"{v}.ai")
            
            logger.info(f"🔎 Checking {len(domains_to_check)} domains...")
            
            # Check availability with BOTH DNS and WhoAPI
            result = await self.check_domains(domains_to_check, user_id)
            
            # Collect available ones
            for r in result['results']:
                if r['available'] and len(available) < 10:
                    available.append(r)
            
            all_checked_domains.extend(domains_to_check)
            checked_variations.update(variations)
            
            logger.info(f"✅ Round {round_num}: Found {len(available)}/10 available")
            
            if len(available) >= 10:
                break
        
        # Save final generation
        generation_id = await self.save_generation(
            user_id=user_id,
            generation_type="domain",
            input_data={
                "business_name": business_name,
                "description": description,
                "checked_variations": list(checked_variations)
            },
            output_data={"results": available[:10]},
            cost=0.00
        )
        
        logger.info(f"🎉 Domain hunt complete: {len(available)} available domains found in {round_num} rounds")
        
        return {
            "results": available[:10],  # Return exactly 10 (or less if couldn't find)
            "generation_id": generation_id,
            "total_checked": len(all_checked_domains),
            "available_count": len(available),
            "rounds": round_num
        }   

    async def check_domains(
        self,
        domains: List[str],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Check availability of multiple domains using BOTH DNS and WhoAPI
        
        Args:
            domains: List of domain names
            user_id: User ID
            
        Returns:
            Availability information for each domain
        """
        # Validate input
        if not domains:
            raise ValidationError("No domains provided")
        
        if len(domains) > MAX_DOMAINS_TO_CHECK:
            domains = domains[:MAX_DOMAINS_TO_CHECK]
        
        # Clean domains
        clean_domains = self._clean_domains(domains)
        
        results = {}
        
        # STEP 1: DNS check (fast, parallel)
        logger.info(f"🔍 DNS checking {len(clean_domains)} domains...")
        dns_results = await self._parallel_dns_check(clean_domains)
        results.update(dns_results)
        
        # STEP 2: WhoAPI check for ALL domains (not just DNS available ones)
        if self.domain_apis:
            logger.info(f"🔍 WhoAPI checking {len(clean_domains)} domains...")
            api_results = await self._check_with_apis(clean_domains)
            
            # Merge results - WhoAPI overrides DNS
            for domain, api_data in api_results.items():
                if domain in results:
                    results[domain].update(api_data)
        
        # Format final results
        formatted_results = []
        for domain, data in results.items():
            # A domain is available if EITHER DNS or WhoAPI says so
            dns_available = data.get('dns_available', False)
            api_available = data.get('available', False)
            
            final_available = dns_available or api_available
            
            formatted_results.append({
                "domain": domain,
                "available": final_available,
                "status": data.get('status', 'unknown'),
                "price": data.get('price', self._get_default_price(domain)),
                "registrar": data.get('registrar'),
                "registrar_link": self._get_registrar_link(domain),
                "checked_at": data.get('checked_at', datetime.now().isoformat()),
                "method": data.get('method', 'dns')
            })
        
        # Save generation (for standalone check_domains calls)
        generation_id = await self.save_generation(
            user_id=user_id,
            generation_type="domain",
            input_data={"domains": clean_domains},
            output_data={"results": formatted_results},
            cost=0.00
        )
        
        logger.info(f"✅ Checked {len(clean_domains)} domains for user {user_id}")
        
        return {
            "results": formatted_results,
            "generation_id": generation_id,
            "total_checked": len(clean_domains),
            "available_count": len([r for r in formatted_results if r['available']])
        }

    async def regenerate(
        self,
        business_name: str,
        description: str,
        feedback: str,
        exclude_domains: List[str],
        user_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Regenerate domains with user feedback
        
        Args:
            business_name: Original business name
            description: Business description
            feedback: User feedback on what they want
            exclude_domains: Domains to exclude from generation
            user_id: User ID
        """
        # Extract already checked variations from exclude_domains
        checked_variations = set()
        for domain in exclude_domains:
            # Remove TLD to get base name
            base = domain.split('.')[0]
            checked_variations.add(base)
        
        logger.info(f"🔄 Regenerating with feedback: {feedback}")
        logger.info(f"🚫 Excluding {len(checked_variations)} variations")
        
        # Call generate with enhanced description including feedback
        enhanced_description = f"{description}\n\nUser preferences: {feedback}"
        
        # Generate new variations (skipping already checked ones)
        available = []
        round_num = 0
        max_rounds = 3
        
        while len(available) < 10 and round_num < max_rounds:
            round_num += 1
            
            variations = await self._generate_variations(
                business_name,
                enhanced_description,
                checked_variations,
                count=10
            )
            
            # Check .com and .ai for each
            domains_to_check = []
            for v in variations:
                domains_to_check.append(f"{v}.com")
                domains_to_check.append(f"{v}.ai")
            
            result = await self.check_domains(domains_to_check, user_id)
            
            for r in result['results']:
                if r['available'] and len(available) < 10:
                    available.append(r)
            
            checked_variations.update(variations)
            
            if len(available) >= 10:
                break
        
        # Save generation
        generation_id = await self.save_generation(
            user_id=user_id,
            generation_type="domain_regenerate",
            input_data={
                "business_name": business_name,
                "feedback": feedback,
                "excluded": list(exclude_domains)
            },
            output_data={"results": available[:10]},
            cost=0.00
        )
        
        return {
            "results": available[:10],
            "generation_id": generation_id,
            "total_checked": round_num * 20,
            "available_count": len(available),
            "rounds": round_num
        }    

    async def _generate_variations(
        self,
        business_name: str,
        description: str,
        already_checked: set,
        count: int = 10
    ) -> List[str]:
        """Use AI to generate domain name variations"""
        
        base = business_name.lower().replace(' ', '').replace('-', '')
        
        prompt = f"""Generate {count} SHORT domain variations based on: "{business_name}"

    Business: {description}

    CRITICAL RULES:
    1. Stay CLOSE to "{business_name}" - these are VARIATIONS not new names!
    2. Try these patterns FIRST:
    - Add prefix: get{base}, my{base}, use{base}, try{base}
    - Add suffix: {base}app, {base}hq, {base}io, {base}hub
    - Remove vowels: {base.translate(str.maketrans('', '', 'aeiou'))}
    - Abbreviate: first 3-4 letters of each word
    - Simple changes: {base}ly, {base}fy, {base}er
    3. Length: 4-12 characters max
    4. Must be pronounceable and memorable
    5. Avoid: {list(already_checked)[:5] if already_checked else 'None'}

    Return ONLY base names (no .com/.ai), one per line, no numbers or explanations."""
        
        try:
            response = await self.ai.generate_text(
                prompt=prompt,
                max_tokens=200
            )
            
            # Parse variations
            variations = []
            for line in response.split('\n'):
                # Clean the line
                name = line.strip().lower()
                # Remove numbers, bullets, dashes at start
                name = ''.join(c for c in name if c.isalnum() or c in '-_')
                name = name.strip('-_')
                
                # Validate
                if name and len(name) >= 3 and len(name) <= 20 and name not in already_checked:
                    variations.append(name)
            
            if not variations:
                # Fallback: simple variations
                variations = [
                    base,
                    f"get{base}",
                    f"{base}app",
                    f"{base}hq",
                    f"{base}io",
                    f"my{base}",
                    f"{base}hub",
                    f"use{base}",
                    f"{base}ly",
                    f"try{base}"
                ]
            
            logger.info(f"✨ Generated {len(variations)} variations: {variations[:5]}...")
            return variations[:count]
            
        except Exception as e:
            logger.error(f"AI variation generation failed: {e}")
            # Fallback to simple variations
            return [
                base,
                f"get{base}",
                f"{base}app",
                f"{base}hq",
                f"{base}io",
                f"my{base}",
                f"{base}hub",
                f"use{base}",
                f"{base}pro",
                f"try{base}"
            ][:count]

    def _clean_domains(self, domains: List[str]) -> List[str]:
        """Clean and normalize domain names"""
        
        clean = []
        for domain in domains:
            # Convert to lowercase
            domain = domain.lower().strip()
            
            # Add .com if no extension
            if '.' not in domain:
                domain += '.com'
            
            # Remove protocol if present
            domain = domain.replace('http://', '').replace('https://', '')
            
            # Remove www
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Remove path
            if '/' in domain:
                domain = domain.split('/')[0]
            
            clean.append(domain)
        
        return clean
    
    async def _parallel_dns_check(self, domains: List[str]) -> Dict[str, Dict]:
        """Check domains using DNS (parallel)"""
        
        async def check_single(domain: str) -> tuple:
            try:
                # Run DNS query in thread pool
                await asyncio.to_thread(
                    dns.resolver.resolve, domain, 'A'
                )
                
                return domain, {
                    'dns_available': False,
                    'available': False,
                    'status': 'registered',
                    'method': 'dns',
                    'checked_at': datetime.now().isoformat()
                }
            except:
                # No DNS record found - might be available
                return domain, {
                    'dns_available': True,
                    'available': True,  # Will be overridden by WhoAPI
                    'status': 'possibly_available',
                    'method': 'dns',
                    'checked_at': datetime.now().isoformat()
                }
        
        # Run all checks in parallel
        tasks = [check_single(domain) for domain in domains]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    async def _check_with_apis(self, domains: List[str]) -> Dict[str, Dict]:
        """Check domains using configured APIs (WhoAPI)"""
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        results = {}
        
        for api_config in self.domain_apis:
            provider = api_config['provider'].lower()
            
            if provider == 'whoisxml':
                api_results = await self._check_whoisxml(domains, api_config)
            elif provider == 'whoapi':
                api_results = await self._check_whoapi(domains, api_config)
            else:
                continue
            
            results.update(api_results)
            
            # If we have results for all domains, stop
            if len(results) >= len(domains):
                break
        
        return results
    
    async def _check_whoisxml(
        self,
        domains: List[str],
        config: Dict
    ) -> Dict[str, Dict]:
        """Check using WhoisXML API"""
        
        results = {}
        
        for domain in domains:
            try:
                url = config['url']
                params = {
                    'apiKey': config['key'],
                    'domainName': domain,
                    'credits': 'DA'
                }
                
                async with self.session.get(
                    url,
                    params=params,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        domain_info = data.get('DomainInfo', {})
                        available = domain_info.get('domainAvailability') == 'AVAILABLE'
                        
                        results[domain] = {
                            'available': available,
                            'status': 'available' if available else 'registered',
                            'method': 'whoisxml_api',
                            'checked_at': datetime.now().isoformat()
                        }
                
                # Rate limit
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"WhoisXML check failed for {domain}: {e}")
        
        return results
    
    async def _check_whoapi(
        self,
        domains: List[str],
        config: Dict
    ) -> Dict[str, Dict]:
        """Check using WhoAPI"""
        
        results = {}
        
        for domain in domains:
            try:
                url = config['url']
                params = {
                    'domain': domain,
                    'r': 'taken',
                    'apikey': config['key']
                }
                
                async with self.session.get(
                    url,
                    params=params,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        taken = str(data.get('taken')) == '1'
                        available = not taken
                        
                        results[domain] = {
                            'available': available,
                            'status': 'available' if available else 'registered',
                            'method': 'whoapi',
                            'checked_at': datetime.now().isoformat()
                        }
                
                # Rate limit
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.error(f"WhoAPI check failed for {domain}: {e}")
        
        return results
    
    def _get_default_price(self, domain: str) -> str:
        """Get default price based on TLD"""
        
        tld_prices = {
            '.com': '$12.99/year',
            '.ai': '$90/year',
            '.io': '$45/year',
            '.net': '$14.99/year',
            '.org': '$12.99/year',
            '.co': '$29.99/year',
            '.app': '$19.99/year',
            '.dev': '$15.99/year'
        }
        
        for tld, price in tld_prices.items():
            if domain.endswith(tld):
                return price
        
        return '$15.99/year'
    
    def _get_registrar_link(self, domain: str) -> str:
        """Generate registrar link (can add affiliates here)"""
        
        # You can add affiliate parameters
        base_urls = {
            'namecheap': f'https://www.namecheap.com/domains/registration/results/?domain={domain}',
            'godaddy': f'https://www.godaddy.com/domainsearch/find?domainToCheck={domain}',
            'google': f'https://domains.google.com/registrar/search?searchTerm={domain}'
        }
        
        # Return Namecheap by default
        return base_urls['namecheap']
    
    async def _check_whoapi(
        self,
        domains: List[str],
        config: Dict
    ) -> Dict[str, Dict]:
        """Check using WhoAPI (parallel with rate limiting)"""
        
        async def check_single(domain: str) -> tuple:
            try:
                url = config['url']
                params = {
                    'domain': domain,
                    'r': 'taken',
                    'apikey': config['key']
                }
                
                async with self.session.get(
                    url,
                    params=params,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        taken = str(data.get('taken')) == '1'
                        available = not taken
                        
                        return domain, {
                            'available': available,
                            'status': 'available' if available else 'registered',
                            'method': 'whoapi',
                            'checked_at': datetime.now().isoformat()
                        }
                        
            except Exception as e:
                logger.error(f"WhoAPI check failed for {domain}: {e}")
            
            return domain, None
        
        # Run checks in parallel batches (5 at a time to respect rate limits)
        results = {}
        batch_size = 5
        
        for i in range(0, len(domains), batch_size):
            batch = domains[i:i + batch_size]
            
            # Run batch in parallel
            tasks = [check_single(domain) for domain in batch]
            batch_results = await asyncio.gather(*tasks)
            
            # Collect results
            for domain, data in batch_results:
                if data:
                    results[domain] = data
            
            # Rate limit between batches
            if i + batch_size < len(domains):
                await asyncio.sleep(1)  # 1 second between batches
        
        return results
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()