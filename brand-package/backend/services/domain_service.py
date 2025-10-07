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
        description: str,  # Not used but required by base class
        user_id: str,
        domains: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Check domain availability (follows base class pattern)
        
        Args:
            description: Not used
            user_id: User ID
            domains: List of domains to check
            
        Returns:
            Availability results
        """
        return await self.check_domains(domains, user_id)
    
    async def check_domains(
        self,
        domains: List[str],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Check availability of multiple domains
        
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
        
        # First, try DNS checking (fast and free)
        dns_results = await self._parallel_dns_check(clean_domains)
        results.update(dns_results)
        
        # For domains that appear available, verify with API if configured
        if self.domain_apis:
            available_domains = [
                d for d, data in dns_results.items()
                if data.get('available', False)
            ]
            
            if available_domains:
                api_results = await self._check_with_apis(available_domains)
                
                # Merge results
                for domain, api_data in api_results.items():
                    if domain in results:
                        results[domain].update(api_data)
        
        # Format final results
        formatted_results = []
        for domain, data in results.items():
            formatted_results.append({
                "domain": domain,
                "available": data.get('available', False),
                "status": data.get('status', 'unknown'),
                "price": data.get('price', self._get_default_price(domain)),
                "registrar": data.get('registrar'),
                "registrar_link": self._get_registrar_link(domain),
                "checked_at": data.get('checked_at', datetime.now().isoformat()),
                "method": data.get('method', 'dns')
            })
        
        # Save generation
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
        generation_id: str,
        feedback: str,
        user_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Regenerate not applicable for domain checking
        Just re-check the same domains
        """
        # Get original generation
        original = await self.get_generation(generation_id, user_id)
        domains = original['input_data'].get('domains', [])
        
        # Re-check
        return await self.check_domains(domains, user_id)
    
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
                    'available': False,
                    'status': 'registered',
                    'method': 'dns',
                    'checked_at': datetime.now().isoformat()
                }
            except:
                # No DNS record found - might be available
                return domain, {
                    'available': True,
                    'status': 'possibly_available',
                    'method': 'dns',
                    'checked_at': datetime.now().isoformat()
                }
        
        # Run all checks in parallel
        tasks = [check_single(domain) for domain in domains]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    async def _check_with_apis(self, domains: List[str]) -> Dict[str, Dict]:
        """Check domains using configured APIs"""
        
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
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()