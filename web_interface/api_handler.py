import sys
import os
from pathlib import Path
import json
import re
import asyncio
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from refined_domain_hunter import (
    DomainConfig, UnifiedAIManager, 
    DocumentProcessor, DomainChecker,
    UnifiedDomainScorer, DomainGenerator
)

class DomainSuggestionAPI:
    def __init__(self):
        # Configure with API + DNS ONLY - NO MARKETPLACE CHECKING
        self.config = DomainConfig()
        self.config.whoapi_key = os.getenv('WHOAPI_KEY', '')
        self.config.whoisxml_api_key = os.getenv('WHOISXML_API_KEY', '')
        self.config.use_stealth_mode = False  # FORCE DISABLE stealth for speed
        self.config.check_strategy = 'api_only'  # API + DNS only, no marketplace
        
        self.ai = UnifiedAIManager(self.config)
        self.doc_processor = DocumentProcessor(self.ai)
        
        # Create checker with ALL marketplace checking DISABLED
        self.checker = DomainChecker(self.config, self.ai)
        # Force disable ALL marketplace and stealth modes
        if hasattr(self.checker, 'enable_stealth'):
            self.checker.enable_stealth = False
            self.checker.stealth_budget = 0
        if hasattr(self.checker, 'enable_marketplace'):
            self.checker.enable_marketplace = False
        
        self.scorer = UnifiedDomainScorer(self.ai)
        self.generator = DomainGenerator(self.ai, self.config)
        self.gemini_analysis = None
        self.merged_file = parent_dir / "merged_output.txt"

    async def generate_suggestions(self, user_idea: str, max_retries: int = 3):
        """Generate domain suggestions based on user's business idea"""
        
        # Load Gemini analysis for context
        if not self.gemini_analysis:
            print("\n=== LOADING RESEARCH CONTEXT ===")
            cache_file = Path('gemini_cache.txt')
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.gemini_analysis = f.read()
                print(f"Loaded cached analysis: {len(self.gemini_analysis)} chars")
            else:
                self.gemini_analysis = self._get_fallback_context()
        
        all_available = []
        all_attempts = []
        retry_count = 0
        
        while len(all_available) < 5 and retry_count < max_retries:
            retry_count += 1
            print(f"\n=== ATTEMPT {retry_count}/{max_retries} ===")
            
            # Generate domains based on the specific business idea
            domains = await self._generate_idea_based_domains(user_idea, all_attempts)
            if not domains:
                continue
            
            all_attempts.extend(domains)
            
            # Use PURE DNS checking - NO API, NO MARKETPLACE
            print(f"\n=== CHECKING {len(domains)} DOMAINS WITH PURE DNS-ONLY (NO MARKETPLACE) ===")
            
            # Use the parallel DNS check method directly to skip all marketplace logic
            availability_data = await self.checker._parallel_dns_check(domains)
            
            # Score available domains
            available_domains = [
                d for d, data in availability_data.items() 
                if data.get('available', False)
            ]
            
            if available_domains:
                scores = await self.scorer.score_domains(available_domains, availability_data)
                
                for domain in available_domains:
                    all_available.append({
                        'domain': domain,
                        'available': True,
                        'score': scores.get(domain, 5.0),
                        'price': availability_data[domain].get('aftermarket_price', '$60/year'),
                        'source': availability_data[domain].get('method', 'hybrid'),
                        'check_method': availability_data[domain].get('method', '')
                    })
                    print(f"✓ AVAILABLE: {domain} via {availability_data[domain].get('method')}")
            
            print(f"\nProgress: Found {len(all_available)} available domains")
            
            if len(all_available) >= 3:
                break
        
        # Sort by score
        all_available.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'suggestions': all_available[:10],
            'total_checked': len(all_attempts),
            'retries': retry_count,
            'check_methods': 'DNS + WhoAPI' if self.config.whoapi_key else 'DNS only'
        }
    
    async def _generate_idea_based_domains(self, user_idea: str, previous_attempts: List[str]) -> List[str]:
        """Generate domains specifically for the business idea"""
        
        # Analyze the business idea to extract key themes
        analysis_prompt = f"""Analyze this business idea and extract key themes:
{user_idea}

List 5 key terms/concepts that define this business:"""
        
        key_terms = await self.ai.generate_async(analysis_prompt, 'claude', max_tokens=200, temperature=0.5)
        
        # Generate domains based on the specific idea
        prompt = f"""Generate creative .ai domain names for this SPECIFIC business idea:

BUSINESS IDEA: {user_idea}

KEY THEMES: {key_terms}

Research insights: {self.gemini_analysis[:2000] if self.gemini_analysis else self._get_fallback_context()}

Already tried: {', '.join(previous_attempts[-20:]) if previous_attempts else 'None'}

IMPORTANT RULES:
1. Domains MUST be directly related to: {user_idea}
2. Use terms from the business description
3. Be creative but relevant
4. Make them memorable and brandable
5. Mix these patterns:
   - Industry-specific terms from the idea
   - Action words related to the business
   - Benefit-focused names
   - Creative combinations

Generate exactly 15 unique .ai domains that would be PERFECT for this specific business.
Return ONLY JSON array: ["domain1.ai", "domain2.ai", ...]"""
        
        response = await self.ai.generate_async(prompt, 'claude', max_tokens=1000, temperature=0.7)
        
        # Parse domains from response
        domains = []
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                domains = json.loads(json_match.group())
            except json.JSONDecodeError:
                print("Failed to parse JSON, trying line-by-line extraction")
                # Fallback: extract domains line by line
                for line in response.split('\n'):
                    if '.ai' in line:
                        domain = re.search(r'[\w-]+\.ai', line)
                        if domain:
                            domains.append(domain.group())
        
        # Clean and validate domains
        clean_domains = []
        for d in domains:
            d = str(d).lower().strip()
            # Remove quotes if present
            d = d.replace('"', '').replace("'", '')
            # Ensure .ai extension
            if not d.endswith('.ai'):
                if '.' in d:
                    d = d.split('.')[0] + '.ai'
                else:
                    d += '.ai'
            # Validate domain format
            if re.match(r'^[a-z0-9-]+\.ai$', d):
                clean_domains.append(d)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_domains = []
        for d in clean_domains:
            if d not in seen and d not in previous_attempts:
                seen.add(d)
                unique_domains.append(d)
        
        print(f"Generated {len(unique_domains)} domains for idea: {user_idea[:50]}...")
        
        return unique_domains[:15]
    
    def _get_fallback_context(self) -> str:
        """Fallback research context if no cached analysis"""
        return """Premium .ai domains characteristics:
        - Short, memorable, brandable names
        - Industry-specific but not too narrow
        - Action-oriented or benefit-focused
        - Easy to spell and pronounce
        - Avoid hyphens and numbers
        - Two-word combinations often work well
        - Tech-forward but accessible"""