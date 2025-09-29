#!/usr/bin/env python3
"""
AI Domain Hunter - Refined & Streamlined Version
Reduces redundancy, improves organization, and maintains all functionality
"""

import asyncio
import aiohttp
import json
import os
import re
import time
import logging
import dns.resolver
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create organized directory structure
Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("debug").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Configure proper logging to logs folder
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/domain_hunt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Component loggers
logger_ai = logging.getLogger('AI')
logger_check = logging.getLogger('CHECK')
logger_score = logging.getLogger('SCORE')

@dataclass
class DomainConfig:
    """Centralized configuration"""
    openrouter_key: str = field(default_factory=lambda: os.getenv('OPENROUTER_API_KEY'))
    models: Dict[str, str] = field(default_factory=lambda: {
        'claude': os.getenv('CLAUDE_MODEL', 'anthropic/claude-3-5-sonnet-20241022'),
        'gpt': os.getenv('GPT_MODEL', 'openai/gpt-4o'),
        'gemini': os.getenv('GEMINI_MODEL', 'google/gemini-exp-1121')
    })
    max_domains: int = 500
    batch_size: int = 50
    check_strategy: str = 'smart'  # 'dns', 'smart', 'robust'
    generation_mode: str = 'hybrid'  # 'basic', 'hybrid', 'emergent'
    excluded_patterns: List[str] = field(default_factory=lambda: [
        'test', 'demo', 'temp', 'example', 'sample'
    ])

class PerformanceTracker:
    """Track performance metrics across operations"""
    
    def __init__(self):
        self.timings = {}
        self.start_times = {}
        self.counters = {}
    
    def start(self, operation: str):
        self.start_times[operation] = time.time()
        logger.debug(f"⏱️ Started: {operation}")
    
    def end(self, operation: str) -> float:
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.timings[operation] = duration
            logger.info(f"⏱️ Completed: {operation} - {duration:.2f}s")
            del self.start_times[operation]
            return duration
        return 0.0
    
    def count(self, item: str):
        self.counters[item] = self.counters.get(item, 0) + 1
    
    def report(self) -> Dict:
        """Generate performance summary"""
        total_time = sum(self.timings.values())
        return {
            'total_time': total_time,
            'operations': self.timings,
            'counters': self.counters,
            'summary': f"Total execution: {total_time:.2f}s"
        }

class UnifiedAIManager:
    """Single AI manager for all operations with error handling and retries"""
    
    def __init__(self, config: DomainConfig):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=config.openrouter_key
            )
            self.models = config.models
            self.tracker = PerformanceTracker()
            logger_ai.info("✓ AI Manager initialized successfully")
        except ImportError:
            logger_ai.error("OpenAI library not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger_ai.error(f"Failed to initialize AI Manager: {e}")
            raise
    
    def generate(self, prompt: str, model: str = 'gemini', 
                max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Unified generation method with retry logic"""
        operation = f"generate_{model}"
        self.tracker.start(operation)
        
        try:
            model_id = self.models.get(model, self.models['gemini'])
            logger_ai.debug(f"Generating with {model_id}")
            
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            self.tracker.end(operation)
            self.tracker.count(f"api_calls_{model}")
            
            logger_ai.debug(f"✓ Generated {len(content)} chars with {model}")
            return content
            
        except Exception as e:
            self.tracker.end(operation)
            logger_ai.error(f"✗ Generation failed with {model}: {e}")
            return ""
    
    def generate_parallel(self, prompt: str, models: List[str] = None) -> Dict[str, str]:
        """Generate from multiple models in parallel"""
        models = models or ['claude', 'gpt']
        results = {}
        
        for model in models:
            result = self.generate(prompt, model)
            if result:
                results[model] = result
        
        return results

class DomainChecker:
    """Unified domain availability checker with smart caching"""
    
    def __init__(self, config: DomainConfig):
        self.config = config
        self.searched_domains = self._load_previous_domains()
        self.tracker = PerformanceTracker()
        self.marketplaces = {
            'afternic': 'https://www.afternic.com/search?q={}',
            'sedo': 'https://sedo.com/search/?keyword={}',
            'dan': 'https://dan.com/search?q={}',
            'godaddy': 'https://auctions.godaddy.com/trpSearchResults.aspx?keyword={}'
        }
        logger_check.info(f"✓ Domain checker initialized with {len(self.searched_domains)} cached domains")
    
    def _load_previous_domains(self) -> Set[str]:
        """Load previously searched domains from all results"""
        domains = set()
        search_paths = [Path("results"), Path("data"), Path(".")]
        
        for path in search_paths:
            if not path.exists():
                continue
                
            # Load from CSV files
            for csv_file in path.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    domain_cols = ['domain', 'name', 'domain_name']
                    for col in domain_cols:
                        if col in df.columns:
                            domains.update(df[col].dropna().astype(str))
                            break
                except Exception as e:
                    logger_check.debug(f"Could not read {csv_file}: {e}")
            
            # Load from text files
            for txt_file in path.glob("*domains*.txt"):
                try:
                    with open(txt_file, 'r') as f:
                        for line in f:
                            domain = line.strip()
                            if domain and '.' in domain:
                                domains.add(domain)
                except Exception as e:
                    logger_check.debug(f"Could not read {txt_file}: {e}")
        
        logger_check.info(f"Loaded {len(domains)} previously searched domains")
        return domains
    
    async def check_availability(self, domains: List[str], strategy: str = 'smart') -> Dict[str, Dict]:
        """Unified availability checking with smart filtering"""
        self.tracker.start("availability_check")
        
        # Clean and filter domains
        clean_domains = []
        for domain in domains:
            clean_domain = domain.lower().strip()
            if not clean_domain.endswith('.ai'):
                clean_domain += '.ai'
            
            # Skip if already checked or invalid
            if (clean_domain not in self.searched_domains and 
                self._is_valid_domain(clean_domain)):
                clean_domains.append(clean_domain)
        
        logger_check.info(f"Checking {len(clean_domains)} new domains "
                         f"(filtered {len(domains) - len(clean_domains)} duplicates/invalid)")
        
        results = {}
        if strategy == 'dns':
            results = await self._dns_check_batch(clean_domains)
        elif strategy == 'smart':
            results = await self._smart_check(clean_domains)
        else:  # robust
            results = await self._robust_check(clean_domains)
        
        # Update cache
        self.searched_domains.update(clean_domains)
        self._save_search_cache()
        
        self.tracker.end("availability_check")
        return results
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Validate domain format and exclude unwanted patterns"""
        if not domain or len(domain) < 5:  # minimum: x.ai
            return False
        
        # Remove .ai for checking
        name_part = domain.replace('.ai', '')
        
        # Check for excluded patterns
        for pattern in self.config.excluded_patterns:
            if pattern in name_part.lower():
                return False
        
        # Basic format validation
        if not re.match(r'^[a-z0-9-]+\.ai$', domain):
            return False
        
        return True
    
    async def _dns_check_batch(self, domains: List[str]) -> Dict[str, Dict]:
        """Fast DNS-based checking"""
        results = {}
        
        async def check_single_dns(domain: str) -> bool:
            try:
                dns.resolver.resolve(domain, 'A')
                return False  # Domain exists (not available)
            except:
                return True   # No DNS record (potentially available)
        
        # Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(domains), batch_size):
            batch = domains[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[check_single_dns(domain) for domain in batch],
                return_exceptions=True
            )
            
            for domain, available in zip(batch, batch_results):
                if isinstance(available, bool):
                    results[domain] = {
                        'available': available,
                        'status': 'dns_available' if available else 'dns_taken',
                        'checked_at': datetime.now().isoformat(),
                        'method': 'dns'
                    }
                else:
                    results[domain] = {
                        'available': None,
                        'status': 'dns_error',
                        'error': str(available),
                        'checked_at': datetime.now().isoformat(),
                        'method': 'dns'
                    }
            
            # Rate limiting
            if i + batch_size < len(domains):
                await asyncio.sleep(1)
        
        return results
    
    async def _smart_check(self, domains: List[str]) -> Dict[str, Dict]:
        """DNS check + basic marketplace scan for promising domains"""
        # Start with DNS check
        results = await self._dns_check_batch(domains)
        
        # Get potentially available domains for marketplace check
        available_domains = [
            domain for domain, data in results.items() 
            if data.get('available', False)
        ]
        
        if available_domains:
            logger_check.info(f"Running marketplace check on {len(available_domains)} DNS-available domains")
            marketplace_results = await self._check_marketplaces_batch(available_domains[:10])  # Limit to top 10
            
            # Merge results
            for domain, marketplace_data in marketplace_results.items():
                if domain in results:
                    results[domain].update(marketplace_data)
        
        return results
    
    async def whois_check_batch(self, domains: List[str], max_batch: int = 10) -> Dict[str, Dict]:
        """WHOIS checking for accurate availability verification"""
        try:
            import whois
        except ImportError:
            logger_check.warning("python-whois not installed. Run: pip install python-whois")
            return {d: {'status': 'whois_unavailable'} for d in domains}
        
        results = {}
        for domain in domains[:max_batch]:  # Limit to prevent rate limiting
            try:
                logger_check.debug(f"WHOIS checking {domain}")
                w = whois.whois(domain)
                
                # Check if domain is registered
                is_registered = bool(w.domain_name) if hasattr(w, 'domain_name') else False
                
                results[domain] = {
                    'available': not is_registered,
                    'status': 'whois_available' if not is_registered else 'whois_registered',
                    'whois_data': {
                        'registrar': getattr(w, 'registrar', None),
                        'creation_date': str(getattr(w, 'creation_date', None)),
                        'expiration_date': str(getattr(w, 'expiration_date', None)),
                        'name_servers': getattr(w, 'name_servers', [])
                    } if is_registered else None,
                    'method': 'whois'
                }
                logger_check.debug(f"WHOIS {domain}: {'registered' if is_registered else 'available'}")
                
            except Exception as e:
                logger_check.debug(f"WHOIS check failed for {domain}: {e}")
                results[domain] = {
                    'available': None, 
                    'status': 'whois_error',
                    'error': str(e),
                    'method': 'whois'
                }
            
            await asyncio.sleep(2)  # Rate limiting to respect WHOIS servers
        
        logger_check.info(f"WHOIS checked {len(results)} domains")
        return results
    
    async def _robust_check(self, domains: List[str]) -> Dict[str, Dict]:
        """Full checking: DNS + WHOIS + Marketplace (Most Accurate)"""
        logger_check.info(f"Starting robust check (DNS + WHOIS + Marketplace) for {len(domains)} domains")
        
        # Pass 1: DNS check (fast initial filter)
        logger_check.info("Pass 1: DNS checking all domains")
        dns_results = await self._dns_check_batch(domains)
        
        # Get DNS-available domains for WHOIS verification
        dns_available = [
            domain for domain, data in dns_results.items() 
            if data.get('available', False)
        ]
        
        logger_check.info(f"Found {len(dns_available)} DNS-available domains for WHOIS verification")
        
        # Pass 2: WHOIS verification on DNS-available domains
        if dns_available:
            logger_check.info(f"Pass 2: WHOIS checking top {min(20, len(dns_available))} DNS-available domains")
            whois_results = await self.whois_check_batch(dns_available[:20])  # Limit for performance
            
            # Merge WHOIS results into DNS results
            for domain, whois_data in whois_results.items():
                if domain in dns_results:
                    # Update availability based on WHOIS (more accurate than DNS)
                    if whois_data.get('available') is not None:
                        dns_results[domain]['available'] = whois_data['available']
                        dns_results[domain]['status'] = whois_data['status']
                    
                    # Add WHOIS metadata
                    dns_results[domain]['whois_data'] = whois_data.get('whois_data')
                    dns_results[domain]['whois_checked'] = True
                    
                    # Update method to show combined checking
                    dns_results[domain]['method'] = 'dns+whois'
        
        # Pass 3: Marketplace check on truly available domains
        truly_available = [
            domain for domain, data in dns_results.items()
            if data.get('available', False) and data.get('whois_checked', False)
        ]
        
        if truly_available:
            logger_check.info(f"Pass 3: Marketplace checking {min(5, len(truly_available))} verified available domains")
            marketplace_results = await self._check_marketplaces_batch(truly_available[:5])
            
            # Merge marketplace results
            for domain, marketplace_data in marketplace_results.items():
                if domain in dns_results:
                    dns_results[domain].update(marketplace_data)
                    dns_results[domain]['method'] = 'dns+whois+marketplace'
        
        logger_check.info(f"Robust check complete: {len(dns_results)} domains processed")
        return dns_results
    
    async def _check_marketplaces_batch(self, domains: List[str]) -> Dict[str, Dict]:
        """Check domains across marketplaces"""
        results = {}
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        ) as session:
            
            for domain in domains[:5]:  # Limit to prevent rate limiting
                marketplace_data = await self._check_single_marketplace(session, domain)
                results[domain] = marketplace_data
                await asyncio.sleep(2)  # Rate limiting
        
        return results
    
    async def _check_single_marketplace(self, session: aiohttp.ClientSession, domain: str) -> Dict:
        """Check single domain across marketplaces"""
        marketplace_data = {
            'marketplace_checked': True,
            'listings': []
        }
        
        for marketplace, url_template in self.marketplaces.items():
            try:
                url = url_template.format(domain.replace('.ai', ''))
                async with session.get(url) as response:
                    if response.status == 200:
                        text = await response.text()
                        # Basic listing detection
                        if any(indicator in text.lower() for indicator in 
                               ['for sale', 'buy now', 'make offer', 'listed']):
                            marketplace_data['listings'].append({
                                'marketplace': marketplace,
                                'url': url,
                                'found_indicators': True
                            })
            except Exception as e:
                logger_check.debug(f"Marketplace check failed for {domain} on {marketplace}: {e}")
        
        return marketplace_data
    
    def _save_search_cache(self):
        """Save updated search cache"""
        cache_file = Path("data") / f"searched_domains_cache_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(cache_file, 'w') as f:
            for domain in sorted(self.searched_domains):
                f.write(f"{domain}\n")
        logger_check.debug(f"Saved {len(self.searched_domains)} domains to cache")

class UnifiedDomainScorer:
    """Simplified, unified domain scoring system"""
    
    def __init__(self, ai_manager: UnifiedAIManager):
        self.ai = ai_manager
        self.tracker = PerformanceTracker()
        
        # Scoring patterns discovered from research
        self.scoring_patterns = """
Based on comprehensive domain valuation research, score domains using these key patterns:

1. Cognitive Fluency vs. Desirable Difficulty
   - High fluency: Short, pronounceable, memorable (trust building)
   - Strategic disruption: Unique enough to be memorable

2. Cultural Arbitrage
   - .ai TLD transformation from Anguilla to AI industry standard
   - Cross-language appeal and meaning

3. Strategic Inevitability
   - Domains that major players will NEED to own
   - Category-defining potential

4. Temporal Arbitrage
   - Early positioning in emerging trends
   - Future value based on market evolution

Score 0-10 based on genuine investment potential.
"""
    
    async def score_domains(self, domains: List[str], market_data: Dict = None) -> Dict[str, float]:
        """Score multiple domains efficiently"""
        self.tracker.start("domain_scoring")
        
        scores = {}
        market_data = market_data or {}
        
        # Process in batches to avoid overwhelming APIs
        batch_size = 10
        for i in range(0, len(domains), batch_size):
            batch = domains[i:i + batch_size]
            batch_scores = await self._score_batch(batch, market_data)
            scores.update(batch_scores)
            
            # Rate limiting between batches
            if i + batch_size < len(domains):
                await asyncio.sleep(2)
        
        self.tracker.end("domain_scoring")
        return scores
    
    async def _score_batch(self, domains: List[str], market_data: Dict) -> Dict[str, float]:
        """Score a batch of domains"""
        scores = {}
        
        for domain in domains:
            try:
                # Get availability context
                availability_context = market_data.get(domain, {})
                
                prompt = f"""
{self.scoring_patterns}

Evaluate domain: {domain}

Availability context:
{json.dumps(availability_context, indent=2)}

Provide ONLY a score from 0-10 as a number, followed by brief reasoning.
Example: "7.5 - High cognitive fluency, strong AI alignment, moderate brandability"
"""
                
                # Use both Claude and GPT for scoring, average them
                claude_response = self.ai.generate(prompt, 'claude', max_tokens=200)
                gpt_response = self.ai.generate(prompt, 'gpt', max_tokens=200)
                
                claude_score = self._extract_score(claude_response)
                gpt_score = self._extract_score(gpt_response)
                
                # Average the scores
                if claude_score > 0 and gpt_score > 0:
                    final_score = (claude_score + gpt_score) / 2
                elif claude_score > 0:
                    final_score = claude_score
                elif gpt_score > 0:
                    final_score = gpt_score
                else:
                    final_score = 3.0  # Default fallback
                
                scores[domain] = final_score
                logger_score.debug(f"{domain}: {final_score:.2f} (Claude: {claude_score}, GPT: {gpt_score})")
                
            except Exception as e:
                logger_score.error(f"Scoring failed for {domain}: {e}")
                scores[domain] = 3.0  # Default score
        
        return scores
    
    async def dual_model_evaluation(self, domain: str, gemini_analysis: str) -> Dict:
        """
        Evaluate domain using both Claude and GPT with document insights
        This dual evaluation is PROVEN to work best
        """
        
        base_prompt = f"""
Based on the research analysis that identified high-value patterns:

{gemini_analysis}  # Key insights only

Evaluate domain: {domain}

Score 0-10 based on:
1. Pattern matches from the research
2. Real investment potential ($10K+ domains score 7+)
3. Strategic value and inevitability

Return: SCORE: X.X | REASONING: brief explanation
"""
        
        # Get evaluations from both models
        claude_eval = self.ai.generate(base_prompt, 'claude', max_tokens=200, temperature=0.3)
        gpt_eval = self.ai.generate(base_prompt, 'gpt', max_tokens=200, temperature=0.3)
        
        # Extract scores
        claude_score = self._extract_score(claude_eval)
        gpt_score = self._extract_score(gpt_eval)
        
        # Average for final score
        final_score = (claude_score + gpt_score) / 2 if claude_score and gpt_score else claude_score or gpt_score or 0.0
        
        return {
            'claude_score': claude_score,
            'gpt_score': gpt_score,
            'final_score': final_score,
            'claude_reasoning': claude_eval,
            'gpt_reasoning': gpt_eval
        }
    
    def _extract_score(self, response: str) -> float:
        """Extract numeric score from AI response"""
        if not response:
            return 0.0
        
        # Look for patterns like "7.5", "8/10", "Score: 6.2"
        patterns = [
            r'(\d+\.?\d*)/10',
            r'(\d+\.?\d*)\s*-',
            r'Score:?\s*(\d+\.?\d*)',
            r'^(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.strip())
            if match:
                try:
                    score = float(match.group(1))
                    return max(0.0, min(10.0, score))  # Clamp to 0-10
                except ValueError:
                    continue
        
        return 0.0

class DomainGenerator:
    """Unified domain generation with multiple strategies"""
    
    def __init__(self, ai_manager: UnifiedAIManager, config: DomainConfig):
        self.ai = ai_manager
        self.config = config
        self.tracker = PerformanceTracker()
    
    async def generate_domains(self, count: int, mode: str = 'hybrid', 
                              exclude_domains: Set[str] = None) -> List[str]:
        """Generate domains using specified strategy"""
        self.tracker.start(f"generation_{mode}")
        
        exclude_domains = exclude_domains or set()
        
        if mode == 'emergent':
            domains = await self._generate_emergent(count, exclude_domains)
        elif mode == 'hybrid':
            domains = await self._generate_hybrid(count, exclude_domains)
        else:  # basic
            domains = await self._generate_basic(count, exclude_domains)
        
        # Clean and deduplicate
        clean_domains = self._clean_domains(domains, exclude_domains)
        final_domains = list(set(clean_domains))[:count]
        
        self.tracker.end(f"generation_{mode}")
        logger.info(f"Generated {len(final_domains)} unique domains using {mode} mode")
        
        return final_domains
    
    async def _generate_hybrid(self, count: int, exclude_domains: Set[str]) -> List[str]:
        """Hybrid generation: proven patterns + current trends + creative"""
        
        # Load research content if available
        research_context = self._load_research_context()
        
        exclusion_list = list(exclude_domains)[:50]  # Limit for prompt size
        
        prompt = f"""
Generate {count} premium .ai domain names using this hybrid approach:

40% PROVEN PATTERNS (high-value, time-tested concepts):
- Core AI terms: neural, brain, mind, agent, learn
- Business terms: hub, pro, edge, lab, core
- Action terms: build, make, grow, scale, optimize

40% CURRENT TRENDS (research what's hot now):
- Latest AI breakthroughs and terminology
- Emerging technology trends
- New AI applications and use cases

20% CREATIVE COMBINATIONS:
- Unexpected but logical pairings
- Cross-industry applications
- Future-looking concepts

Research context (if available):
{research_context[:2000] if research_context else "Use your knowledge of current AI trends"}

AVOID these already-searched domains:
{exclusion_list[:20]}

Return as clean JSON array: ["domain1", "domain2", ...]
Each domain should be brandable, memorable, and investment-worthy.
"""
        
        # Generate from both Claude and GPT
        responses = self.ai.generate_parallel(prompt, ['claude', 'gpt'])
        
        all_domains = []
        for model, response in responses.items():
            domains = self._parse_domain_response(response)
            all_domains.extend(domains)
            logger.debug(f"{model} generated {len(domains)} domains")
        
        return all_domains
    
    async def generate_with_document_analysis(self, count: int, 
                                             gemini_analysis: str,
                                             exclude_domains: Set[str]) -> Dict[str, List[str]]:
        """
        Generate domains using Gemini's document analysis
        This is the PROVEN BEST method from original code
        """
        
        # Prepare exclusion list
        exclusion_text = ""
        if exclude_domains:
            sample = list(exclude_domains)[:50]
            exclusion_text = f"\nAVOID these already-checked domains:\n{', '.join(sample)}\n"
        
        # Claude generation prompt - focus on premium patterns
        claude_prompt = f"""
Based on this comprehensive analysis of domain research:

{gemini_analysis}  # Limit for Claude's context

{exclusion_text}

Generate {count//2} premium .ai domains based on patterns YOU discovered in the analysis.
Don't follow fixed rules - use the insights from the research.
Focus on domains that could sell for $10K+ based on the patterns found.

Return ONLY a JSON array: ["domain1", "domain2", ...]
"""
        
        # GPT generation prompt - focus on emerging opportunities  
        gpt_prompt = f"""
Based on this research analysis:

{gemini_analysis}  # Limit for GPT's context

{exclusion_text}

Generate {count//2} .ai domains using the unexpected patterns and opportunities discovered.
Focus on contrarian insights and emerging trends from the analysis.
Create domains that match the high-value patterns identified.

Return ONLY a JSON array: ["domain1", "domain2", ...]
"""
        
        # Generate from both models
        claude_response = self.ai.generate(claude_prompt, 'claude', max_tokens=2000)
        gpt_response = self.ai.generate(gpt_prompt, 'gpt', max_tokens=2000)
        
        # Parse responses
        claude_domains = self._parse_domain_response(claude_response)
        gpt_domains = self._parse_domain_response(gpt_response)
        
        logger.info(f"Claude generated {len(claude_domains)} domains from analysis")
        logger.info(f"GPT generated {len(gpt_domains)} domains from analysis")
        
        return {
            'claude_domains': claude_domains,
            'gpt_domains': gpt_domains,
            'all_domains': list(set(claude_domains + gpt_domains))
        }
    
    async def _generate_emergent(self, count: int, exclude_domains: Set[str]) -> List[str]:
        """Emergent generation: let AI discover its own patterns"""
        
        prompt = f"""
You are an AI domain discovery system. Don't follow prescribed patterns - discover your own.

Analyze the current AI landscape and identify emerging opportunities that others might miss.
Generate {count} .ai domains based on YOUR understanding of:
- What patterns create real value
- Where the AI industry is heading
- What domains would be strategically valuable

Avoid these already-searched domains: {list(exclude_domains)[:20]}

Don't just combine obvious keywords. Find sophisticated opportunities.
Return as JSON array: ["domain1", "domain2", ...]
"""
        
        # Use Gemini for emergent discovery
        response = self.ai.generate(prompt, 'gemini', max_tokens=3000)
        domains = self._parse_domain_response(response)
        
        logger.info(f"Emergent generation produced {len(domains)} unique concepts")
        return domains
    
    async def _generate_basic(self, count: int, exclude_domains: Set[str]) -> List[str]:
        """Basic generation: straightforward AI domain creation"""
        
        prompt = f"""
Generate {count} .ai domain names for businesses and applications.
Focus on clear, brandable names that would work for AI companies.

Avoid: {list(exclude_domains)[:20]}

Return as JSON array: ["domain1", "domain2", ...]
"""
        
        response = self.ai.generate(prompt, 'claude', max_tokens=2000)
        return self._parse_domain_response(response)
    
    def _parse_domain_response(self, response: str) -> List[str]:
        """Parse domains from AI response with multiple fallback methods"""
        domains = []
        
        if not response:
            return domains
        
        try:
            # Method 1: JSON parsing
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
                for item in items:
                    if isinstance(item, str):
                        clean_domain = self._normalize_domain(item)
                        if clean_domain:
                            domains.append(clean_domain)
        except json.JSONDecodeError:
            pass
        
        # Method 2: Line-by-line parsing
        if not domains:
            for line in response.split('\n'):
                # Remove numbering, bullets, quotes
                cleaned = re.sub(r'^\d+\.|^[-*•]\s*|^["\']\s*|["\']\s*$', '', line).strip()
                clean_domain = self._normalize_domain(cleaned)
                if clean_domain:
                    domains.append(clean_domain)
        
        return domains
    
    def _normalize_domain(self, domain_text: str) -> Optional[str]:
        """Normalize and validate domain text"""
        if not domain_text:
            return None
        
        # Clean the text
        domain = domain_text.lower().strip()
        domain = re.sub(r'[^\w.-]', '', domain)
        
        # Ensure .ai extension
        if not domain.endswith('.ai'):
            if domain.endswith('.'):
                domain = domain[:-1]
            domain += '.ai'
        
        # Extract just the domain name part
        domain_name = domain.replace('.ai', '')
        
        # Validate
        if (3 <= len(domain_name) <= 20 and 
            re.match(r'^[a-z0-9-]+$', domain_name) and
            not domain_name.startswith('-') and 
            not domain_name.endswith('-')):
            return domain
        
        return None
    
    def _clean_domains(self, domains: List[str], exclude_domains: Set[str]) -> List[str]:
        """Final cleaning and filtering of generated domains"""
        clean_domains = []
        
        for domain in domains:
            clean_domain = self._normalize_domain(domain)
            if (clean_domain and 
                clean_domain not in exclude_domains and
                clean_domain not in clean_domains):  # Deduplicate
                clean_domains.append(clean_domain)
        
        return clean_domains
    
    def _load_research_context(self) -> str:
        """Load research context if available"""
        research_files = [
            "merged_output.txt",
            "data/research.txt",
            "research_content.txt"
        ]
        
        for file_path in research_files:
            try:
                if Path(file_path).exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        logger.debug(f"Loaded research context from {file_path}")
                        return content[:5000]  # Limit size for prompt
            except Exception as e:
                logger.debug(f"Could not load {file_path}: {e}")
        
        return ""

class DocumentProcessor:
    """Process merged research document with Gemini's large context window"""
    
    def __init__(self, ai_manager: UnifiedAIManager):
        self.ai = ai_manager
        self.gemini_analysis = None
        self.analysis_timestamp = None
        
    async def process_merged_document(self, file_path: str = "merged_output.txt") -> Dict:
        """
        Load and analyze merged_output.txt with Gemini
        This is the PROVEN BEST STRATEGY from original code
        """
        # Step 1: Load the merged file
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                merged_content = f.read()
            logger.info(f"Loaded merged document: {len(merged_content):,} characters")
        except FileNotFoundError:
            logger.error(f"merged_output.txt not found")
            return {}
        
        # Step 2: Send ENTIRE document to Gemini (1M token capacity)
        analysis_prompt = f"""
Analyze this comprehensive research document ({len(merged_content):,} characters) about domains.

RESEARCH CONTENT:
{merged_content}

Your task:
1. Read EVERYTHING - all 32 documents merged here
2. Find ALL patterns that influence domain value
3. Discover unexpected connections and opportunities
4. Identify what REALLY drives value based on the data
5. Find contrarian insights others would miss

Don't follow templates. Let patterns emerge from the data.
What patterns predict $10K+ domain value?
What emerging trends create opportunity?
What combinations are powerful but non-obvious?

Provide comprehensive analysis with:
- Top 20 most valuable patterns discovered
- Unexpected insights and connections
- Contrarian opportunities
- Specific domain generation strategies based on your findings
"""
        
        logger.info("Sending full document to Gemini for analysis...")
        self.gemini_analysis = self.ai.generate(
            analysis_prompt, 
            model='gemini',
            max_tokens=16000,  # Allow long comprehensive response
            temperature=0.4
        )
        
        self.analysis_timestamp = datetime.now()
        logger.info(f"Gemini analysis complete: {len(self.gemini_analysis):,} characters")
        
        # Save analysis for debugging
        debug_file = Path("debug") / f"gemini_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(debug_file, 'w') as f:
            f.write(self.gemini_analysis)
        
        return {
            'analysis': self.gemini_analysis,
            'timestamp': self.analysis_timestamp,
            'document_size': len(merged_content),
            'analysis_size': len(self.gemini_analysis)
        }
    
    def get_analysis(self) -> str:
        """Get cached Gemini analysis"""
        return self.gemini_analysis or ""

class RefinedDomainHunter:
    """Main unified domain hunter class"""
    
    def __init__(self, config: DomainConfig = None):
        self.config = config or DomainConfig()
        
        # Validate configuration
        if not self.config.openrouter_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY environment variable.")
        
        # Initialize components
        self.ai = UnifiedAIManager(self.config)
        self.checker = DomainChecker(self.config)
        self.scorer = UnifiedDomainScorer(self.ai)
        self.generator = DomainGenerator(self.ai, self.config)
        self.document_processor = DocumentProcessor(self.ai)
        self.tracker = PerformanceTracker()
        
        logger.info("✓ Refined Domain Hunter initialized successfully")
    
    async def hunt(self, 
                   count: int = 500,
                   mode: str = 'hybrid',
                   check_strategy: str = 'smart',
                   score_top_n: int = 100) -> pd.DataFrame:
        """
        Single unified hunting method that replaces all previous hunt methods
        
        Args:
            count: Number of domains to generate
            mode: Generation mode ('basic', 'hybrid', 'emergent')
            check_strategy: Checking strategy ('dns', 'smart', 'robust')
            score_top_n: How many top domains to score (expensive operation)
        """
        
        self.tracker.start("total_hunt")
        logger.info("="*60)
        logger.info(f"🚀 STARTING DOMAIN HUNT")
        logger.info(f"Count: {count}, Mode: {mode}, Strategy: {check_strategy}")
        logger.info("="*60)
        
        try:
            # Step 1: Generate domains
            logger.info("Phase 1: Domain Generation")
            domains = await self.generator.generate_domains(
                count=count, 
                mode=mode, 
                exclude_domains=self.checker.searched_domains
            )
            logger.info(f"✓ Generated {len(domains)} domains")
            
            # Step 2: Check availability
            logger.info("Phase 2: Availability Checking")
            availability_data = await self.checker.check_availability(domains, check_strategy)
            logger.info(f"✓ Checked availability for {len(availability_data)} domains")
            
            # Step 3: Score top domains
            logger.info("Phase 3: Domain Scoring")
            # Sort by availability preference (available first)
            available_domains = [
                d for d, data in availability_data.items() 
                if data.get('available', False)
            ]
            other_domains = [
                d for d, data in availability_data.items() 
                if not data.get('available', False)
            ]
            
            # Score top available domains + some others
            domains_to_score = (available_domains + other_domains)[:score_top_n]
            scores = await self.scorer.score_domains(domains_to_score, availability_data)
            logger.info(f"✓ Scored {len(scores)} domains")
            
            # Step 4: Compile results
            logger.info("Phase 4: Compiling Results")
            results_data = []
            
            for domain in domains:
                availability = availability_data.get(domain, {})
                score = scores.get(domain, 0.0)
                
                results_data.append({
                    'domain': domain,
                    'score': score,
                    'available': availability.get('available', False),
                    'status': availability.get('status', 'unknown'),
                    'checked_at': availability.get('checked_at', ''),
                    'method': availability.get('method', ''),
                    'marketplace_listings': len(availability.get('listings', [])),
                    'generation_mode': mode,
                    'check_strategy': check_strategy
                })
            
            # Create DataFrame and sort
            df = pd.DataFrame(results_data)
            df = df.sort_values(['available', 'score'], ascending=[False, False])
            
            # Step 5: Save results
            self._save_results(df, mode, check_strategy)
            
            # Step 6: Generate summary
            summary = self._generate_summary(df)
            logger.info("="*60)
            logger.info("✅ HUNT COMPLETE")
            logger.info(summary)
            logger.info("="*60)
            
            self.tracker.end("total_hunt")
            
            return df
            
        except Exception as e:
            logger.error(f"Hunt failed: {e}")
            self.tracker.end("total_hunt")
            raise
    
    async def hunt_with_document_analysis(self, 
                                        count: int = 500,
                                        check_strategy: str = 'smart',
                                        score_top_n: int = 100,
                                        document_path: str = "merged_output.txt") -> pd.DataFrame:
        """
        Enhanced hunt method that uses Gemini document analysis
        This is the PROVEN BEST approach from the original system
        """
        
        self.tracker.start("enhanced_hunt")
        logger.info("="*60)
        logger.info(f"🚀 STARTING ENHANCED DOCUMENT-BASED HUNT")
        logger.info(f"Count: {count}, Strategy: {check_strategy}, Document: {document_path}")
        logger.info("="*60)
        
        try:
            # Step 1: Process merged document with Gemini
            logger.info("Phase 1: Document Analysis with Gemini")
            doc_analysis = await self.document_processor.process_merged_document(document_path)
            
            if not doc_analysis or not doc_analysis.get('analysis'):
                logger.warning("No document analysis available, falling back to standard hunt")
                return await self.hunt(count, 'hybrid', check_strategy, score_top_n)
            
            gemini_analysis = doc_analysis['analysis']
            logger.info(f"✓ Gemini analyzed {doc_analysis['document_size']:,} chars → {doc_analysis['analysis_size']:,} chars")
            
            # Step 2: Generate domains using document insights
            logger.info("Phase 2: Document-Informed Domain Generation")
            generation_result = await self.generator.generate_with_document_analysis(
                count=count,
                gemini_analysis=gemini_analysis,
                exclude_domains=self.checker.searched_domains
            )
            
            all_domains = generation_result['all_domains']
            claude_domains = generation_result['claude_domains']
            gpt_domains = generation_result['gpt_domains']
            
            logger.info(f"✓ Generated {len(all_domains)} domains (Claude: {len(claude_domains)}, GPT: {len(gpt_domains)})")
            
            # Step 3: Check availability
            logger.info("Phase 3: Availability Checking")
            availability_data = await self.checker.check_availability(all_domains, check_strategy)
            logger.info(f"✓ Checked availability for {len(availability_data)} domains")
            
            # Step 4: Enhanced scoring with document insights
            logger.info("Phase 4: Enhanced Domain Scoring")
            # Sort by availability preference (available first)
            available_domains = [
                d for d, data in availability_data.items() 
                if data.get('available', False)
            ]
            other_domains = [
                d for d, data in availability_data.items() 
                if not data.get('available', False)
            ]
            
            # Score top available domains + some others using dual evaluation
            domains_to_score = (available_domains + other_domains)[:score_top_n]
            enhanced_scores = {}
            
            for domain in domains_to_score[:20]:  # Limit dual evaluation to top 20 for cost control
                dual_eval = await self.scorer.dual_model_evaluation(domain, gemini_analysis)
                enhanced_scores[domain] = {
                    'score': dual_eval['final_score'],
                    'claude_score': dual_eval['claude_score'],
                    'gpt_score': dual_eval['gpt_score'],
                    'claude_reasoning': dual_eval['claude_reasoning'],
                    'gpt_reasoning': dual_eval['gpt_reasoning']
                }
                await asyncio.sleep(1)  # Rate limiting for dual evaluation
            
            # Use standard scoring for remaining domains
            remaining_domains = domains_to_score[20:]
            if remaining_domains:
                standard_scores = await self.scorer.score_domains(remaining_domains, availability_data)
                for domain, score in standard_scores.items():
                    enhanced_scores[domain] = {
                        'score': score,
                        'claude_score': score,
                        'gpt_score': score,
                        'claude_reasoning': 'Standard evaluation',
                        'gpt_reasoning': 'Standard evaluation'
                    }
            
            logger.info(f"✓ Enhanced scoring complete for {len(enhanced_scores)} domains")
            
            # Step 5: Compile enhanced results
            logger.info("Phase 5: Compiling Enhanced Results")
            results_data = []
            
            for domain in all_domains:
                availability = availability_data.get(domain, {})
                score_data = enhanced_scores.get(domain, {'score': 0.0})
                
                # Determine source model
                source_model = 'both'
                if domain in claude_domains and domain not in gpt_domains:
                    source_model = 'claude'
                elif domain in gpt_domains and domain not in claude_domains:
                    source_model = 'gpt'
                
                results_data.append({
                    'domain': domain,
                    'score': score_data['score'],
                    'claude_score': score_data.get('claude_score', 0.0),
                    'gpt_score': score_data.get('gpt_score', 0.0),
                    'available': availability.get('available', False),
                    'status': availability.get('status', 'unknown'),
                    'checked_at': availability.get('checked_at', ''),
                    'method': availability.get('method', ''),
                    'marketplace_listings': len(availability.get('listings', [])),
                    'source_model': source_model,
                    'generation_mode': 'document_analysis',
                    'check_strategy': check_strategy,
                    'claude_reasoning': score_data.get('claude_reasoning', ''),
                    'gpt_reasoning': score_data.get('gpt_reasoning', '')
                })
            
            # Create DataFrame and sort
            df = pd.DataFrame(results_data)
            df = df.sort_values(['available', 'score'], ascending=[False, False])
            
            # Step 6: Save enhanced results
            self._save_enhanced_results(df, check_strategy, doc_analysis)
            
            # Step 7: Generate enhanced summary
            summary = self._generate_enhanced_summary(df, doc_analysis)
            logger.info("="*60)
            logger.info("✅ ENHANCED HUNT COMPLETE")
            logger.info(summary)
            logger.info("="*60)
            
            self.tracker.end("enhanced_hunt")
            
            return df
            
        except Exception as e:
            logger.error(f"Enhanced hunt failed: {e}")
            self.tracker.end("enhanced_hunt")
            raise
    
    def _save_enhanced_results(self, df: pd.DataFrame, strategy: str, doc_analysis: Dict):
        """Save enhanced results with document analysis metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Main enhanced results CSV
        results_file = Path("results") / f"enhanced_hunt_{strategy}_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        
        # Top available domains with detailed scoring
        available_df = df[df['available'] == True].head(20)
        if not available_df.empty:
            available_file = Path("results") / f"enhanced_available_{timestamp}.csv"
            available_df.to_csv(available_file, index=False)
        
        # Document analysis metadata
        analysis_meta = {
            'document_analysis': doc_analysis,
            'hunt_timestamp': timestamp,
            'total_domains': len(df),
            'available_count': len(df[df['available'] == True]),
            'avg_score': df['score'].mean(),
            'top_score': df['score'].max()
        }
        
        meta_file = Path("debug") / f"enhanced_analysis_meta_{timestamp}.json"
        with open(meta_file, 'w') as f:
            json.dump(analysis_meta, f, indent=2, default=str)
        
        # Performance report
        perf_report = self.tracker.report()
        perf_file = Path("debug") / f"enhanced_performance_{timestamp}.json"
        with open(perf_file, 'w') as f:
            json.dump(perf_report, f, indent=2)
        
        logger.info(f"Enhanced results saved to {results_file}")
    
    def _generate_enhanced_summary(self, df: pd.DataFrame, doc_analysis: Dict) -> str:
        """Generate enhanced execution summary"""
        total_domains = len(df)
        available_count = len(df[df['available'] == True])
        scored_count = len(df[df['score'] > 0])
        avg_score = df[df['score'] > 0]['score'].mean() if scored_count > 0 else 0
        top_score = df['score'].max() if scored_count > 0 else 0
        
        # Model breakdown
        claude_count = len(df[df['source_model'].isin(['claude', 'both'])])
        gpt_count = len(df[df['source_model'].isin(['gpt', 'both'])])
        
        summary = f"""
ENHANCED EXECUTION SUMMARY:
  📊 Total domains: {total_domains}
  ✅ Available: {available_count}
  🎯 Scored: {scored_count}
  📈 Average score: {avg_score:.2f}
  🏆 Top score: {top_score:.2f}
  
DOCUMENT ANALYSIS:
  📄 Document size: {doc_analysis.get('document_size', 0):,} chars
  🤖 Gemini analysis: {doc_analysis.get('analysis_size', 0):,} chars
  ⏰ Analysis time: {doc_analysis.get('timestamp', 'Unknown')}
  
MODEL BREAKDOWN:
  🟦 Claude domains: {claude_count}
  🟩 GPT domains: {gpt_count}
  
TOP 5 AVAILABLE DOMAINS:
"""
        
        top_available = df[df['available'] == True].head(5)
        for i, row in top_available.iterrows():
            summary += f"  {i+1}. {row['domain']} - Score: {row['score']:.2f} ({row['source_model']})\n"
        
        perf_report = self.tracker.report()
        summary += f"\n⏱️ Total time: {perf_report['total_time']:.2f}s"
        
        return summary
    
    def _save_results(self, df: pd.DataFrame, mode: str, strategy: str):
        """Save results to organized files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Main results CSV
        results_file = Path("results") / f"hunt_{mode}_{strategy}_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        
        # Top available domains
        available_df = df[df['available'] == True].head(20)
        if not available_df.empty:
            available_file = Path("results") / f"available_domains_{timestamp}.csv"
            available_df.to_csv(available_file, index=False)
        
        # Performance report
        perf_report = self.tracker.report()
        perf_file = Path("debug") / f"performance_{timestamp}.json"
        with open(perf_file, 'w') as f:
            json.dump(perf_report, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def _generate_summary(self, df: pd.DataFrame) -> str:
        """Generate execution summary"""
        total_domains = len(df)
        available_count = len(df[df['available'] == True])
        scored_count = len(df[df['score'] > 0])
        avg_score = df[df['score'] > 0]['score'].mean() if scored_count > 0 else 0
        top_score = df['score'].max() if scored_count > 0 else 0
        
        summary = f"""
EXECUTION SUMMARY:
  📊 Total domains: {total_domains}
  ✅ Available: {available_count}
  🎯 Scored: {scored_count}
  📈 Average score: {avg_score:.2f}
  🏆 Top score: {top_score:.2f}
  
TOP 5 AVAILABLE DOMAINS:
"""
        
        top_available = df[df['available'] == True].head(5)
        for i, row in top_available.iterrows():
            summary += f"  {i+1}. {row['domain']} - Score: {row['score']:.2f}\n"
        
        perf_report = self.tracker.report()
        summary += f"\n⏱️ Total time: {perf_report['total_time']:.2f}s"
        
        return summary

# Main execution function
async def main():
    """Streamlined main function"""
    
    print("🚀 Refined AI Domain Hunter")
    print("=" * 50)
    
    # Configuration
    config = DomainConfig()
    
    # Initialize hunter
    try:
        hunter = RefinedDomainHunter(config)
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nPlease set your OpenRouter API key:")
        print("export OPENROUTER_API_KEY='your-key-here'")
        return
    
    # Run hunt with customizable parameters
    try:
        # Check if merged_output.txt exists for enhanced analysis
        merged_file = Path("merged_output.txt")
        
        if merged_file.exists():
            print("📄 Found merged_output.txt - Using ENHANCED document analysis mode")
            df = await hunter.hunt_with_document_analysis(
                count=100,              # Number of domains to generate
                check_strategy='smart', # 'dns', 'smart', 'robust'
                score_top_n=50,        # Score top N domains (dual evaluation is expensive)
                document_path="merged_output.txt"
            )
        else:
            print("📝 No merged_output.txt found - Using standard hybrid mode")
            df = await hunter.hunt(
                count=100,              # Number of domains to generate
                mode='hybrid',          # 'basic', 'hybrid', 'emergent'
                check_strategy='smart', # 'dns', 'smart', 'robust'
                score_top_n=50         # Score top N domains (AI calls are expensive)
            )
        
        # Display top results
        print("\n🏆 TOP 10 RESULTS:")
        for i, row in df.head(10).iterrows():
            status_icon = "✅" if row['available'] else "❌"
            print(f"{i+1:2d}. {status_icon} {row['domain']:<20} Score: {row['score']:5.2f} ({row['status']})")
        
        # Show available domains
        available = df[df['available'] == True]
        if not available.empty:
            print(f"\n💎 {len(available)} AVAILABLE DOMAINS FOUND!")
            print("Consider registering these high-value opportunities.")
        
        print(f"\n📁 All results saved to 'results/' folder")
        print(f"📊 Logs saved to 'logs/' folder")
        print(f"🔧 Debug info saved to 'debug/' folder")
        
    except Exception as e:
        logger.error(f"Hunt execution failed: {e}")
        print(f"❌ Hunt failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())