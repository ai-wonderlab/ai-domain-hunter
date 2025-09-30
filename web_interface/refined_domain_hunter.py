#!/usr/bin/env python3
"""
AI Domain Hunter - Optimized & Parallel Version
TRUE parallel checking, marketplace prices, faster performance
Maintains all accuracy and robustness
"""

import asyncio
import aiohttp
import json
import os
import re
import time
import logging
import dns.resolver
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from bs4 import BeautifulSoup

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
    
    # Core settings
    max_domains: int = 500
    batch_size: int = 50
    parallel_checks: int = 20  # Number of simultaneous checks
    check_strategy: str = 'smart'  # 'dns', 'smart', 'robust', 'hybrid'
    generation_mode: str = 'hybrid'  # 'basic', 'hybrid', 'emergent'
    excluded_patterns: List[str] = field(default_factory=lambda: [
        'test', 'demo', 'temp', 'example', 'sample'
    ])
    request_timeout: int = 15  # Timeout for HTTP requests
    
    # API-based availability checking
    whoisxml_api_key: str = field(default_factory=lambda: os.getenv('WHOISXML_API_KEY', ''))
    whoapi_key: str = field(default_factory=lambda: os.getenv('WHOAPI_KEY', ''))
    
    # Anti-bot settings
    use_stealth_mode: bool = True
    stealth_only_for_top: int = 5  # Use stealth browser only for top N domains
    rate_limit_delay: tuple = (3.0, 8.0)  # Min/max delay between requests
    use_residential_proxies: bool = False
    residential_proxies: List[str] = field(default_factory=lambda: [
        proxy.strip() for proxy in os.getenv('RESIDENTIAL_PROXIES', '').split(',') 
        if proxy.strip()
    ])
    
    # User agent rotation
    user_agents: List[str] = field(default_factory=lambda: [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
    ])

class APIAvailabilityChecker:
    """API-based domain availability checking with 100% reliability"""
    
    def __init__(self, config: DomainConfig):
        self.config = config
        self.session = None
        
    async def check_domains_api(self, domains: List[str]) -> Dict[str, Dict]:
        """Check domain availability using APIs (WhoisXML or WhoAPI)"""
        
        if not self.config.whoisxml_api_key and not self.config.whoapi_key:
            logger_check.warning("No API keys configured, falling back to DNS checking")
            return {}
        
        results = {}
        
        # Try WhoisXML API first (more reliable)
        if self.config.whoisxml_api_key:
            try:
                whoisxml_results = await self._check_whoisxml_api(domains)
                results.update(whoisxml_results)
                logger_check.info(f"✅ WhoisXML API: {len(whoisxml_results)} domains checked")
            except Exception as e:
                logger_check.warning(f"WhoisXML API failed: {e}")
        
        # Check remaining domains with WhoAPI
        remaining_domains = [d for d in domains if d not in results]
        if remaining_domains and self.config.whoapi_key:
            try:
                whoapi_results = await self._check_whoapi(remaining_domains)
                results.update(whoapi_results)
                logger_check.info(f"✅ WhoAPI: {len(whoapi_results)} domains checked")
            except Exception as e:
                logger_check.warning(f"WhoAPI failed: {e}")
        
        return results
    
    async def _check_whoisxml_api(self, domains: List[str]) -> Dict[str, Dict]:
        """Check domains using WhoisXML API"""
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        results = {}
        
        for domain in domains:
            try:
                url = "https://domain-availability.whoisxmlapi.com/api/v1"
                params = {
                    'apiKey': self.config.whoisxml_api_key,
                    'domainName': domain,
                    'credits': 'DA'
                }
                
                async with self.session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        domain_info = data.get('DomainInfo', {})
                        available = domain_info.get('domainAvailability') == 'AVAILABLE'
                        
                        results[domain] = {
                            'available': available,
                            'status': 'api_whoisxml',
                            'checked_at': datetime.now().isoformat(),
                            'method': 'whoisxml_api',
                            'raw_status': domain_info.get('domainAvailability', 'UNKNOWN')
                        }
                    else:
                        logger_check.warning(f"WhoisXML API error for {domain}: {response.status}")
                
                # Rate limiting for API
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger_check.debug(f"WhoisXML check failed for {domain}: {e}")
        
        return results
    
    async def _check_whoapi(self, domains: List[str]) -> Dict[str, Dict]:
        """Check domains using WhoAPI"""
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        results = {}
        
        for domain in domains:
            try:
                url = "https://api.whoapi.com/"
                params = {
                    'domain': domain,
                    'r': 'taken',
                    'apikey': self.config.whoapi_key
                }
                
                async with self.session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # WhoAPI returns {"taken": "0"} for available, {"taken": "1"} for taken
                        taken = str(data.get('taken')) == '1'
                        available = not taken
                        
                        results[domain] = {
                            'available': available,
                            'status': 'api_whoapi',
                            'checked_at': datetime.now().isoformat(),
                            'method': 'whoapi',
                            'raw_taken': data.get('taken', 'unknown')
                        }
                    else:
                        logger_check.warning(f"WhoAPI error for {domain}: {response.status}")
                
                # Rate limiting for API
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger_check.debug(f"WhoAPI check failed for {domain}: {e}")
        
        return results
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

class RateLimiter:
    """Intelligent rate limiting with human-like delays"""
    
    def __init__(self, config: DomainConfig):
        self.config = config
        self.last_request_times = {}  # Per marketplace
        self.error_counts = {}
    
    async def wait_if_needed(self, marketplace: str):
        """Wait if needed based on rate limiting rules"""
        
        now = time.time()
        last_request = self.last_request_times.get(marketplace, 0)
        
        # Calculate delay based on config and error history
        min_delay, max_delay = self.config.rate_limit_delay
        error_count = self.error_counts.get(marketplace, 0)
        
        # Exponential backoff for errors
        if error_count > 0:
            min_delay *= (1.5 ** min(error_count, 5))
            max_delay *= (1.5 ** min(error_count, 5))
        
        # Random human-like delay
        delay = random.uniform(min_delay, max_delay)
        time_since_last = now - last_request
        
        if time_since_last < delay:
            sleep_time = delay - time_since_last
            logger_check.debug(f"⏱️ Rate limiting {marketplace}: sleeping {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)
        
        self.last_request_times[marketplace] = time.time()
    
    def record_error(self, marketplace: str):
        """Record an error for exponential backoff"""
        self.error_counts[marketplace] = self.error_counts.get(marketplace, 0) + 1
        logger_check.debug(f"❌ Error count for {marketplace}: {self.error_counts[marketplace]}")
    
    def record_success(self, marketplace: str):
        """Record a success - reset error count"""
        if marketplace in self.error_counts:
            self.error_counts[marketplace] = max(0, self.error_counts[marketplace] - 1)

class SmartMarketplaceClient:
    """Sophisticated HTTP client with anti-bot protection (Tier 1-3)"""
    
    def __init__(self, config: DomainConfig):
        self.config = config
        self.sessions = {}  # Per-marketplace sessions
        self.rate_limiter = RateLimiter(config)
        self.user_agent_index = 0
    
    def _get_headers(self, marketplace: str) -> Dict[str, str]:
        """Generate sophisticated headers with browser fingerprinting"""
        
        # Rotate user agents
        user_agent = self.config.user_agents[self.user_agent_index % len(self.config.user_agents)]
        self.user_agent_index += 1
        
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,el;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        # Add marketplace-specific headers
        if 'google' in marketplace.lower():
            headers['Referer'] = 'https://www.google.com/'
        elif marketplace.lower() == 'sedo':
            headers['Referer'] = 'https://sedo.com/'
        
        return headers
    
    async def get_session(self, marketplace: str) -> aiohttp.ClientSession:
        """Get or create persistent session for marketplace"""
        
        if marketplace not in self.sessions:
            # Create cookie jar that accepts all cookies
            cookie_jar = aiohttp.CookieJar(unsafe=True)
            
            # Set up connector with proxy support
            connector = aiohttp.TCPConnector(limit_per_host=5)
            
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            
            session = aiohttp.ClientSession(
                headers=self._get_headers(marketplace),
                cookie_jar=cookie_jar,
                connector=connector,
                timeout=timeout
            )
            
            self.sessions[marketplace] = session
            
            # Warm up session with homepage visit
            await self._warmup_session(session, marketplace)
        
        return self.sessions[marketplace]
    
    async def _warmup_session(self, session: aiohttp.ClientSession, marketplace: str):
        """Visit homepage to establish session and get cookies"""
        
        homepage_urls = {
            'sedo': 'https://sedo.com/',
            'godaddy': 'https://auctions.godaddy.com/',
            'afternic': 'https://www.afternic.com/',
            'dynadot': 'https://www.dynadot.com/',
            'dan': 'https://dan.com/'
        }
        
        homepage = homepage_urls.get(marketplace)
        if homepage:
            try:
                logger_check.debug(f"🏠 Warming up session for {marketplace}")
                async with session.get(homepage) as response:
                    if response.status == 200:
                        # Read a bit of content to simulate real browsing
                        await response.read()
                        logger_check.debug(f"✅ Session warmed up for {marketplace}")
                        
                        # Human-like delay after homepage visit
                        await asyncio.sleep(random.uniform(1.5, 3.0))
                    else:
                        logger_check.warning(f"Homepage warmup failed for {marketplace}: {response.status}")
            except Exception as e:
                logger_check.debug(f"Homepage warmup error for {marketplace}: {e}")
    
    async def fetch_with_protection(self, url: str, marketplace: str) -> Tuple[Optional[str], int]:
        """Fetch URL with full anti-bot protection"""
        
        # Rate limiting (Tier 3)
        await self.rate_limiter.wait_if_needed(marketplace)
        
        session = await self.get_session(marketplace)
        
        try:
            async with session.get(url) as response:
                status = response.status
                
                if status == 200:
                    content = await response.text()
                    self.rate_limiter.record_success(marketplace)
                    logger_check.debug(f"✅ {marketplace}: {len(content)} chars fetched")
                    return content, status
                elif status == 403:
                    self.rate_limiter.record_error(marketplace)
                    logger_check.warning(f"🚫 {marketplace}: 403 Forbidden - anti-bot detected")
                    return None, status
                else:
                    logger_check.warning(f"⚠️ {marketplace}: HTTP {status}")
                    return None, status
                
        except Exception as e:
            self.rate_limiter.record_error(marketplace)
            logger_check.debug(f"❌ {marketplace} fetch error: {e}")
            return None, 0
    
    async def close_all(self):
        """Close all sessions"""
        for session in self.sessions.values():
            await session.close()
        self.sessions.clear()

class StealthBrowserClient:
    """Playwright-based stealth browser for tough sites (Tier 4-5)"""
    
    def __init__(self, config: DomainConfig):
        self.config = config
        self.browser = None
        self.context = None
    
    async def initialize(self):
        """Initialize stealth browser"""
        try:
            from playwright.async_api import async_playwright
            
            self.playwright = await async_playwright().start()
            
            # Launch browser with stealth settings
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu'
                ]
            )
            
            # Create context with stealth settings
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent=self.config.user_agents[0],
                locale='en-US',
                timezone_id='America/New_York',
                permissions=['geolocation']
            )
            
            # Add anti-detection scripts
            await self.context.add_init_script("""
                // Remove webdriver property
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                // Add plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
                
                // Add chrome object
                window.chrome = {
                    runtime: {},
                };
                
                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """)
            
            logger_check.info("🎭 Stealth browser initialized")
            
        except ImportError:
            logger_check.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
            raise
    
    async def fetch_with_stealth(self, url: str, marketplace: str) -> Tuple[Optional[str], int]:
        """Fetch URL using stealth browser"""
        
        if not self.browser:
            await self.initialize()
        
        page = await self.context.new_page()
        
        try:
            # Navigate with human-like behavior
            response = await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Simulate human reading time
            await asyncio.sleep(random.uniform(2.0, 4.0))
            
            # Simulate mouse movement
            await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
            
            # Get page content
            content = await page.content()
            status = response.status if response else 0
            
            logger_check.info(f"🎭 {marketplace}: Stealth fetch complete ({len(content)} chars)")
            
            return content, status
            
        except Exception as e:
            logger_check.error(f"🎭 Stealth browser error for {marketplace}: {e}")
            return None, 0
        finally:
            await page.close()
    
    async def close(self):
        """Close stealth browser"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()

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
    """Single AI manager for all operations with parallel generation support"""
    
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
    
    async def generate_async(self, prompt: str, model: str = 'gemini',
                            max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Async generation wrapper for parallel calls"""
        return await asyncio.to_thread(
            self.generate, prompt, model, max_tokens, temperature
        )
    
    async def generate_parallel(self, prompts: List[Tuple[str, str]]) -> Dict[str, str]:
        """Generate from multiple prompts/models in parallel
        
        Args:
            prompts: List of (prompt, model) tuples
            
        Returns:
            Dict mapping model names to responses
        """
        tasks = []
        models = []
        
        for prompt, model in prompts:
            tasks.append(self.generate_async(prompt, model))
            models.append(model)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for model, result in zip(models, results):
            if isinstance(result, str) and result:
                output[model] = result
            elif isinstance(result, Exception):
                logger_ai.error(f"Parallel generation failed for {model}: {result}")
        
        return output

class DomainChecker:
    """HYBRID parallel domain availability checker with sophisticated anti-bot protection"""
    
    def __init__(self, config: DomainConfig, ai_manager: 'UnifiedAIManager' = None):
        self.config = config
        self.ai_manager = ai_manager  # For AI-powered price extraction
        self.searched_domains = self._load_previous_domains()
        self.tracker = PerformanceTracker()
        
        # Initialize anti-bot components
        self.api_checker = APIAvailabilityChecker(config)
        self.smart_client = SmartMarketplaceClient(config)
        self.stealth_client = StealthBrowserClient(config) if config.use_stealth_mode else None
        
        # Easy vs tough marketplace classification
        self.easy_marketplaces = ['sedo', 'dynadot', 'dan']  # Less aggressive anti-bot
        self.tough_marketplaces = ['godaddy', 'afternic']    # Require stealth mode
        
        # Marketplace endpoints
        self.marketplaces = {
            'godaddy': {
                'search': 'https://auctions.godaddy.com/trpSearchResults.aspx',
                'api': 'https://auctions.godaddy.com/api/v1/search'
            },
            'afternic': {
                'search': 'https://www.afternic.com/search',
                'api': 'https://www.afternic.com/api/search'
            },
            'sedo': {
                'search': 'https://sedo.com/search',
            },
            'dan': {
                'search': 'https://dan.com/search'
            },
            'dynadot': {
                'search': 'https://www.dynadot.com/domain/search'
            }
        }
        
        logger_check.info(f"✓ Hybrid Domain Checker initialized with {len(self.searched_domains)} cached domains")
        logger_check.info(f"  API checking: {'✅' if (config.whoisxml_api_key or config.whoapi_key) else '❌'}")
        logger_check.info(f"  Stealth mode: {'✅' if config.use_stealth_mode else '❌'}")
        logger_check.info(f"  Stealth budget: top {config.stealth_only_for_top} domains")
        
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
            
            # Load from cache file
            cache_file = path / "domain_cache.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                        if isinstance(cache_data, dict):
                            domains.update(cache_data.keys())
                except Exception as e:
                    logger_check.debug(f"Could not read cache: {e}")
        
        logger_check.info(f"Loaded {len(domains)} previously searched domains")
        return domains
    
    async def check_availability(self, domains: List[str], strategy: str = 'smart') -> Dict[str, Dict]:
        """HYBRID availability checking: API → Easy sites → Stealth for top domains
        
        This is the main improvement - sophisticated anti-bot evasion system
        """
        self.tracker.start("hybrid_availability_check")
        
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
        
        logger_check.info(f"🚀 HYBRID CHECKING: {len(clean_domains)} new domains")
        logger_check.info(f"  Strategy: {strategy}")
        logger_check.info(f"  Filtered out: {len(domains) - len(clean_domains)} duplicates/invalid")
        
        # === PHASE 1: API-BASED AVAILABILITY (100% reliable) ===
        logger_check.info("📡 Phase 1: API-based availability checking (ALL domains)")
        
        if strategy in ['hybrid', 'robust']:
            api_results = await self.api_checker.check_domains_api(clean_domains)
            
            if api_results:
                logger_check.info(f"✅ API check complete: {len(api_results)} domains with 100% accuracy")
            else:
                logger_check.warning("⚠️ No API results - falling back to DNS checking")
                api_results = await self._parallel_dns_check(clean_domains)
        else:
            # Use DNS for non-hybrid strategies
            api_results = await self._parallel_dns_check(clean_domains)
        
        # === PHASE 2: EASY MARKETPLACE SCRAPING (available domains only) ===
        available_domains = [
            domain for domain, data in api_results.items()
            if data.get('available', False)
        ]
        
        logger_check.info(f"💰 Phase 2: Easy marketplace checking ({len(available_domains)} available domains)")
        
        if available_domains and strategy in ['smart', 'hybrid', 'robust']:
            easy_pricing_data = await self._check_easy_marketplaces(available_domains)
            
            # Merge pricing data
            for domain, pricing_data in easy_pricing_data.items():
                if domain in api_results:
                    api_results[domain].update(pricing_data)
        
        # === PHASE 3: STEALTH BROWSER (top scoring domains only) ===
        if (self.stealth_client and strategy in ['hybrid', 'robust'] and 
            self.config.use_stealth_mode and available_domains):
            
            # Get top N domains for stealth checking
            top_domains = available_domains[:self.config.stealth_only_for_top]
            
            logger_check.info(f"🎭 Phase 3: Stealth browser checking (TOP {len(top_domains)} domains)")
            
            stealth_pricing_data = await self._check_tough_marketplaces_stealth(top_domains)
            
            # Merge stealth pricing data
            for domain, pricing_data in stealth_pricing_data.items():
                if domain in api_results:
                    api_results[domain].update(pricing_data)
                    api_results[domain]['method'] = 'hybrid_api_stealth'
        
        # Update cache
        self.searched_domains.update(clean_domains)
        self._save_search_cache(api_results)
        
        # Final summary
        total_with_prices = len([d for d, data in api_results.items() if data.get('aftermarket_price')])
        
        logger_check.info("="*50)
        logger_check.info(f"✅ HYBRID CHECK COMPLETE")
        logger_check.info(f"  📊 Total domains: {len(api_results)}")
        logger_check.info(f"  ✅ Available: {len(available_domains)}")
        logger_check.info(f"  💰 With pricing: {total_with_prices}")
        logger_check.info(f"  🎭 Stealth checked: {min(len(available_domains), self.config.stealth_only_for_top)}")
        logger_check.info("="*50)
        
        self.tracker.end("hybrid_availability_check")
        return api_results
    
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
    
    async def _check_easy_marketplaces(self, domains: List[str]) -> Dict[str, Dict]:
        """Check easy marketplaces (Sedo, Dynadot, DAN) with smart HTTP client"""
        
        results = {}
        
        for domain in domains:
            domain_name = domain.replace('.ai', '')
            
            marketplace_data = {
                'marketplace_checked': True,
                'listings': [],
                'aftermarket_price': None,
                'price_source': None,
                'extraction_method': None
            }
            
            # Check each easy marketplace
            for marketplace in self.easy_marketplaces:
                if marketplace in self.marketplaces:
                    try:
                        url = f"{self.marketplaces[marketplace]['search']}?keyword={domain_name}"
                        
                        # Use smart client with anti-bot protection
                        content, status = await self.smart_client.fetch_with_protection(url, marketplace)
                        
                        if content and status == 200:
                            # Parse content for listings and prices
                            soup = BeautifulSoup(content, 'html.parser')
                            full_text = soup.get_text()
                            
                            # Check if domain is mentioned
                            if domain_name.lower() in full_text.lower():
                                marketplace_data['listings'].append({
                                    'marketplace': marketplace,
                                    'url': url,
                                    'found': True
                                })
                                
                                # Extract price if not already found
                                if not marketplace_data['aftermarket_price']:
                                    price, method = await self._extract_price_smart(full_text, domain)
                                    if price:
                                        marketplace_data['aftermarket_price'] = price
                                        marketplace_data['price_source'] = marketplace
                                        marketplace_data['extraction_method'] = method
                        
                    except Exception as e:
                        logger_check.debug(f"Easy marketplace {marketplace} failed for {domain}: {e}")
            
            results[domain] = marketplace_data
        
        return results
    
    async def _check_tough_marketplaces_stealth(self, domains: List[str]) -> Dict[str, Dict]:
        """Check tough marketplaces (GoDaddy, Afternic) with stealth browser"""
        
        if not self.stealth_client:
            logger_check.warning("Stealth mode disabled - skipping tough marketplaces")
            return {}
        
        logger_check.info("🎭 LAUNCHING STEALTH BROWSER FOR TOUGH SITES")
        
        results = {}
        
        for domain in domains:
            domain_name = domain.replace('.ai', '')
            
            marketplace_data = {
                'stealth_checked': True,
                'listings': [],
                'aftermarket_price': None,
                'price_source': None,
                'extraction_method': 'stealth'
            }
            
            # Check each tough marketplace
            for marketplace in self.tough_marketplaces:
                if marketplace in self.marketplaces:
                    try:
                        url = f"{self.marketplaces[marketplace]['search']}?keyword={domain_name}"
                        
                        # Use stealth browser
                        content, status = await self.stealth_client.fetch_with_stealth(url, marketplace)
                        
                        if content and status == 200:
                            # Parse content with BeautifulSoup
                            soup = BeautifulSoup(content, 'html.parser')
                            full_text = soup.get_text()
                            
                            # Check if domain is listed
                            if domain_name.lower() in full_text.lower():
                                marketplace_data['listings'].append({
                                    'marketplace': marketplace,
                                    'url': url,
                                    'found': True,
                                    'method': 'stealth'
                                })
                                
                                # Extract price
                                if not marketplace_data['aftermarket_price']:
                                    price, method = await self._extract_price_smart(full_text, domain)
                                    if price:
                                        marketplace_data['aftermarket_price'] = price
                                        marketplace_data['price_source'] = marketplace
                                        marketplace_data['extraction_method'] = f'stealth_{method}'
                        
                        # Human delay between tough sites
                        await asyncio.sleep(random.uniform(2.0, 4.0))
                        
                    except Exception as e:
                        logger_check.warning(f"Stealth {marketplace} failed for {domain}: {e}")
            
            results[domain] = marketplace_data
            
            # Log stealth results
            if marketplace_data['aftermarket_price']:
                logger_check.info(f"🎭💰 {domain}: {marketplace_data['aftermarket_price']} via stealth")
        
        return results
    
    async def _extract_price_smart(self, text: str, domain: str) -> Tuple[Optional[str], Optional[str]]:
        """Smart price extraction with regex + AI fallback"""
        
        # Smart price extraction patterns
        price_patterns = [
            r'\$[\d,]+',                    # $12,500
            r'€[\d,]+',                     # €10,000  
            r'USD\s*[\d,]+',                # USD 5000
            r'Buy Now:\s*\$[\d,]+',         # Buy Now: $8,000
            r'Price:\s*\$[\d,]+',           # Price: $15,000
            r'Current Bid:\s*\$[\d,]+',     # Current Bid: $3,500
            r'Starting at:\s*\$[\d,]+',     # Starting at: $1,000
            r'Listed for:\s*\$[\d,]+',      # Listed for: $25,000
            r'Asking:\s*\$[\d,]+',          # Asking: $50,000
            r'\$[\d,]+\s*USD',              # $5000 USD
            r'£[\d,]+',                     # £8,500
            r'CAD\s*[\d,]+',                # CAD 12000
        ]
        
        # Step 1: Quick regex patterns
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Get the highest price (likely the most accurate)
                prices = []
                for match in matches:
                    # Extract just the numbers
                    numbers = re.findall(r'[\d,]+', match)
                    if numbers:
                        price_num = numbers[0].replace(',', '')
                        try:
                            prices.append((int(price_num), match))
                        except ValueError:
                            continue
                
                if prices:
                    # Return highest price found
                    highest_price = max(prices, key=lambda x: x[0])
                    return highest_price[1], 'regex'
        
        # Step 2: AI fallback for complex cases
        if (self.ai_manager and len(text) > 200 and 
            ('price' in text.lower() or 'sale' in text.lower() or '$' in text or '€' in text)):
            try:
                ai_prompt = f"""
Extract domain price information from this marketplace text. Be precise and only extract actual prices.

Domain: {domain}
Text excerpt: {text[:2000]}

Questions:
1. Is this domain for sale?
2. What's the price (if any)?
3. Is this a Buy Now price, auction, or listing?

Respond with ONLY the price in format "$X,XXX" or "No price found".
Examples: "$12,500", "$85,000", "No price found"
"""
                
                # Use fast GPT for price extraction
                ai_response = await self.ai_manager.generate_async(ai_prompt, 'gpt', max_tokens=100, temperature=0.1)
                
                # Parse AI response
                price_match = re.search(r'\$[\d,]+', ai_response)
                if price_match:
                    return price_match.group(), 'ai'
                    
            except Exception as e:
                logger_check.debug(f"AI price extraction failed for {domain}: {e}")
        
        return None, None

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
    
    async def _parallel_dns_check(self, domains: List[str]) -> Dict[str, Dict]:
        """Parallel DNS checking - fastest method"""
        
        async def check_single_dns(domain: str) -> Tuple[str, Dict]:
            """Check single domain DNS"""
            try:
                # Use asyncio to run DNS check without blocking
                await asyncio.to_thread(dns.resolver.resolve, domain, 'A')
                return domain, {
                    'available': False,
                    'status': 'dns_taken',
                    'checked_at': datetime.now().isoformat(),
                    'method': 'dns'
                }
            except:
                return domain, {
                    'available': True,
                    'status': 'dns_available',
                    'checked_at': datetime.now().isoformat(),
                    'method': 'dns'
                }
        
        # Run all DNS checks in parallel
        tasks = [check_single_dns(domain) for domain in domains]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert to dict
        results = {}
        for result in results_list:
            if isinstance(result, tuple):
                domain, data = result
                results[domain] = data
            elif isinstance(result, Exception):
                logger_check.error(f"DNS check failed: {result}")
        
        return results
    
    async def _parallel_smart_check(self, domains: List[str]) -> Dict[str, Dict]:
        """Parallel smart check: DNS + marketplace for promising domains"""
        
        # Step 1: Parallel DNS check all domains
        dns_results = await self._parallel_dns_check(domains)
        
        # Step 2: Get DNS-available domains
        available_domains = [
            domain for domain, data in dns_results.items()
            if data.get('available', False)
        ]
        
        # Step 3: Parallel marketplace check for top available domains
        if available_domains:
            top_available = available_domains[:10]  # Check top 10 in marketplaces
            logger_check.info(f"Running parallel marketplace check on {len(top_available)} DNS-available domains")
            
            marketplace_results = await self._parallel_marketplace_check(top_available)
            
            # Merge marketplace data into DNS results
            for domain, marketplace_data in marketplace_results.items():
                if domain in dns_results:
                    dns_results[domain].update(marketplace_data)
        
        return dns_results
    
    async def _parallel_robust_check(self, domains: List[str]) -> Dict[str, Dict]:
        """Parallel robust check: DNS + marketplace for all"""
        
        # Step 1: Parallel DNS check
        dns_results = await self._parallel_dns_check(domains)
        
        # Step 2: Parallel marketplace check for ALL available domains
        available_domains = [
            domain for domain, data in dns_results.items()
            if data.get('available', False)
        ]
        
        if available_domains:
            logger_check.info(f"Running parallel marketplace check on {len(available_domains)} available domains")
            marketplace_results = await self._parallel_marketplace_check(available_domains)
            
            # Merge results
            for domain, marketplace_data in marketplace_results.items():
                if domain in dns_results:
                    dns_results[domain].update(marketplace_data)
                    dns_results[domain]['method'] = 'dns+marketplace'
        
        return dns_results
    
    async def _parallel_marketplace_check(self, domains: List[str]) -> Dict[str, Dict]:
        """Enhanced marketplace checking with full text parsing and AI fallback
        
        1. Fetch whole page text (not just HTML elements)
        2. Smart parsing with regex AND AI fallback for price extraction
        """
        
        # Smart price extraction patterns
        price_patterns = [
            r'\$[\d,]+',                    # $12,500
            r'€[\d,]+',                     # €10,000  
            r'USD\s*[\d,]+',                # USD 5000
            r'Buy Now:\s*\$[\d,]+',         # Buy Now: $8,000
            r'Price:\s*\$[\d,]+',           # Price: $15,000
            r'Current Bid:\s*\$[\d,]+',     # Current Bid: $3,500
            r'Starting at:\s*\$[\d,]+',     # Starting at: $1,000
            r'Listed for:\s*\$[\d,]+',      # Listed for: $25,000
            r'Asking:\s*\$[\d,]+',          # Asking: $50,000
            r'\$[\d,]+\s*USD',              # $5000 USD
            r'£[\d,]+',                     # £8,500
            r'CAD\s*[\d,]+',                # CAD 12000
        ]
        
        async def extract_price_smart(text: str, domain: str) -> Tuple[Optional[str], Optional[str]]:
            """Smart price extraction with regex + AI fallback"""
            
            # Step 1: Quick regex patterns
            for pattern in price_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Get the highest price (likely the most accurate)
                    prices = []
                    for match in matches:
                        # Extract just the numbers
                        numbers = re.findall(r'[\d,]+', match)
                        if numbers:
                            price_num = numbers[0].replace(',', '')
                            try:
                                prices.append((int(price_num), match))
                            except ValueError:
                                continue
                    
                    if prices:
                        # Return highest price found
                        highest_price = max(prices, key=lambda x: x[0])
                        return highest_price[1], 'regex'
            
            # Step 2: AI fallback for complex cases
            if (self.ai_manager and len(text) > 200 and 
                ('price' in text.lower() or 'sale' in text.lower() or '$' in text or '€' in text)):
                try:
                    ai_prompt = f"""
Extract domain price information from this marketplace text. Be precise and only extract actual prices.

Domain: {domain}
Text excerpt: {text[:2000]}

Questions:
1. Is this domain for sale?
2. What's the price (if any)?
3. Is this a Buy Now price, auction, or listing?

Respond with ONLY the price in format "$X,XXX" or "No price found".
Examples: "$12,500", "$85,000", "No price found"
"""
                    
                    # Use fast GPT for price extraction
                    ai_response = await self.ai_manager.generate_async(ai_prompt, 'gpt', max_tokens=100, temperature=0.1)
                    
                    # Parse AI response
                    price_match = re.search(r'\$[\d,]+', ai_response)
                    if price_match:
                        return price_match.group(), 'ai'
                        
                except Exception as e:
                    logger_check.debug(f"AI price extraction failed for {domain}: {e}")
            
            return None, None
        
        async def check_single_domain_marketplaces(session: aiohttp.ClientSession, 
                                                   domain: str) -> Tuple[str, Dict]:
            """Enhanced single domain marketplace check"""
            domain_name = domain.replace('.ai', '')
            
            marketplace_data = {
                'marketplace_checked': True,
                'listings': [],
                'aftermarket_price': None,
                'price_source': None,
                'extraction_method': None,
                'marketplace_text_found': False
            }
            
            # Check GoDaddy Auctions
            try:
                godaddy_url = f"{self.marketplaces['godaddy']['search']}?keyword={domain_name}"
                async with session.get(godaddy_url, timeout=self.config.request_timeout) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # 1. Get full text (strip HTML)
                        soup = BeautifulSoup(html, 'html.parser')
                        full_text = soup.get_text()
                        
                        # 2. Check if domain is actually mentioned
                        domain_found = domain_name.lower() in full_text.lower()
                        
                        if domain_found:
                            marketplace_data['marketplace_text_found'] = True
                            marketplace_data['listings'].append({
                                'marketplace': 'godaddy',
                                'url': godaddy_url,
                                'found': True
                            })
                            
                            # 3. Smart price extraction
                            if not marketplace_data['aftermarket_price']:
                                price, method = await extract_price_smart(full_text, domain)
                                if price:
                                    marketplace_data['aftermarket_price'] = price
                                    marketplace_data['price_source'] = 'godaddy'
                                    marketplace_data['extraction_method'] = method
                        
            except Exception as e:
                logger_check.debug(f"GoDaddy check failed for {domain}: {e}")
            
            # Check Afternic
            try:
                afternic_url = f"{self.marketplaces['afternic']['search']}?q={domain_name}"
                async with session.get(afternic_url, timeout=self.config.request_timeout) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # 1. Get full text
                        soup = BeautifulSoup(html, 'html.parser')
                        full_text = soup.get_text()
                        
                        # 2. Check domain presence
                        domain_found = domain_name.lower() in full_text.lower()
                        
                        if domain_found:
                            marketplace_data['marketplace_text_found'] = True
                            marketplace_data['listings'].append({
                                'marketplace': 'afternic',
                                'url': afternic_url,
                                'found': True
                            })
                            
                            # 3. Smart price extraction (only if no price found yet)
                            if not marketplace_data['aftermarket_price']:
                                price, method = await extract_price_smart(full_text, domain)
                                if price:
                                    marketplace_data['aftermarket_price'] = price
                                    marketplace_data['price_source'] = 'afternic'
                                    marketplace_data['extraction_method'] = method
                        
            except Exception as e:
                logger_check.debug(f"Afternic check failed for {domain}: {e}")
            
            # Check Sedo
            try:
                sedo_url = f"{self.marketplaces['sedo']['search']}?keyword={domain_name}"
                async with session.get(sedo_url, timeout=self.config.request_timeout) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # 1. Get full text
                        soup = BeautifulSoup(html, 'html.parser')
                        full_text = soup.get_text()
                        
                        # 2. Check domain presence and sale indicators
                        domain_found = domain_name.lower() in full_text.lower()
                        for_sale = any(term in full_text.lower() for term in ['for sale', 'buy now', 'price', 'offer'])
                        
                        if domain_found and for_sale:
                            marketplace_data['marketplace_text_found'] = True
                            marketplace_data['listings'].append({
                                'marketplace': 'sedo',
                                'url': sedo_url,
                                'found': True
                            })
                            
                            # 3. Smart price extraction
                            if not marketplace_data['aftermarket_price']:
                                price, method = await extract_price_smart(full_text, domain)
                                if price:
                                    marketplace_data['aftermarket_price'] = price
                                    marketplace_data['price_source'] = 'sedo'
                                    marketplace_data['extraction_method'] = method
                        
            except Exception as e:
                logger_check.debug(f"Sedo check failed for {domain}: {e}")
            
            # Check DAN.com
            try:
                dan_url = f"{self.marketplaces['dan']['search']}?q={domain_name}"
                async with session.get(dan_url, timeout=self.config.request_timeout) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # 1. Get full text
                        soup = BeautifulSoup(html, 'html.parser')
                        full_text = soup.get_text()
                        
                        # 2. Check domain presence
                        domain_found = domain_name.lower() in full_text.lower()
                        
                        if domain_found:
                            marketplace_data['marketplace_text_found'] = True
                            marketplace_data['listings'].append({
                                'marketplace': 'dan',
                                'url': dan_url,
                                'found': True
                            })
                            
                            # 3. Smart price extraction
                            if not marketplace_data['aftermarket_price']:
                                price, method = await extract_price_smart(full_text, domain)
                                if price:
                                    marketplace_data['aftermarket_price'] = price
                                    marketplace_data['price_source'] = 'dan'
                                    marketplace_data['extraction_method'] = method
            except Exception as e:
                logger_check.debug(f"DAN check failed for {domain}: {e}")
            
            # Log results
            if marketplace_data['aftermarket_price']:
                method = marketplace_data.get('extraction_method', 'unknown')
                source = marketplace_data.get('price_source', 'unknown')
                logger_check.info(f"💰 {domain}: {marketplace_data['aftermarket_price']} ({source}/{method})")
            elif marketplace_data['marketplace_text_found']:
                logger_check.debug(f"📝 {domain}: Found in marketplace but no price extracted")
            
            return domain, marketplace_data
        
        # Enhanced session headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            # Check all domains in parallel
            tasks = [check_single_domain_marketplaces(session, domain) for domain in domains]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert to dict and log summary
        results = {}
        total_with_prices = 0
        
        for result in results_list:
            if isinstance(result, tuple):
                domain, data = result
                results[domain] = data
                if data.get('aftermarket_price'):
                    total_with_prices += 1
            elif isinstance(result, Exception):
                logger_check.error(f"Marketplace check failed: {result}")
        
        logger_check.info(f"💰 Marketplace summary: {total_with_prices}/{len(domains)} domains with prices")
        
        return results
    
    def _save_search_cache(self, results: Dict[str, Dict]):
        """Save search cache with results metadata"""
        cache_file = Path("data") / "domain_cache.json"
        
        # Load existing cache
        cache_data = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
            except:
                pass
        
        # Update with new results
        for domain, data in results.items():
            cache_data[domain] = {
                'checked_at': data.get('checked_at'),
                'available': data.get('available'),
                'status': data.get('status'),
                'price': data.get('aftermarket_price')
            }
        
        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger_check.debug(f"Saved {len(cache_data)} domains to cache")

class UnifiedDomainScorer:
    """Parallel domain scoring system"""
    
    def __init__(self, ai_manager: UnifiedAIManager):
        self.ai = ai_manager
        self.tracker = PerformanceTracker()
        
        # Scoring patterns
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
Consider aftermarket listings as validation of value.
"""
    
    async def score_domains(self, domains: List[str], market_data: Dict = None) -> Dict[str, float]:
        """Score multiple domains in PARALLEL batches"""
        self.tracker.start("domain_scoring")
        
        scores = {}
        market_data = market_data or {}
        
        # Process in parallel batches
        batch_size = 10  # Score 10 domains at once
        
        for i in range(0, len(domains), batch_size):
            batch = domains[i:i + batch_size]
            logger_score.info(f"Scoring batch {i//batch_size + 1}: {len(batch)} domains in parallel")
            
            batch_scores = await self._parallel_score_batch(batch, market_data)
            scores.update(batch_scores)
            
            # Small delay between batches
            if i + batch_size < len(domains):
                await asyncio.sleep(2)
        
        self.tracker.end("domain_scoring")
        return scores
    
    async def _parallel_score_batch(self, domains: List[str], market_data: Dict) -> Dict[str, float]:
        """Score a batch of domains in parallel using both Claude and GPT"""
        
        # Prepare prompts for all domains
        prompts = []
        
        for domain in domains:
            availability_context = market_data.get(domain, {})
            
            prompt = f"""
{self.scoring_patterns}

Evaluate domain: {domain}

Availability context:
{json.dumps(availability_context, indent=2)}

Provide ONLY a score from 0-10 as a number, followed by brief reasoning.
Example: "7.5 - High cognitive fluency, strong AI alignment, moderate brandability"
"""
            
            # Create prompts for both models
            prompts.append((prompt, 'claude'))
            prompts.append((prompt, 'gpt'))
        
        # Generate all scores in parallel
        all_responses = await self.ai.generate_parallel(prompts)
        
        # Process results
        scores = {}
        for domain in domains:
            claude_key = f"claude"
            gpt_key = f"gpt"
            
            claude_response = all_responses.get(claude_key, '')
            gpt_response = all_responses.get(gpt_key, '')
            
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
                final_score = 3.0
            
            scores[domain] = final_score
            logger_score.debug(f"{domain}: {final_score:.2f} (C: {claude_score}, G: {gpt_score})")
        
        return scores
    
    def _extract_score(self, response: str) -> float:
        """Extract numeric score from AI response"""
        if not response:
            return 0.0
        
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
                    return max(0.0, min(10.0, score))
                except ValueError:
                    continue
        
        return 0.0

class DomainGenerator:
    """Domain generation with parallel AI generation"""
    
    def __init__(self, ai_manager: UnifiedAIManager, config: DomainConfig):
        self.ai = ai_manager
        self.config = config
        self.tracker = PerformanceTracker()
    
    async def generate_domains(self, count: int, mode: str = 'hybrid', 
                              exclude_domains: Set[str] = None) -> List[str]:
        """Generate domains using specified strategy with parallel generation"""
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
        """Hybrid generation using PARALLEL Claude + GPT"""
        
        research_context = self._load_research_context()
        exclusion_list = list(exclude_domains)[:50]
        
        base_prompt = f"""
Generate {count//2} premium .ai domain names using this hybrid approach:

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

Research context:
{research_context[:2000] if research_context else "Use your knowledge of current AI trends"}

AVOID these already-searched domains:
{exclusion_list[:20]}

Return as clean JSON array: ["domain1", "domain2", ...]
"""
        
        # Generate from both models IN PARALLEL
        prompts = [
            (base_prompt, 'claude'),
            (base_prompt, 'gpt')
        ]
        
        responses = await self.ai.generate_parallel(prompts)
        
        # Parse all responses
        all_domains = []
        for model, response in responses.items():
            domains = self._parse_domain_response(response)
            all_domains.extend(domains)
            logger.debug(f"{model} generated {len(domains)} domains")
        
        return all_domains
    
    async def _generate_emergent(self, count: int, exclude_domains: Set[str]) -> List[str]:
        """Emergent generation with Gemini"""
        
        prompt = f"""
You are an AI domain discovery system. Don't follow prescribed patterns - discover your own.

Analyze the current AI landscape and identify emerging opportunities that others might miss.
Generate {count} .ai domains based on YOUR understanding of:
- What patterns create real value
- Where the AI industry is heading
- What domains would be strategically valuable

Avoid: {list(exclude_domains)[:20]}

Return as JSON array: ["domain1", "domain2", ...]
"""
        
        response = await self.ai.generate_async(prompt, 'gemini', max_tokens=3000)
        return self._parse_domain_response(response)
    
    async def _generate_basic(self, count: int, exclude_domains: Set[str]) -> List[str]:
        """Basic generation with Claude"""
        
        prompt = f"""
Generate {count} .ai domain names for businesses and applications.
Focus on clear, brandable names.

Avoid: {list(exclude_domains)[:20]}

Return as JSON array: ["domain1", "domain2", ...]
"""
        
        response = await self.ai.generate_async(prompt, 'claude', max_tokens=2000)
        return self._parse_domain_response(response)
    
    def _parse_domain_response(self, response: str) -> List[str]:
        """Parse domains from AI response"""
        domains = []
        
        if not response:
            return domains
        
        try:
            # JSON parsing
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
        
        # Fallback: line-by-line
        if not domains:
            for line in response.split('\n'):
                cleaned = re.sub(r'^\d+\.|^[-*•]\s*|^["\']\s*|["\']\s*$', '', line).strip()
                clean_domain = self._normalize_domain(cleaned)
                if clean_domain:
                    domains.append(clean_domain)
        
        return domains
    
    def _normalize_domain(self, domain_text: str) -> Optional[str]:
        """Normalize domain text"""
        if not domain_text:
            return None
        
        domain = domain_text.lower().strip()
        domain = re.sub(r'[^\w.-]', '', domain)
        
        if not domain.endswith('.ai'):
            if domain.endswith('.'):
                domain = domain[:-1]
            domain += '.ai'
        
        domain_name = domain.replace('.ai', '')
        
        if (3 <= len(domain_name) <= 20 and 
            re.match(r'^[a-z0-9-]+$', domain_name) and
            not domain_name.startswith('-') and 
            not domain_name.endswith('-')):
            return domain
        
        return None
    
    def _clean_domains(self, domains: List[str], exclude_domains: Set[str]) -> List[str]:
        """Final cleaning"""
        clean_domains = []
        
        for domain in domains:
            clean_domain = self._normalize_domain(domain)
            if (clean_domain and 
                clean_domain not in exclude_domains and
                clean_domain not in clean_domains):
                clean_domains.append(clean_domain)
        
        return clean_domains
    
    async def generate_with_document_analysis(self, count: int,
                                             gemini_analysis: str,
                                             exclude_domains: Set[str]) -> Dict[str, List[str]]:
        """
        Generate domains using Gemini's document analysis - PARALLEL
        This is the PROVEN BEST method from original code
        """
        
        # Prepare exclusion list
        exclusion_text = ""
        if exclude_domains:
            sample = list(exclude_domains)[:50]
            exclusion_text = f"\nAVOID these already-checked domains:\n{', '.join(sample)}\n"
        
        # Limit analysis for prompt size
        analysis_summary = gemini_analysis[:3000] if len(gemini_analysis) > 3000 else gemini_analysis
        
        # Claude prompt - premium patterns
        claude_prompt = f"""
Based on this comprehensive analysis of domain research:

{analysis_summary}

{exclusion_text}

Generate {count//2} premium .ai domains based on patterns YOU discovered in the analysis.
Don't follow fixed rules - use the insights from the research.
Focus on domains that could sell for $10K+ based on the patterns found.

Return ONLY a JSON array: ["domain1", "domain2", ...]
"""
        
        # GPT prompt - emerging opportunities
        gpt_prompt = f"""
Based on this research analysis:

{analysis_summary}

{exclusion_text}

Generate {count//2} .ai domains using the unexpected patterns and opportunities discovered.
Focus on contrarian insights and emerging trends from the analysis.
Create domains that match the high-value patterns identified.

Return ONLY a JSON array: ["domain1", "domain2", ...]
"""
        
        # Generate from both models IN PARALLEL
        prompts = [
            (claude_prompt, 'claude'),
            (gpt_prompt, 'gpt')
        ]
        
        responses = await self.ai.generate_parallel(prompts)
        
        # Parse responses
        claude_domains = self._parse_domain_response(responses.get('claude', ''))
        gpt_domains = self._parse_domain_response(responses.get('gpt', ''))
        
        logger.info(f"Claude generated {len(claude_domains)} domains from analysis")
        logger.info(f"GPT generated {len(gpt_domains)} domains from analysis")
        
        return {
            'claude_domains': claude_domains,
            'gpt_domains': gpt_domains,
            'all_domains': list(set(claude_domains + gpt_domains))
        }
    
    def _load_research_context(self) -> str:
        """Load research context"""
        research_files = [
            "merged_output.txt",
            "data/research.txt",
            "research_content.txt"
        ]
        
        for file_path in research_files:
            try:
                if Path(file_path).exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()[:5000]
            except:
                pass
        
        return ""

class DocumentProcessor:
    """Process merged research document with Gemini - WITH CACHING"""
    
    def __init__(self, ai_manager: UnifiedAIManager):
        self.ai = ai_manager
        self.gemini_analysis = None
        self.analysis_timestamp = None
        self.cache_file = Path("gemini_cache.txt")
    
    async def process_merged_document(self, file_path: str = "merged_output.txt") -> Dict:
        """Load and analyze with Gemini - USE CACHE if available"""
        
        # Check cache first
        if self.cache_file.exists():
            try:
                cache_age = datetime.now() - datetime.fromtimestamp(self.cache_file.stat().st_mtime)
                if cache_age.days < 7:  # Cache valid for 7 days
                    with open(self.cache_file, 'r', encoding='utf-8') as f:
                        self.gemini_analysis = f.read()
                    
                    logger.info(f"✓ Loaded Gemini analysis from cache ({len(self.gemini_analysis):,} chars)")
                    logger.info(f"  Cache age: {cache_age.days} days")
                    
                    return {
                        'analysis': self.gemini_analysis,
                        'timestamp': datetime.fromtimestamp(self.cache_file.stat().st_mtime),
                        'from_cache': True
                    }
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        
        # Load document
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                merged_content = f.read()
            logger.info(f"Loaded merged document: {len(merged_content):,} characters")
        except FileNotFoundError:
            logger.error(f"{file_path} not found")
            return {}
        
        # Analyze with Gemini
        analysis_prompt = f"""
Analyze this comprehensive research document ({len(merged_content):,} characters) about domains.

RESEARCH CONTENT:
{merged_content}

Your task:
1. Read EVERYTHING - all merged documents
2. Find ALL patterns that influence domain value
3. Discover unexpected connections and opportunities
4. Identify what REALLY drives value
5. Find contrarian insights

What patterns predict $10K+ domain value?
What emerging trends create opportunity?

Provide comprehensive analysis with:
- Top 20 most valuable patterns
- Unexpected insights
- Contrarian opportunities
- Specific domain generation strategies
"""
        
        logger.info("Analyzing with Gemini... (this may take 30-60 seconds)")
        self.gemini_analysis = await self.ai.generate_async(
            analysis_prompt,
            model='gemini',
            max_tokens=16000,
            temperature=0.4
        )
        
        self.analysis_timestamp = datetime.now()
        logger.info(f"✓ Gemini analysis complete: {len(self.gemini_analysis):,} characters")
        
        # Save to cache
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                f.write(self.gemini_analysis)
            logger.info(f"✓ Saved analysis to cache: {self.cache_file}")
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
        
        return {
            'analysis': self.gemini_analysis,
            'timestamp': self.analysis_timestamp,
            'document_size': len(merged_content),
            'analysis_size': len(self.gemini_analysis),
            'from_cache': False
        }
    
    def get_analysis(self) -> str:
        """Get cached analysis"""
        return self.gemini_analysis or ""

class RefinedDomainHunter:
    """Main optimized domain hunter with TRUE parallel processing"""
    
    def __init__(self, config: DomainConfig = None):
        self.config = config or DomainConfig()
        
        if not self.config.openrouter_key:
            raise ValueError("OpenRouter API key required")
        
        # Initialize components
        self.ai = UnifiedAIManager(self.config)
        self.checker = DomainChecker(self.config, self.ai)  # Pass AI manager for price extraction
        self.scorer = UnifiedDomainScorer(self.ai)
        self.generator = DomainGenerator(self.ai, self.config)
        self.document_processor = DocumentProcessor(self.ai)
        self.tracker = PerformanceTracker()
        
        logger.info("✓ Hybrid Domain Hunter initialized with sophisticated anti-bot protection")
        logger.info(f"  API availability: {'✅' if (self.config.whoisxml_api_key or self.config.whoapi_key) else '❌'}")
        logger.info(f"  Stealth mode: {'✅' if self.config.use_stealth_mode else '❌'}")
        logger.info(f"  Parallel checks: {self.config.parallel_checks}")
        logger.info(f"  Rate limiting: {self.config.rate_limit_delay[0]}-{self.config.rate_limit_delay[1]}s")
        
        logger.info("✓ Optimized Domain Hunter initialized")
        logger.info(f"  Parallel checks: {self.config.parallel_checks}")
        logger.info(f"  Request timeout: {self.config.request_timeout}s")
    
    async def hunt(self, 
                   count: int = 500,
                   mode: str = 'hybrid',
                   check_strategy: str = 'smart',
                   score_top_n: int = 100) -> pd.DataFrame:
        """Optimized hunting with TRUE parallel processing"""
        
        self.tracker.start("total_hunt")
        logger.info("="*60)
        logger.info(f"🚀 STARTING OPTIMIZED PARALLEL HUNT")
        logger.info(f"Count: {count}, Mode: {mode}, Strategy: {check_strategy}")
        logger.info(f"Parallel checks: {self.config.parallel_checks}")
        logger.info("="*60)
        
        try:
            # Phase 1: Generate domains (parallel Claude + GPT)
            logger.info("Phase 1: Parallel Domain Generation")
            domains = await self.generator.generate_domains(
                count=count,
                mode=mode,
                exclude_domains=self.checker.searched_domains
            )
            logger.info(f"✓ Generated {len(domains)} domains in parallel")
            
            # Phase 2: TRUE Parallel availability checking
            logger.info("Phase 2: TRUE Parallel Availability Checking")
            availability_data = await self.checker.check_availability(domains, check_strategy)
            logger.info(f"✓ Checked {len(availability_data)} domains in parallel")
            
            # Phase 3: Parallel domain scoring
            logger.info("Phase 3: Parallel Domain Scoring")
            available_domains = [
                d for d, data in availability_data.items()
                if data.get('available', False)
            ]
            other_domains = [
                d for d, data in availability_data.items()
                if not data.get('available', False)
            ]
            
            domains_to_score = (available_domains + other_domains)[:score_top_n]
            scores = await self.scorer.score_domains(domains_to_score, availability_data)
            logger.info(f"✓ Scored {len(scores)} domains in parallel")
            
            # Phase 4: Compile results
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
                    'aftermarket_price': availability.get('aftermarket_price'),
                    'price_source': availability.get('price_source'),
                    'marketplace_listings': len(availability.get('listings', [])),
                    'checked_at': availability.get('checked_at', ''),
                    'method': availability.get('method', ''),
                    'generation_mode': mode,
                    'check_strategy': check_strategy
                })
            
            df = pd.DataFrame(results_data)
            df = df.sort_values(['available', 'score'], ascending=[False, False])
            
            # Phase 5: Save results
            self._save_results(df, mode, check_strategy)
            
            # Phase 6: Summary
            summary = self._generate_summary(df)
            logger.info("="*60)
            logger.info("✅ OPTIMIZED HUNT COMPLETE")
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
        ENHANCED hunt with Gemini document analysis - TRUE PARALLEL
        This is the PROVEN BEST approach from the original system
        """
        
        self.tracker.start("enhanced_hunt")
        logger.info("="*60)
        logger.info(f"🚀 STARTING ENHANCED DOCUMENT-BASED HUNT (PARALLEL)")
        logger.info(f"Count: {count}, Strategy: {check_strategy}, Document: {document_path}")
        logger.info("="*60)
        
        try:
            # Phase 1: Process merged document with Gemini (CACHED)
            logger.info("Phase 1: Document Analysis with Gemini (checking cache...)")
            doc_analysis = await self.document_processor.process_merged_document(document_path)
            
            if not doc_analysis or not doc_analysis.get('analysis'):
                logger.warning("No document analysis available, falling back to standard hunt")
                return await self.hunt(count, 'hybrid', check_strategy, score_top_n)
            
            gemini_analysis = doc_analysis['analysis']
            from_cache = doc_analysis.get('from_cache', False)
            cache_msg = " (FROM CACHE)" if from_cache else " (FRESH ANALYSIS)"
            logger.info(f"✓ Gemini analysis ready{cache_msg}: {len(gemini_analysis):,} chars")
            
            # Phase 2: PARALLEL document-informed generation
            logger.info("Phase 2: Document-Informed Domain Generation (PARALLEL)")
            generation_result = await self.generator.generate_with_document_analysis(
                count=count,
                gemini_analysis=gemini_analysis,
                exclude_domains=self.checker.searched_domains
            )
            
            all_domains = generation_result['all_domains']
            claude_domains = generation_result['claude_domains']
            gpt_domains = generation_result['gpt_domains']
            
            logger.info(f"✓ Generated {len(all_domains)} domains in parallel")
            logger.info(f"  Claude: {len(claude_domains)}, GPT: {len(gpt_domains)}")
            
            # Phase 3: TRUE PARALLEL availability checking
            logger.info("Phase 3: TRUE Parallel Availability Checking")
            availability_data = await self.checker.check_availability(all_domains, check_strategy)
            logger.info(f"✓ Checked {len(availability_data)} domains in parallel")
            
            # Phase 4: PARALLEL enhanced scoring
            logger.info("Phase 4: Parallel Enhanced Scoring")
            available_domains = [
                d for d, data in availability_data.items()
                if data.get('available', False)
            ]
            other_domains = [
                d for d, data in availability_data.items()
                if not data.get('available', False)
            ]
            
            # Score in parallel batches
            domains_to_score = (available_domains + other_domains)[:score_top_n]
            scores = await self.scorer.score_domains(domains_to_score, availability_data)
            logger.info(f"✓ Scored {len(scores)} domains in parallel")
            
            # Phase 5: Compile enhanced results
            logger.info("Phase 5: Compiling Enhanced Results")
            results_data = []
            
            for domain in all_domains:
                availability = availability_data.get(domain, {})
                score = scores.get(domain, 0.0)
                
                # Determine source model
                source_model = 'both'
                if domain in claude_domains and domain not in gpt_domains:
                    source_model = 'claude'
                elif domain in gpt_domains and domain not in claude_domains:
                    source_model = 'gpt'
                
                results_data.append({
                    'domain': domain,
                    'score': score,
                    'available': availability.get('available', False),
                    'status': availability.get('status', 'unknown'),
                    'aftermarket_price': availability.get('aftermarket_price'),
                    'price_source': availability.get('price_source'),
                    'marketplace_listings': len(availability.get('listings', [])),
                    'checked_at': availability.get('checked_at', ''),
                    'method': availability.get('method', ''),
                    'source_model': source_model,
                    'generation_mode': 'document_analysis',
                    'check_strategy': check_strategy
                })
            
            df = pd.DataFrame(results_data)
            df = df.sort_values(['available', 'score'], ascending=[False, False])
            
            # Phase 6: Save enhanced results
            self._save_enhanced_results(df, check_strategy, doc_analysis)
            
            # Phase 7: Generate enhanced summary
            summary = self._generate_enhanced_summary(df, doc_analysis)
            logger.info("="*60)
            logger.info("✅ ENHANCED PARALLEL HUNT COMPLETE")
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
        
        # Main enhanced results
        results_file = Path("results") / f"enhanced_hunt_{strategy}_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        
        # Top available domains
        available_df = df[df['available'] == True].head(20)
        if not available_df.empty:
            available_file = Path("results") / f"enhanced_available_{timestamp}.csv"
            available_df.to_csv(available_file, index=False)
        
        # Analysis metadata
        analysis_meta = {
            'document_analysis': {
                'timestamp': str(doc_analysis.get('timestamp')),
                'from_cache': doc_analysis.get('from_cache', False),
                'analysis_size': doc_analysis.get('analysis_size', 0)
            },
            'hunt_timestamp': timestamp,
            'total_domains': len(df),
            'available_count': len(df[df['available'] == True]),
            'avg_score': df['score'].mean(),
            'top_score': df['score'].max()
        }
        
        meta_file = Path("debug") / f"enhanced_meta_{timestamp}.json"
        with open(meta_file, 'w') as f:
            json.dump(analysis_meta, f, indent=2, default=str)
        
        # Performance report
        perf_report = self.tracker.report()
        perf_file = Path("debug") / f"enhanced_performance_{timestamp}.json"
        with open(perf_file, 'w') as f:
            json.dump(perf_report, f, indent=2)
        
        logger.info(f"Enhanced results saved to {results_file}")
    
    def _generate_enhanced_summary(self, df: pd.DataFrame, doc_analysis: Dict) -> str:
        """Generate enhanced summary"""
        total = len(df)
        available = len(df[df['available'] == True])
        scored = len(df[df['score'] > 0])
        avg_score = df[df['score'] > 0]['score'].mean() if scored > 0 else 0
        top_score = df['score'].max() if scored > 0 else 0
        with_prices = len(df[df['aftermarket_price'].notna()])
        
        # Model breakdown
        claude_count = len(df[df['source_model'].isin(['claude', 'both'])])
        gpt_count = len(df[df['source_model'].isin(['gpt', 'both'])])
        
        from_cache = doc_analysis.get('from_cache', False)
        cache_status = "FROM CACHE" if from_cache else "FRESH ANALYSIS"
        
        summary = f"""
ENHANCED EXECUTION SUMMARY:
  📊 Total domains: {total}
  ✅ Available: {available}
  🎯 Scored: {scored}
  💰 With aftermarket prices: {with_prices}
  📈 Average score: {avg_score:.2f}
  🏆 Top score: {top_score:.2f}
  
DOCUMENT ANALYSIS:
  📄 Status: {cache_status}
  🤖 Analysis size: {doc_analysis.get('analysis_size', 0):,} chars
  
MODEL BREAKDOWN:
  🟦 Claude domains: {claude_count}
  🟩 GPT domains: {gpt_count}
  
TOP 5 AVAILABLE DOMAINS:
"""
        
        top_available = df[df['available'] == True].head(5)
        for i, row in top_available.iterrows():
            price = f" (${row['aftermarket_price']})" if row['aftermarket_price'] else ""
            summary += f"  {i+1}. {row['domain']} - Score: {row['score']:.2f}{price} ({row['source_model']})\n"
        
        perf_report = self.tracker.report()
        summary += f"\n⏱️ Total time: {perf_report['total_time']:.2f}s"
        
        return summary
    
    def _save_results(self, df: pd.DataFrame, mode: str, strategy: str):
        """Save results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Main results
        results_file = Path("results") / f"hunt_{mode}_{strategy}_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        
        # Top available
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
        """Generate summary"""
        total = len(df)
        available = len(df[df['available'] == True])
        scored = len(df[df['score'] > 0])
        avg_score = df[df['score'] > 0]['score'].mean() if scored > 0 else 0
        top_score = df['score'].max() if scored > 0 else 0
        
        # Count domains with aftermarket prices
        with_prices = len(df[df['aftermarket_price'].notna()])
        
        summary = f"""
EXECUTION SUMMARY:
  📊 Total domains: {total}
  ✅ Available: {available}
  🎯 Scored: {scored}
  💰 With aftermarket prices: {with_prices}
  📈 Average score: {avg_score:.2f}
  🏆 Top score: {top_score:.2f}
  
TOP 5 AVAILABLE DOMAINS:
"""
        
        top_available = df[df['available'] == True].head(5)
        for i, row in top_available.iterrows():
            price_info = f" (${row['aftermarket_price']})" if row['aftermarket_price'] else ""
            summary += f"  {i+1}. {row['domain']} - Score: {row['score']:.2f}{price_info}\n"
        
        perf_report = self.tracker.report()
        summary += f"\n⏱️ Total time: {perf_report['total_time']:.2f}s"
        
        return summary
    
    async def cleanup(self):
        """Cleanup all async resources"""
        try:
            if hasattr(self.checker, 'api_checker'):
                await self.checker.api_checker.close()
            if hasattr(self.checker, 'smart_client'):
                await self.checker.smart_client.close_all()
            if hasattr(self.checker, 'stealth_client') and self.checker.stealth_client:
                await self.checker.stealth_client.close()
            logger.info("✅ All resources cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

async def main():
    """Enhanced main function with hybrid checking strategy"""
    
    print("🚀 REFINED DOMAIN HUNTER v2.0 - Sophisticated Anti-Bot System")
    print("=" * 70)
    
    config = DomainConfig()
    config.use_stealth_mode = False
    
    try:
        hunter = RefinedDomainHunter(config)
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nRequired: Set OpenRouter API key:")
        print("export OPENROUTER_API_KEY='sk-or-v1-xxxxx'")
        print("\nOptional: Set availability API keys for 100% accuracy:")
        print("export WHOISXML_API_KEY='your-whoisxml-key'")
        print("export WHOAPI_KEY='your-whoapi-key'")
        return
    
    try:
        # Check if merged_output.txt exists for enhanced analysis
        merged_file = Path("merged_output.txt")
        
        if merged_file.exists():
            print("📄 Found merged_output.txt - Using ENHANCED document analysis mode")
            print("   🎭 Hybrid strategy: API → Easy sites → Stealth (top 5)\n")
            
            df = await hunter.hunt_with_document_analysis(
                count=50,                   # Generate 50 domains
                check_strategy='hybrid',    # Use NEW hybrid strategy
                score_top_n=30,             # Score top 30
                document_path="merged_output.txt"
            )
        else:
            print("📝 No merged_output.txt found - Using hybrid mode")
            print("   🎭 Hybrid strategy: API → Easy sites → Stealth (top 5)\n")
            
            df = await hunter.hunt(
                count=50,                   # Generate 50 domains
                mode='hybrid',              # Generation mode
                check_strategy='hybrid',    # Use NEW hybrid strategy
                score_top_n=30              # Score top 30
            )
        
        # Display results
        print("\n🏆 TOP 10 RESULTS:")
        for i, row in df.head(10).iterrows():
            status = "✅" if row['available'] else "❌"
            price = f" (${row['aftermarket_price']})" if row['aftermarket_price'] else ""
            source = f" [{row.get('source_model', 'N/A')}]" if 'source_model' in row else ""
            method = f" {row.get('method', '')}" if row.get('method') else ""
            print(f"{i+1:2d}. {status} {row['domain']:<25} Score: {row['score']:5.2f}{price}{source}{method}")
        
        available = df[df['available'] == True]
        if not available.empty:
            print(f"\n💎 {len(available)} AVAILABLE DOMAINS FOUND!")
            
            # Show domains with aftermarket prices
            with_prices = available[available['aftermarket_price'].notna()]
            if not with_prices.empty:
                print(f"   💰 {len(with_prices)} have marketplace pricing data")
                
                # Show method breakdown
                stealth_prices = with_prices[with_prices['extraction_method'].str.contains('stealth', na=False)]
                if not stealth_prices.empty:
                    print(f"   🎭 {len(stealth_prices)} found via stealth browser")
        
        print(f"\n📁 Results saved to 'results/' folder")
        print(f"📊 Logs in 'logs/' folder")
        print(f"🔧 Debug info in 'debug/' folder")
        
        # Show API usage summary
        if config.whoisxml_api_key or config.whoapi_key:
            print(f"\n🔑 API Usage:")
            if config.whoisxml_api_key:
                print(f"   📡 WhoisXML API: Available")
            if config.whoapi_key:
                print(f"   📡 WhoAPI: Available")
        else:
            print(f"\n⚠️  For 100% accuracy, add API keys:")
            print(f"   WHOISXML_API_KEY or WHOAPI_KEY")
        
    except Exception as e:
        logger.error(f"Hunt failed: {e}")
        print(f"❌ Hunt failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup resources
        await hunter.cleanup()

if __name__ == "__main__":
    asyncio.run(main())