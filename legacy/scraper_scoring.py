#!/usr/bin/env python3
"""
AI Domain Hunter - Multi-Provider Version
Supports OpenAI, Claude/Anthropic, and Google Gemini
User can specify which AI to use for each phase
"""

import asyncio
import aiohttp
import json
import time
import re
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from collections import defaultdict
import nltk
from nltk.corpus import words, wordnet
import itertools
import requests
from bs4 import BeautifulSoup
import dns.resolver
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dotenv import load_dotenv
from abc import ABC, abstractmethod

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('words', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('brown', quiet=True)
except:
    pass

# ============================================================================
# AI PROVIDER INTERFACES
# ============================================================================

class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on prompt"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get provider name"""
        pass

class OpenAIProvider(AIProvider):
    """OpenAI/GPT Provider"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI provider initialized with model: {self.model}")
            except ImportError:
                logger.warning("OpenAI library not installed. Run: pip install openai")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000, response_format: str = None) -> str:
        if not self.client:
            return ""
        
        try:
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add response format if specified
            if response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return ""
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def get_name(self) -> str:
        return f"OpenAI/{self.model}"

class ClaudeProvider(AIProvider):
    """Anthropic Claude Provider"""
    
    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        self.client = None
        
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Claude provider initialized with model: {self.model}")
            except ImportError:
                logger.warning("Anthropic library not installed. Run: pip install anthropic")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000, **kwargs) -> str:
        if not self.client:
            return ""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            # Claude returns content as a list of blocks
            return response.content[0].text if response.content else ""
        except Exception as e:
            logger.error(f"Claude generation error: {e}")
            return ""
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def get_name(self) -> str:
        return f"Claude/{self.model}"

class GeminiProvider(AIProvider):
    """Google Gemini Provider"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        self.model = model
        self.client = None
        
        if self.api_key:
            try:
                from google import genai
                from google.genai import types
                self.client = genai.Client(api_key=self.api_key)
                self.types = types
                logger.info(f"Gemini provider initialized with model: {self.model}")
            except ImportError:
                logger.warning("Google GenAI library not installed. Run: pip install google-genai")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000, **kwargs) -> str:
        if not self.client:
            return ""
        
        try:
            # Gemini uses different parameter names
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                ) if hasattr(self, 'types') else None
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return ""
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def get_name(self) -> str:
        return f"Gemini/{self.model}"

# ============================================================================
# AI PROVIDER MANAGER
# ============================================================================

class AIProviderManager:
    """Manages multiple AI providers and routes requests"""
    
    def __init__(self):
        # Initialize all providers
        self.providers = {
            'openai': OpenAIProvider(),
            'claude': ClaudeProvider(),
            'gemini': GeminiProvider()
        }
        
        # Default provider preferences for each task
        self.generation_provider = None
        self.analysis_provider = None
        self.evaluation_provider = None
        
        # Set defaults based on availability
        self._set_default_providers()
        
    def _set_default_providers(self):
        """Set default providers based on what's available"""
        available = [name for name, p in self.providers.items() if p.is_available()]
        
        if available:
            # Default preferences (can be overridden)
            if 'claude' in available:
                self.generation_provider = 'claude'  # Claude is good at creative generation
            elif 'openai' in available:
                self.generation_provider = 'openai'
            else:
                self.generation_provider = available[0]
            
            if 'gemini' in available:
                self.analysis_provider = 'gemini'  # Gemini is good at analysis
            elif 'openai' in available:
                self.analysis_provider = 'openai'
            else:
                self.analysis_provider = available[0]
            
            if 'claude' in available:
                self.evaluation_provider = 'claude'  # Claude is good at detailed evaluation
            elif 'openai' in available:
                self.evaluation_provider = 'openai'
            else:
                self.evaluation_provider = available[0]
            
            logger.info(f"Available AI providers: {available}")
            logger.info(f"Default providers set:")
            logger.info(f"  Generation: {self.generation_provider}")
            logger.info(f"  Analysis: {self.analysis_provider}")
            logger.info(f"  Evaluation: {self.evaluation_provider}")
        else:
            logger.warning("No AI providers configured! Set API keys for at least one provider.")
    
    def set_provider_config(self, generation: str = None, analysis: str = None, evaluation: str = None):
        """Manually set which provider to use for each task"""
        if generation and generation in self.providers and self.providers[generation].is_available():
            self.generation_provider = generation
            logger.info(f"Generation provider set to: {generation}")
        
        if analysis and analysis in self.providers and self.providers[analysis].is_available():
            self.analysis_provider = analysis
            logger.info(f"Analysis provider set to: {analysis}")
        
        if evaluation and evaluation in self.providers and self.providers[evaluation].is_available():
            self.evaluation_provider = evaluation
            logger.info(f"Evaluation provider set to: {evaluation}")
    
    def generate_for_task(self, task: str, prompt: str, **kwargs) -> str:
        """Generate text for a specific task using the appropriate provider"""
        provider_name = None
        
        if task == 'generation':
            provider_name = self.generation_provider
        elif task == 'analysis':
            provider_name = self.analysis_provider
        elif task == 'evaluation':
            provider_name = self.evaluation_provider
        
        if not provider_name:
            logger.warning(f"No provider configured for task: {task}")
            return ""
        
        provider = self.providers[provider_name]
        if not provider.is_available():
            logger.warning(f"Provider {provider_name} not available for task: {task}")
            return ""
        
        return provider.generate(prompt, **kwargs)
    
    def get_status(self) -> Dict:
        """Get status of all providers"""
        return {
            'providers': {
                name: {
                    'available': p.is_available(),
                    'name': p.get_name() if p.is_available() else 'Not configured'
                }
                for name, p in self.providers.items()
            },
            'task_assignments': {
                'generation': self.generation_provider,
                'analysis': self.analysis_provider,
                'evaluation': self.evaluation_provider
            }
        }

# ============================================================================
# DOMAIN CANDIDATE DATA CLASS
# ============================================================================

@dataclass
class DomainCandidate:
    """Data class for domain candidates"""
    name: str
    length: int
    syllables: int
    category: str
    source: str
    is_available: bool = None
    registration_cost: float = 140.00  # Base .ai cost
    cls_score: float = 0.0  # Cognitive-Linguistic Score
    mes_score: float = 0.0  # Market-Economic Score
    hbs_score: float = 0.0  # Hybrid Balanced Score
    avg_score: float = 0.0
    estimated_value: str = ""
    whois_check_time: str = ""
    ai_rationale: str = ""  # AI generation reasoning
    ai_provider: str = ""  # Which AI was used
    market_data: Dict = None

# ============================================================================
# AI-POWERED DOMAIN GENERATOR
# ============================================================================

class MultiAIDomainGenerator:
    """Generate domain names using multiple AI providers based on the valuation framework"""
    
    def __init__(self, ai_manager: AIProviderManager):
        """Initialize with AI provider manager"""
        self.ai_manager = ai_manager
        
        # Load framework once at initialization
        self.framework = self.read_framework_pdf()
        
        # Parse framework into sections for targeted use
        self.framework_sections = self._parse_framework_sections()
    
    def read_framework_pdf(self, pdf_path: str = "ai-domain-valuation-framework.md.pdf") -> str:
        """Read the valuation framework with encoding fallbacks"""
        
        # Try multiple paths and encodings
        possible_paths = [
            pdf_path,
            "ai-domain-valuation-framework.md",
            "./ai-domain-valuation-framework.md",
            "../ai-domain-valuation-framework.md",
            "/mnt/user-data/uploads/ai-domain-valuation-framework.md"
        ]
        
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-8-sig', 'ascii']
        
        for path in possible_paths:
            if not os.path.exists(path):
                continue
                
            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                        logging.info(f"Successfully read framework from {path} with {encoding} encoding")
                        return content
                except Exception as e:
                    logging.debug(f"Failed to read {path} with {encoding}: {e}")
        
        logging.warning("Could not read framework file, using embedded framework summary")
        return self.get_embedded_framework()
    
    def get_embedded_framework(self) -> str:
        """Return embedded framework summary if file cannot be read"""
        return """
AI DOMAIN VALUATION FRAMEWORK - CONDENSED VERSION

SCORING CRITERIA:

1. COGNITIVE LOAD SCORE (CLS) - Weight: 40%
- Length: 4 chars optimal (score 5.0), -0.3 per additional character
- Pronunciation: Easy to say (no complex consonant clusters)
- Memory: Real words > brandable > random (47% better recall for real words)

2. MARKET EFFICIENCY SCORE (MES) - Weight: 35%  
- Scarcity: Single word (5.0) > Two words (3.0) > Three+ (1.0)
- AI Relevance: Direct (5.0) > Related (3.0) > None (1.0)
- Commercial Value: High demand sectors score higher

3. HYBRID BALANCED SCORE (HBS) - Weight: 25%
- Brand Potential: Unique + flexible (5.0) > Generic (1.0)
- Trust Signals: Professional + positive (5.0) > Neutral (3.0)
- Global Appeal: Universal (5.0) > Regional (3.0)

HIGH-VALUE PATTERNS:
- Action verbs: learn, build, create, think ($50K-$500K)
- Tech terms: neural, quantum, cognitive ($25K-$250K)
- Abstract concepts: mind, core, meta ($100K+)

TARGET: Domains scoring 4.0+ are premium investments
"""
    
    def _parse_framework_sections(self) -> Dict[str, str]:
        """Parse framework into sections for targeted use"""
        sections = {
            'full': self.framework,
            'generation': '',  # For generation: overview + patterns + examples
            'scoring': '',     # For scoring: formulas + calculations
            'evaluation': '',  # For evaluation: comprehensive
            'market': ''       # For market analysis: market data + trends
        }
        
        framework_lower = self.framework.lower()
        
        # Find scoring section
        scoring_start = framework_lower.find('scoring criteria')
        if scoring_start == -1:
            scoring_start = framework_lower.find('triple-formula')
        
        scoring_end = framework_lower.find('investment strategy')
        if scoring_end == -1:
            scoring_end = framework_lower.find('part 4:')
        
        if scoring_start != -1 and scoring_end != -1:
            sections['scoring'] = self.framework[scoring_start:scoring_end]
        else:
            sections['scoring'] = self.framework[:4000]
        
        # Find market section
        market_start = framework_lower.find('market opportunity')
        if market_start == -1:
            market_start = framework_lower.find('emerging ai sub-niches')
        
        market_end = framework_lower.find('part 5:')
        if market_end == -1:
            market_end = framework_lower.find('valuation process')
        
        if market_start != -1 and market_end != -1:
            sections['market'] = self.framework[market_start:market_end]
        else:
            sections['market'] = self.framework[4000:7000]
        
        # Generation section
        gen_parts = []
        gen_parts.append(self.framework[:1500])
        
        patterns_start = framework_lower.find('high-value patterns')
        if patterns_start == -1:
            patterns_start = framework_lower.find('premium domain patterns')
        if patterns_start != -1:
            gen_parts.append(self.framework[patterns_start:patterns_start+2000])
        
        semantic_start = framework_lower.find('semantic power words')
        if semantic_start == -1:
            semantic_start = framework_lower.find('top 50 ai value words')
        if semantic_start != -1:
            gen_parts.append(self.framework[semantic_start:semantic_start+1500])
        
        sections['generation'] = '\n'.join(gen_parts) if gen_parts else self.framework[:5000]
        
        # Evaluation gets most comprehensive view
        sections['evaluation'] = self.framework[:7000]
        
        return sections
    
    def generate_single_words_ai(self, max_length: int = 8, count: int = 100) -> List[str]:
        """Generate single dictionary words using AI"""
        
        framework_context = self.framework_sections.get('generation', self.framework[:5000])
        
        prompt = f"""Based on this comprehensive domain valuation framework:

{framework_context}

Generate {count} single-word .ai domains that:
1. Are {3} to {max_length} characters long
2. Are real English words or strong brandables
3. Have clear pronunciation
4. Score high on memorability
5. Have potential AI/tech relevance

Focus on:
- Action verbs (learn, build, create)
- Abstract concepts (mind, logic, vision)
- Tech terms (data, model, neural)
- Business terms (scale, growth, market)

Return ONLY a JSON array of domain names (without .ai extension):
["word1", "word2", "word3", ...]"""

        content = self.ai_manager.generate_for_task('generation', prompt, temperature=0.7, max_tokens=2000)
        
        if not content:
            return []
        
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                domains_list = json.loads(json_match.group())
                return [f"{d.lower()}.ai" for d in domains_list if isinstance(d, str) and 3 <= len(d) <= max_length]
        except Exception as e:
            logger.error(f"JSON parsing error in single words generation: {e}")
        
        return []
    
    def generate_compounds_ai(self, count: int = 100) -> List[str]:
        """Generate two-word compounds using AI"""
        
        framework_context = self.framework_sections.get('generation', self.framework[:5000])
        
        prompt = f"""Based on this comprehensive domain valuation framework:

{framework_context}

Generate {count} two-word compound .ai domains following these patterns:
1. [Adjective] + [AI Noun] (SmartAgent, DeepMind)
2. [Tech Verb] + [Object] (ParseData, TrainModel)
3. [Prefix] + [Core Word] (MetaLearn, HyperScale)

Requirements:
- Total length 6-12 characters
- Easy to pronounce
- Clear AI/tech relevance
- No hyphens or numbers
- Brandable and memorable

The framework shows these high-value patterns, use them as guidance but create new combinations.

Return ONLY a JSON array of domain names (without .ai extension):
["compound1", "compound2", ...]"""

        content = self.ai_manager.generate_for_task('generation', prompt, temperature=0.8, max_tokens=2000)
        
        if not content:
            return []
        
        try:
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                domains_list = json.loads(json_match.group())
                return [f"{d.lower()}.ai" for d in domains_list if isinstance(d, str) and 6 <= len(d) <= 12]
        except Exception as e:
            logger.error(f"JSON parsing error in compounds generation: {e}")
        
        return []
    
    def generate_from_trends_ai(self, count: int = 100) -> List[str]:
        """Generate domains from current AI trends"""
        
        framework_context = self.framework_sections.get('market', self.framework_sections.get('generation', self.framework[:5000]))
        
        prompt = f"""Based on this framework and market analysis:

{framework_context}

Identify the hottest AI trends from the framework and generate {count} domain names targeting them.

The framework identifies these emerging trends - use them as guidance to generate domains:
- Look for sections about "Emerging AI Sub-Niches"
- Note growth rates and trajectory indicators
- Focus on categories showing explosive or rapid growth

Generate domains that:
1. Capture emerging trends from the framework
2. Are 4-12 characters
3. Could become category-defining
4. Have high commercial potential

Return ONLY a JSON array of domain names (without .ai extension):
["trend1", "trend2", ...]"""

        content = self.ai_manager.generate_for_task('generation', prompt, temperature=0.7, max_tokens=2000)
        
        if not content:
            return []
        
        try:
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                domains_list = json.loads(json_match.group())
                return [f"{d.lower()}.ai" for d in domains_list if isinstance(d, str) and 4 <= len(d) <= 12]
        except Exception as e:
            logger.error(f"JSON parsing error in trends generation: {e}")
        
        return []
    
    def evaluate_with_ai(self, domain: str) -> Dict:
        """Get AI evaluation of a domain based on the framework"""
        
        framework_context = self.framework_sections.get('evaluation', self.framework[:7000])
        scoring_section = self.framework_sections.get('scoring', '')
        
        prompt = f"""Evaluate the domain '{domain}' using this comprehensive framework:

{framework_context}

DETAILED SCORING FORMULAS:
{scoring_section[:2000] if scoring_section else ''}

Apply these exact scoring criteria:
1. CLS (Cognitive-Linguistic Score): 
   - Length (4 chars = 5.0, each additional -0.3)
   - Pronunciation ease (1-5)
   - Memory retention potential (1-5)

2. MES (Market-Economic Score):
   - AI relevance (direct = 5.0, tangential = 3.0, none = 1.0)
   - Commercial potential (1-5)
   - Scarcity value (single word = 5.0, two words = 3.0)

3. HBS (Hybrid Balanced Score):
   - Brand potential (1-5)
   - Trust signals (1-5)
   - Global appeal (1-5)

Provide a realistic, critical evaluation. Most domains score 2.5-3.5.
Only exceptional domains score above 4.0.

Return JSON with:
{{
  "cls_score": 0.0,
  "mes_score": 0.0,
  "hbs_score": 0.0,
  "estimated_value": "$X",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "recommendation": "buy/hold/pass"
}}"""

        content = self.ai_manager.generate_for_task('evaluation', prompt, temperature=0.3, max_tokens=500)
        
        if not content:
            return {"error": "No response from AI"}
        
        try:
            return json.loads(content)
        except:
            return {"analysis": content}
    
    def ai_interpret_market_data(self, domain: str, raw_data: Dict) -> Dict:
        """Use AI to interpret raw market data"""
        
        prompt = f"""Analyze this market data for domain '{domain}':

{json.dumps(raw_data, indent=2)[:2000]}

Extract and provide:
1. Estimated market value based on the data
2. Similar domains that sold and their prices
3. Demand indicators
4. Best potential buyers
5. Investment recommendation

Return as JSON with clear insights."""

        content = self.ai_manager.generate_for_task('analysis', prompt, temperature=0.3, max_tokens=500)
        
        if not content:
            return raw_data
        
        try:
            interpreted = json.loads(content)
            raw_data['ai_analysis'] = interpreted
        except:
            raw_data['ai_analysis'] = content
        
        return raw_data

# ============================================================================
# DOMAIN SCORING (UNCHANGED FROM ORIGINAL)
# ============================================================================

class DomainScorer:
    """Implements the three scoring formulas from our framework"""
    
    def __init__(self):
        self.ai_keywords = {
            'premium': ['mind', 'brain', 'think', 'learn', 'vision', 'intelligence', 
                       'wisdom', 'logic', 'neural', 'cognitive', 'smart', 'genius'],
            'strong': ['agent', 'model', 'data', 'algo', 'compute', 'train', 'infer',
                      'predict', 'analyze', 'optimize', 'automate', 'assist'],
            'emerging': ['agentic', 'quantum', 'neuro', 'synthetic', 'augment', 'hybrid',
                        'adaptive', 'generative', 'multimodal', 'foundation'],
            'technical': ['tensor', 'vector', 'embed', 'transform', 'latent', 'gradient',
                         'network', 'layer', 'attention', 'decoder', 'encoder']
        }
        
        self.commercial_verticals = ['health', 'legal', 'finance', 'retail', 'energy',
                                     'education', 'insurance', 'real estate', 'travel']
        
        self.phonetic_scores = {
            'excellent': ['ai', 'ay', 'ee', 'oh'],
            'good': ['ar', 'er', 'or'],
            'poor': ['ough', 'augh', 'eigh']
        }
    
    def calculate_cls(self, domain: DomainCandidate) -> float:
        """Calculate Cognitive-Linguistic Score"""
        score = 0.0
        
        # Length Score (30%)
        if domain.length <= 4:
            length_score = 5.0
        elif domain.length <= 6:
            length_score = 4.0
        elif domain.length <= 8:
            length_score = 3.0
        elif domain.length <= 10:
            length_score = 2.0
        else:
            length_score = 1.0
        score += length_score * 0.30
        
        # Pronunciation Score (25%)
        pronunciation_score = self._calculate_pronunciation(domain.name)
        score += pronunciation_score * 0.25
        
        # Memorability Score (20%)
        memorability_score = self._calculate_memorability(domain.name)
        score += memorability_score * 0.20
        
        # Phonetic Appeal (15%)
        phonetic_score = self._calculate_phonetic_appeal(domain.name)
        score += phonetic_score * 0.15
        
        # Typing Ease (10%)
        typing_score = self._calculate_typing_ease(domain.name)
        score += typing_score * 0.10
        
        return round(score, 2)
    
    def calculate_mes(self, domain: DomainCandidate) -> float:
        """Calculate Market-Economic Score"""
        score = 0.0
        
        # Scarcity Score (35%)
        if domain.length <= 3:
            scarcity_score = 5.0
        elif domain.length == 4 and domain.category == 'dictionary':
            scarcity_score = 4.5
        elif domain.length <= 6 and domain.category in ['dictionary', 'premium_compound']:
            scarcity_score = 3.5
        elif domain.length <= 8:
            scarcity_score = 2.5
        else:
            scarcity_score = 1.5
        score += scarcity_score * 0.35
        
        # AI Relevance (25%)
        ai_score = self._calculate_ai_relevance(domain.name)
        score += ai_score * 0.25
        
        # Commercial Intent (20%)
        commercial_score = self._calculate_commercial_intent(domain.name)
        score += commercial_score * 0.20
        
        # Trend Alignment (15%)
        trend_score = self._calculate_trend_alignment(domain.name)
        score += trend_score * 0.15
        
        # Network Effects (5%)
        network_score = self._calculate_network_potential(domain.name)
        score += network_score * 0.05
        
        return round(score, 2)
    
    def calculate_hbs(self, domain: DomainCandidate) -> float:
        """Calculate Hybrid Balanced Score"""
        score = 0.0
        
        # Cognitive Efficiency (20%)
        cognitive = (self._calculate_pronunciation(domain.name) + 
                    self._calculate_memorability(domain.name) + 
                    (5.0 if domain.length <= 6 else 3.0)) / 3
        score += cognitive * 0.20
        
        # Market Position (20%)
        market = (self._calculate_ai_relevance(domain.name) + 
                 (4.0 if domain.length <= 5 else 2.5)) / 2
        score += market * 0.20
        
        # Brand Potential (20%)
        brand_score = self._calculate_brand_potential(domain.name)
        score += brand_score * 0.20
        
        # Trust & Authority (15%)
        trust_score = self._calculate_trust_score(domain.name)
        score += trust_score * 0.15
        
        # Technical Performance (10%)
        tech_score = (self._calculate_typing_ease(domain.name) + 
                     self._calculate_voice_compatibility(domain.name)) / 2
        score += tech_score * 0.10
        
        # Legal Safety (10%)
        legal_score = self._calculate_legal_safety(domain.name)
        score += legal_score * 0.10
        
        # Global Scalability (5%)
        global_score = self._calculate_global_appeal(domain.name)
        score += global_score * 0.05
        
        return round(score, 2)
    
    # [Include all the private calculation methods from the original script]
    # _calculate_pronunciation, _calculate_memorability, etc.
    # (These remain unchanged from your original script)
    
    def _calculate_pronunciation(self, name: str) -> float:
        name = name.replace('.ai', '')
        difficult_patterns = ['xz', 'qz', 'vw', 'ght', 'tch', 'dge']
        for pattern in difficult_patterns:
            if pattern in name.lower():
                return 2.0
        vowels = sum(1 for c in name.lower() if c in 'aeiou')
        consonants = sum(1 for c in name.lower() if c.isalpha() and c not in 'aeiou')
        if vowels == 0 or consonants / (vowels + 1) > 3:
            return 2.5
        if len(name) <= 5 and vowels >= 1:
            return 5.0
        elif len(name) <= 8:
            return 4.0
        else:
            return 3.0
    
    def _calculate_memorability(self, name: str) -> float:
        name = name.replace('.ai', '')
        try:
            if name.lower() in words.words():
                base_score = 4.5
            else:
                base_score = 3.0
        except:
            base_score = 3.0
        if len(name) <= 4:
            base_score += 0.5
        elif len(name) > 8:
            base_score -= 1.0
        if any(name.count(c) > 2 for c in name):
            base_score += 0.3
        return min(5.0, max(1.0, base_score))
    
    def _calculate_phonetic_appeal(self, name: str) -> float:
        name = name.replace('.ai', '').lower()
        if any(pattern in name for pattern in ['ly', 'la', 'lo', 'na', 'ra']):
            return 4.5
        if any(pattern in name for pattern in ['ck', 'sk', 'gz', 'ks']):
            return 2.5
        if name[0] in 'ptkbdg':
            return 4.0
        return 3.0
    
    def _calculate_typing_ease(self, name: str) -> float:
        name = name.replace('.ai', '')
        left_hand = set('qwertasdfgzxcvb')
        right_hand = set('yuiophjklnm')
        alternating = 0
        for i in range(len(name) - 1):
            if (name[i].lower() in left_hand and name[i+1].lower() in right_hand) or \
               (name[i].lower() in right_hand and name[i+1].lower() in left_hand):
                alternating += 1
        if alternating > len(name) * 0.6:
            typing_score = 4.5
        else:
            typing_score = 3.0
        if any(name[i] == name[i+1] for i in range(len(name)-1)):
            typing_score -= 0.5
        if len(name) <= 5:
            typing_score += 0.5
        return min(5.0, max(1.0, typing_score))
    
    def _calculate_ai_relevance(self, name: str) -> float:
        name_lower = name.replace('.ai', '').lower()
        for keyword in self.ai_keywords['premium']:
            if keyword in name_lower:
                return 5.0
        for keyword in self.ai_keywords['strong']:
            if keyword in name_lower:
                return 4.0
        for keyword in self.ai_keywords['emerging']:
            if keyword in name_lower:
                return 3.5
        for keyword in self.ai_keywords['technical']:
            if keyword in name_lower:
                return 3.0
        tech_terms = ['tech', 'digital', 'cyber', 'data', 'cloud', 'quantum']
        if any(term in name_lower for term in tech_terms):
            return 2.5
        return 1.5
    
    def _calculate_commercial_intent(self, name: str) -> float:
        name_lower = name.replace('.ai', '').lower()
        for vertical in self.commercial_verticals:
            if vertical in name_lower:
                return 5.0
        business_terms = ['business', 'enterprise', 'solution', 'platform', 'service',
                         'product', 'market', 'trade', 'commerce', 'pay']
        if any(term in name_lower for term in business_terms):
            return 4.0
        action_verbs = ['buy', 'sell', 'get', 'find', 'book', 'hire', 'build', 'create']
        if any(verb in name_lower for verb in action_verbs):
            return 3.5
        return 2.0
    
    def _calculate_trend_alignment(self, name: str) -> float:
        name_lower = name.replace('.ai', '').lower()
        hot_trends = ['agent', 'agentic', 'gpt', 'llm', 'rag', 'multimodal',
                     'foundation', 'generative', 'prompt', 'embedding']
        for trend in hot_trends:
            if trend in name_lower:
                return 5.0
        emerging = ['neuro', 'quantum', 'edge', 'federated', 'causal', 'symbolic']
        for trend in emerging:
            if trend in name_lower:
                return 4.0
        return 2.5
    
    def _calculate_network_potential(self, name: str) -> float:
        name_lower = name.replace('.ai', '').lower()
        platform_terms = ['connect', 'network', 'social', 'share', 'community',
                         'marketplace', 'exchange', 'hub', 'platform']
        if any(term in name_lower for term in platform_terms):
            return 4.5
        ecosystem_terms = ['api', 'integrate', 'plugin', 'app', 'tool', 'suite']
        if any(term in name_lower for term in ecosystem_terms):
            return 3.5
        return 2.0
    
    def _calculate_brand_potential(self, name: str) -> float:
        name = name.replace('.ai', '')
        try:
            if name.lower() not in words.words() and len(name) <= 8:
                if self._calculate_pronunciation(name) >= 4.0:
                    return 4.5
        except:
            pass
        try:
            if name.lower() in words.words() and len(name) <= 6:
                return 4.0
        except:
            pass
        if len(name) > 6 and len(name) <= 12:
            return 3.0
        return 2.0
    
    def _calculate_trust_score(self, name: str) -> float:
        name_lower = name.replace('.ai', '').lower()
        professional = ['pro', 'expert', 'master', 'certified', 'official',
                       'authentic', 'verified', 'trusted', 'secure']
        if any(term in name_lower for term in professional):
            return 4.0
        negative = ['fake', 'scam', 'spam', 'junk', 'trash', 'stupid', 'dumb']
        if any(term in name_lower for term in negative):
            return 1.0
        if len(name) <= 8 and name.isalpha():
            return 3.5
        return 3.0
    
    def _calculate_voice_compatibility(self, name: str) -> float:
        name = name.replace('.ai', '').lower()
        if not name.isalpha():
            return 2.0
        if self._calculate_pronunciation(name) >= 4.0:
            return 4.5
        if len(name) <= 6:
            return 4.0
        elif len(name) <= 10:
            return 3.0
        else:
            return 2.0
    
    def _calculate_legal_safety(self, name: str) -> float:
        name_lower = name.replace('.ai', '').lower()
        risky_terms = ['google', 'apple', 'microsoft', 'amazon', 'facebook', 'meta',
                      'tesla', 'netflix', 'adobe', 'oracle', 'salesforce', 'nvidia']
        for term in risky_terms:
            if term in name_lower:
                return 1.0
        try:
            if name_lower in words.words():
                return 4.0
        except:
            pass
        return 3.5
    
    def _calculate_global_appeal(self, name: str) -> float:
        name = name.replace('.ai', '').lower()
        try:
            if name in words.words() and len(name) <= 6:
                return 4.0
        except:
            pass
        if len(name) <= 4:
            return 4.5
        tech_universal = ['data', 'code', 'app', 'web', 'net', 'tech', 'digital']
        if any(term in name for term in tech_universal):
            return 3.5
        return 2.5
    
    def get_value_estimate(self, domain: DomainCandidate) -> str:
        """Estimate value based on scores"""
        avg = domain.avg_score
        if avg >= 4.5:
            return "$500K-$1M+"
        elif avg >= 4.0:
            return "$100K-$500K"
        elif avg >= 3.5:
            return "$25K-$100K"
        elif avg >= 3.0:
            return "$5K-$25K"
        elif avg >= 2.5:
            return "$1K-$5K"
        else:
            return "<$1K"

# ============================================================================
# DOMAIN AVAILABILITY CHECKER (UNCHANGED)
# ============================================================================

class DomainChecker:
    """Checks domain availability using multiple methods"""
    
    def __init__(self):
        self.session = requests.Session()
        self.checked_domains = {}
        
    async def check_availability_bulk(self, domains: List[str]) -> Dict[str, bool]:
        """Check multiple domains asynchronously"""
        results = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for domain in domains:
                task = self.check_single_domain(session, domain)
                tasks.append(task)
            
            batch_size = 10
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                
                for j, result in enumerate(batch_results):
                    domain_name = domains[i + j]
                    if isinstance(result, Exception):
                        results[domain_name] = None
                    else:
                        results[domain_name] = result
                
                await asyncio.sleep(1)
        
        return results
    
    async def check_single_domain(self, session: aiohttp.ClientSession, domain: str) -> bool:
        """Check if a single domain is available"""
        try:
            resolver = dns.resolver.Resolver()
            resolver.timeout = 2
            resolver.lifetime = 2
            answers = resolver.resolve(domain, 'A')
            return False
        except:
            pass
        try:
            return True
        except:
            return None
    
    def check_availability_simple(self, domain: str) -> bool:
        """Simple synchronous availability check"""
        try:
            resolver = dns.resolver.Resolver()
            resolver.timeout = 2
            resolver.lifetime = 2
            resolver.resolve(domain, 'A')
            return False
        except:
            return True

# ============================================================================
# MAIN DOMAIN HUNTER ORCHESTRATOR
# ============================================================================

class MultiAIDomainHunter:
    """Main orchestrator for multi-AI domain hunting"""
    
    def __init__(self, provider_config: Dict[str, str] = None):
        """Initialize with optional provider configuration
        
        Args:
            provider_config: Dict specifying which provider to use for each task
                            e.g., {'generation': 'claude', 'analysis': 'gemini', 'evaluation': 'openai'}
        """
        self.ai_manager = AIProviderManager()
        
        # Apply custom configuration if provided
        if provider_config:
            self.ai_manager.set_provider_config(
                generation=provider_config.get('generation'),
                analysis=provider_config.get('analysis'),
                evaluation=provider_config.get('evaluation')
            )
        
        self.generator = MultiAIDomainGenerator(self.ai_manager)
        self.scorer = DomainScorer()
        self.checker = DomainChecker()
        self.results = []
    
    async def hunt_domains(self, 
                          max_domains: int = 500,
                          check_availability: bool = False,
                          use_ai_generation: bool = True,
                          ai_domains_count: int = 100,
                          focus_area: str = "general") -> pd.DataFrame:
        """Main hunting process with multi-AI support"""
        
        logger.info("Starting multi-AI domain hunting process...")
        
        # Show AI provider status
        status = self.ai_manager.get_status()
        logger.info(f"AI Provider Configuration:")
        for task, provider in status['task_assignments'].items():
            logger.info(f"  {task}: {provider} ({status['providers'][provider]['name'] if provider else 'None'})")
        
        # Step 1: Generate domain candidates
        logger.info("Generating domain candidates...")
        all_domains = []
        
        if use_ai_generation:
            logger.info(f"Using {self.ai_manager.generation_provider} for domain generation...")
            
            # Generate different types of domains
            logger.info("Generating single words...")
            ai_singles = self.generator.generate_single_words_ai(max_length=8, count=ai_domains_count)
            all_domains.extend(ai_singles)
            
            logger.info("Generating compounds...")
            ai_compounds = self.generator.generate_compounds_ai(count=ai_domains_count)
            all_domains.extend(ai_compounds)
            
            logger.info("Generating trending domains...")
            ai_trends = self.generator.generate_from_trends_ai(count=ai_domains_count)
            all_domains.extend(ai_trends)
            
            logger.info(f"AI generated {len(all_domains)} domain suggestions")
        else:
            # Fallback to basic generation
            logger.warning("AI generation disabled, using basic methods")
            word_list = set(words.words()) if 'words' in dir(words) else set()
            for word in list(word_list)[:200]:
                if 3 <= len(word) <= 8 and word.isalpha():
                    all_domains.append(f"{word.lower()}.ai")
        
        # Remove duplicates and limit
        all_domains = list(set(all_domains))[:max_domains]
        logger.info(f"Generated {len(all_domains)} unique domain candidates")
        
        # Step 2: Check availability
        availability = {}
        if check_availability:
            logger.info("Checking domain availability...")
            availability = await self.checker.check_availability_bulk(all_domains)
        else:
            availability = {d: True for d in all_domains}
        
        # Step 3: Score all domains
        logger.info("Scoring domains...")
        for domain_name in all_domains:
            if check_availability and not availability.get(domain_name, False):
                continue
            
            # Create candidate
            candidate = DomainCandidate(
                name=domain_name,
                length=len(domain_name.split('.')[0]),
                syllables=self._count_syllables(domain_name.split('.')[0]),
                category=self._categorize_domain(domain_name),
                source='ai_generated' if use_ai_generation else 'basic',
                is_available=availability.get(domain_name, None),
                ai_provider=self.ai_manager.generation_provider if use_ai_generation else 'none'
            )
            
            # Calculate scores
            candidate.cls_score = self.scorer.calculate_cls(candidate)
            candidate.mes_score = self.scorer.calculate_mes(candidate)
            candidate.hbs_score = self.scorer.calculate_hbs(candidate)
            candidate.avg_score = round((candidate.cls_score + 
                                        candidate.mes_score + 
                                        candidate.hbs_score) / 3, 2)
            
            # Get value estimate
            candidate.estimated_value = self.scorer.get_value_estimate(candidate)
            
            # If score is high enough, get AI evaluation
            if candidate.avg_score >= 3.5:
                logger.info(f"Getting {self.ai_manager.evaluation_provider} evaluation for {domain_name}...")
                ai_eval = self.generator.evaluate_with_ai(domain_name)
                if ai_eval and 'recommendation' in ai_eval:
                    candidate.ai_rationale = f"AI: {ai_eval.get('recommendation', '')} - {ai_eval.get('strengths', [])}"
            
            self.results.append(candidate)
        
        logger.info(f"Scored {len(self.results)} available domains")
        
        # Step 4: Convert to DataFrame and sort
        df = pd.DataFrame([asdict(r) for r in self.results])
        df = df.sort_values('avg_score', ascending=False)
        
        return df
    
    def get_top_domains(self, df: pd.DataFrame, top_n: int = 100) -> Dict:
        """Get top domains by each scoring method"""
        
        results = {
            'top_by_cls': df.nlargest(top_n, 'cls_score')[
                ['name', 'cls_score', 'estimated_value', 'ai_provider']
            ].to_dict('records'),
            
            'top_by_mes': df.nlargest(top_n, 'mes_score')[
                ['name', 'mes_score', 'estimated_value', 'ai_provider']
            ].to_dict('records'),
            
            'top_by_hbs': df.nlargest(top_n, 'hbs_score')[
                ['name', 'hbs_score', 'estimated_value', 'ai_provider']
            ].to_dict('records'),
            
            'top_by_average': df.nlargest(top_n, 'avg_score')[
                ['name', 'avg_score', 'cls_score', 'mes_score', 'hbs_score', 
                 'estimated_value', 'ai_provider']
            ].to_dict('records'),
            
            'triple_crown': self._find_triple_crown(df, top_n)
        }
        
        return results
    
    def _find_triple_crown(self, df: pd.DataFrame, top_n: int) -> List[Dict]:
        """Find domains in top N of all three scoring methods"""
        
        top_cls = set(df.nlargest(top_n, 'cls_score')['name'])
        top_mes = set(df.nlargest(top_n, 'mes_score')['name'])
        top_hbs = set(df.nlargest(top_n, 'hbs_score')['name'])
        
        triple_crown = top_cls & top_mes & top_hbs
        
        results = []
        for name in triple_crown:
            row = df[df['name'] == name].iloc[0]
            results.append({
                'name': name,
                'cls_score': row['cls_score'],
                'mes_score': row['mes_score'],
                'hbs_score': row['hbs_score'],
                'avg_score': row['avg_score'],
                'estimated_value': row['estimated_value'],
                'ai_provider': row['ai_provider']
            })
        
        results.sort(key=lambda x: x['avg_score'], reverse=True)
        
        return results
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count"""
        word = word.lower()
        vowels = 'aeiouy'
        syllables = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllables -= 1
        
        return max(1, syllables)
    
    def _categorize_domain(self, domain: str) -> str:
        """Categorize domain type"""
        name = domain.split('.')[0].lower()
        
        try:
            if name in words.words():
                return 'dictionary'
        except:
            pass
        
        if len(name) <= 5:
            return 'short_brandable'
        elif any(term in name for term in ['ai', 'ml', 'tech', 'data']):
            return 'tech_compound'
        else:
            return 'creative'
    
    def export_results(self, df: pd.DataFrame, filename: str = 'multi_ai_domain_results.csv'):
        """Export results to CSV"""
        df.to_csv(filename, index=False)
        logger.info(f"Results exported to {filename}")
    
    def generate_report(self, results: Dict) -> str:
        """Generate a text report of findings"""
        
        report = []
        report.append("=" * 80)
        report.append("MULTI-AI DOMAIN HUNTER - RESULTS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show AI provider status
        status = self.ai_manager.get_status()
        report.append("\nAI PROVIDERS USED:")
        for task, provider in status['task_assignments'].items():
            if provider:
                report.append(f"  {task.capitalize()}: {status['providers'][provider]['name']}")
        report.append("")
        
        # Triple Crown Winners
        report.append("-" * 80)
        report.append("TRIPLE CROWN WINNERS (Top in ALL three scoring methods)")
        report.append("-" * 80)
        
        if results['triple_crown']:
            for i, domain in enumerate(results['triple_crown'][:20], 1):
                report.append(f"{i}. {domain['name']}")
                report.append(f"   CLS: {domain['cls_score']} | MES: {domain['mes_score']} | HBS: {domain['hbs_score']}")
                report.append(f"   Estimated Value: {domain['estimated_value']}")
                report.append(f"   Generated by: {domain['ai_provider']}")
                report.append("")
        else:
            report.append("No domains found in top 100 of all three methods")
            report.append("")
        
        # Top by Average Score
        report.append("-" * 80)
        report.append("TOP 20 BY AVERAGE SCORE")
        report.append("-" * 80)
        
        for i, domain in enumerate(results['top_by_average'][:20], 1):
            report.append(f"{i}. {domain['name']} - Avg: {domain['avg_score']}")
            report.append(f"   Value: {domain['estimated_value']} | AI: {domain['ai_provider']}")
        
        report.append("")
        report.append("-" * 80)
        report.append("INVESTMENT SUMMARY")
        report.append("-" * 80)
        
        premium_count = sum(1 for d in results['top_by_average'] 
                          if d['avg_score'] >= 4.0)
        investment_count = sum(1 for d in results['top_by_average'] 
                             if 3.5 <= d['avg_score'] < 4.0)
        opportunity_count = sum(1 for d in results['top_by_average'] 
                              if 3.0 <= d['avg_score'] < 3.5)
        
        report.append(f"Premium Tier (4.0+): {premium_count} domains")
        report.append(f"Investment Grade (3.5-3.99): {investment_count} domains")
        report.append(f"Opportunity Zone (3.0-3.49): {opportunity_count} domains")
        
        total_cost = len(results['top_by_average']) * 140
        report.append(f"\nTotal Registration Cost (Top 100): ${total_cost:,}")
        
        return "\n".join(report)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function with multi-AI support"""
    
    print("=" * 80)
    print("MULTI-AI DOMAIN HUNTER")
    print("=" * 80)
    print("")
    
    # Check which AI providers are available
    print("Checking AI provider availability...")
    print("")
    
    # You can customize which AI to use for each phase
    # Example configurations:
    
    # Config 1: Use Claude for generation, Gemini for analysis, OpenAI for evaluation
    config1 = {
        'generation': 'claude',
        'analysis': 'gemini',
        'evaluation': 'claude'
    }
    
    # Config 2: Use OpenAI for everything
    config2 = {
        'generation': 'openai',
        'analysis': 'openai',
        'evaluation': 'openai'
    }
    
    # Config 3: Use Gemini for generation and analysis, Claude for evaluation
    config3 = {
        'generation': 'gemini',
        'analysis': 'gemini',
        'evaluation': 'claude'
    }
    
    # Choose your configuration (or let it auto-detect best options)
    # To use auto-detection, pass None or {}
    # To use a specific config, pass one of the configs above
    
    hunter = MultiAIDomainHunter(provider_config=config1)  # Auto-detect best providers
    # hunter = MultiAIDomainHunter(provider_config=config1)  # Use specific config
    
    # Show status
    status = hunter.ai_manager.get_status()
    print("CONFIGURED AI PROVIDERS:")
    for provider, info in status['providers'].items():
        print(f"  {provider}: {'✓' if info['available'] else '✗'} {info['name']}")
    print("")
    print("TASK ASSIGNMENTS:")
    for task, provider in status['task_assignments'].items():
        print(f"  {task}: {provider or 'None'}")
    print("")
    
    if not any(p['available'] for p in status['providers'].values()):
        print("ERROR: No AI providers configured!")
        print("")
        print("Please set at least one API key:")
        print("  export OPENAI_API_KEY='your-openai-key'")
        print("  export ANTHROPIC_API_KEY='your-claude-key'")
        print("  export GEMINI_API_KEY='your-gemini-key'")
        return
    
    # Hunt for domains
    print("Starting domain hunt...")
    df = await hunter.hunt_domains(
        max_domains=500,
        check_availability=True,
        use_ai_generation=True,
        ai_domains_count=100,
        focus_area="high-value AI business domains"
    )
    
    # Get top domains
    results = hunter.get_top_domains(df, top_n=100)
    
    # Generate and print report
    report = hunter.generate_report(results)
    print(report)
    
    # Export to CSV
    hunter.export_results(df, 'multi_ai_domain_results.csv')
    
    # Save report to file
    with open('multi_ai_domain_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n✓ Results saved to 'multi_ai_domain_results.csv'")
    print(f"✓ Report saved to 'multi_ai_domain_report.txt'")
    
    return df, results

if __name__ == "__main__":
    # Set your API keys as environment variables:
    # export OPENAI_API_KEY='sk-...'
    # export ANTHROPIC_API_KEY='sk-ant-...'
    # export GEMINI_API_KEY='AI...'
    
    asyncio.run(main())