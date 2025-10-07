"""
Unified AI Manager for text and image generation - Agnostic version
"""
import asyncio
import httpx
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from openai import AsyncOpenAI
import logging
from functools import lru_cache

from config.settings import settings
from core.exceptions import (
    AIGenerationError,
    ImageGenerationError,
    MissingAPIKeyError,
    ServiceUnavailableError
)

logger = logging.getLogger(__name__)


class AIManager:
    """
    Singleton AI Manager handling all AI operations
    Agnostic - works with any API configured in settings
    """
    
    _instance = None
    
    def __init__(self):
        """Initialize AI clients based on configuration"""
        
        # Text generation client
        self.text_client = None
        self.text_models = {}
        self._init_text_client()
        
        # Image generation clients
        self.image_clients = []
        self._init_image_clients()
        
        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
        logger.info("✅ AI Manager initialized")
    
    def _init_text_client(self):
        """Initialize text generation client"""
        if not settings.text_api_key:
            logger.warning("⚠️ No text API key configured")
            return
        
        # Support for OpenRouter/OpenAI compatible APIs
        if settings.text_api_provider in ["openrouter", "openai", "together"]:
            self.text_client = AsyncOpenAI(
                base_url=settings.text_api_url,
                api_key=settings.text_api_key
            )
        else:
            # For custom APIs, use httpx
            self.text_client = httpx.AsyncClient(
                base_url=settings.text_api_url,
                headers={"Authorization": f"Bearer {settings.text_api_key}"}
            )
        
        # Load configured models
        self.text_models = settings.get_text_models()
        logger.info(f"✅ Text client initialized with {len(self.text_models)} models")
    
    def _init_image_clients(self):
        """Initialize image generation clients"""
        image_apis = settings.get_image_apis()
        
        if not image_apis:
            logger.warning("⚠️ No image APIs configured")
            return
        
        for api_config in image_apis:
            try:
                client = self._create_image_client(api_config)
                self.image_clients.append({
                    "provider": api_config["provider"],
                    "client": client,
                    "config": api_config,
                    "failures": 0,
                    "successes": 0
                })
                logger.info(f"✅ Image client initialized: {api_config['provider']}")
            except Exception as e:
                logger.error(f"❌ Failed to init {api_config['provider']}: {e}")
    
    def _create_image_client(self, api_config: Dict) -> Any:
        """Create image generation client based on provider"""
        provider = api_config["provider"].lower()
        
        # OpenAI-compatible APIs
        if provider in ["openrouter", "openai", "together"]:
            return AsyncOpenAI(
                base_url=api_config["url"],
                api_key=api_config["key"]
            )
        
        # Custom HTTP client for generic APIs - NO base_url
        # Full URL will be provided in each request
        return httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_config['key']}"},
            timeout=60.0
        )
    
    @classmethod
    def initialize(cls):
        """Initialize singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            raise RuntimeError("AIManager not initialized")
        return cls._instance
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False
    ) -> str:
        """
        Generate text using configured models
        
        Args:
            prompt: The prompt to generate from
            model: Model identifier or None for default
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: Whether to enforce JSON output
        
        Returns:
            Generated text
        """
        if not self.text_client:
            raise MissingAPIKeyError("text_generation")
        
        # Select model
        if not model:
            # Use first configured model as default
            model = list(self.text_models.values())[0] if self.text_models else None
        elif model in self.text_models:
            model = self.text_models[model]
        
        if not model:
            raise AIGenerationError("No text model available")
        
        try:
            logger.debug(f"Generating text with {model[:20]}...")
            
            # OpenAI-compatible API
            if isinstance(self.text_client, AsyncOpenAI):
                messages = [{"role": "user", "content": prompt}]
                
                # Add JSON mode if requested
                kwargs = {}
                if json_mode and settings.text_api_provider == "openai":
                    kwargs["response_format"] = {"type": "json_object"}
                
                response = await self.text_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                content = response.choices[0].message.content
                
                # Track usage
                if hasattr(response, 'usage'):
                    self.total_tokens += response.usage.total_tokens
                    self.request_count += 1
                
                return content
            
            # Custom HTTP API
            else:
                response = await self.text_client.post(
                    "/completions",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data.get("text", "")
                
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise AIGenerationError(f"Text generation failed: {str(e)}", model=model)
    
    async def generate_parallel(
        self,
        prompts: List[Tuple[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, str]:
        """
        Generate from multiple prompts in parallel
        
        Args:
            prompts: List of (prompt, model_key) tuples
            temperature: Sampling temperature
            max_tokens: Maximum tokens per generation
        
        Returns:
            Dictionary mapping prompt indices to results
        """
        tasks = []
        for prompt, model_key in prompts:
            task = self.generate_text(prompt, model_key, temperature, max_tokens)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for i, result in enumerate(results):
            if isinstance(result, str):
                output[f"result_{i}"] = result
            else:
                logger.error(f"Parallel generation {i} failed: {result}")
                output[f"result_{i}"] = ""
        
        return output
    
    async def generate_image(
        self,
        prompt: str,
        style: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        provider_index: Optional[int] = None
    ) -> bytes:
        """
        Generate image using configured APIs with automatic failover
        
        Args:
            prompt: Image description
            style: Style modifier
            negative_prompt: What to avoid
            width: Image width
            height: Image height
            provider_index: Force specific provider (0-based)
        
        Returns:
            Image bytes
        """
        if not self.image_clients:
            raise ImageGenerationError("No image generation APIs configured")
        
        # Build full prompt
        full_prompt = prompt
        if style:
            full_prompt = f"{style} style: {prompt}"
        
        # Try each provider in order
        clients_to_try = self.image_clients.copy()
        
        # If specific provider requested, try it first
        if provider_index is not None and 0 <= provider_index < len(clients_to_try):
            clients_to_try = [clients_to_try[provider_index]] + \
                           [c for i, c in enumerate(clients_to_try) if i != provider_index]
        
        errors = []
        
        for client_info in clients_to_try:
            provider = client_info["provider"]
            client = client_info["client"]
            config = client_info["config"]
            
            try:
                logger.info(f"🎨 Trying image generation with {provider}")
                
                # Provider-specific generation
                if provider.lower() in ["openrouter", "openai"]:
                    image_bytes = await self._generate_openai_image(
                        client, config, full_prompt, width, height
                    )
                elif provider.lower() == "replicate":
                    image_bytes = await self._generate_replicate_image(
                        client, config, full_prompt, negative_prompt, width, height
                    )
                elif provider.lower() == "together":
                    image_bytes = await self._generate_together_image(
                        client, config, full_prompt, negative_prompt, width, height
                    )
                else:
                    # Generic HTTP API
                    image_bytes = await self._generate_generic_image(
                        client, config, full_prompt, negative_prompt, width, height
                    )
                
                # Success!
                client_info["successes"] += 1
                logger.info(f"✅ Image generated successfully with {provider}")
                return image_bytes
                
            except Exception as e:
                client_info["failures"] += 1
                error_msg = f"{provider}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ {provider} failed: {e}")
                continue
        
        # All providers failed
        raise ImageGenerationError(
            f"All image providers failed. Errors: {'; '.join(errors)}"
        )
    
    async def _generate_openai_image(
        self, client: AsyncOpenAI, config: Dict,
        prompt: str, width: int, height: int
    ) -> bytes:
        """Generate image using OpenAI/OpenRouter API"""
        
        model = config.get("model", "dall-e-3")
        size = f"{width}x{height}"
        
        # DALL-E has specific size requirements
        if "dall-e" in model:
            if width == height:
                size = "1024x1024" if width >= 1024 else "512x512"
            else:
                size = "1792x1024" if width > height else "1024x1792"
        
        response = await client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=size,
            response_format="b64_json"
        )
        
        # Decode base64 image
        import base64
        image_b64 = response.data[0].b64_json
        return base64.b64decode(image_b64)
    
    async def _generate_replicate_image(
        self, client: httpx.AsyncClient, config: Dict,
        prompt: str, negative_prompt: Optional[str],
        width: int, height: int
    ) -> bytes:
        """Generate image using Replicate API"""
        
        model = config.get("model", "stability-ai/sdxl")
        
        # Create prediction
        response = await client.post(
            "/predictions",
            json={
                "version": model,
                "input": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt or "",
                    "width": width,
                    "height": height,
                    "num_outputs": 1
                }
            }
        )
        response.raise_for_status()
        prediction = response.json()
        
        # Poll for completion
        prediction_id = prediction["id"]
        while True:
            await asyncio.sleep(2)
            
            check_response = await client.get(f"/predictions/{prediction_id}")
            check_response.raise_for_status()
            result = check_response.json()
            
            if result["status"] == "succeeded":
                image_url = result["output"][0]
                break
            elif result["status"] == "failed":
                raise ImageGenerationError(f"Replicate generation failed: {result.get('error')}")
        
        # Download image
        image_response = await client.get(image_url)
        return image_response.content
    
    async def _generate_together_image(
        self, client: Any, config: Dict,
        prompt: str, negative_prompt: Optional[str],
        width: int, height: int
    ) -> bytes:
        """Generate image using Together API"""
        
        model = config.get("model", "black-forest-labs/FLUX.1-schnell")
        
        # Together uses OpenAI-compatible API
        if isinstance(client, AsyncOpenAI):
            # Use their image endpoint
            response = await client.images.generate(
                model=model,
                prompt=prompt,
                n=1,
                size=f"{width}x{height}"
            )
            
            # Get image URL and download
            image_url = response.data[0].url
            async with httpx.AsyncClient() as http_client:
                image_response = await http_client.get(image_url)
                return image_response.content
        
        # Fallback to HTTP API
        response = await client.post(
            "/images/generations",
            json={
                "model": model,
                "prompt": prompt,
                "n": 1,
                "width": width,
                "height": height
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Download image from URL
        image_url = data["data"][0]["url"]
        async with httpx.AsyncClient() as http_client:
            image_response = await http_client.get(image_url)
            return image_response.content
    
    async def _generate_generic_image(
        self, client: httpx.AsyncClient, config: Dict,
        prompt: str, negative_prompt: Optional[str],
        width: int, height: int
    ) -> bytes:
        """Generate image using generic HTTP API with polling support"""
        
        # Initial request
        response = await client.post(
            config['url'],
            json={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "model": config.get("model", "default")
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Check if this is an async API that needs polling
        if 'data' in data and 'urls' in data.get('data', {}):
            # Wavespeed-style async API
            result_url = data['data']['urls'].get('get')
            if result_url:
                return await self._poll_async_image(client, result_url)
        
        # Handle different synchronous response formats
        if "image" in data:
            import base64
            return base64.b64decode(data["image"])
        elif "url" in data:
            async with httpx.AsyncClient() as http_client:
                image_response = await http_client.get(data["url"])
                return image_response.content
        elif "data" in data:
            return await self._handle_nested_response(data["data"])
        else:
            raise ImageGenerationError(f"Unknown response format from {config['provider']}")

    async def _poll_async_image(self, client: httpx.AsyncClient, result_url: str, max_attempts: int = 60) -> bytes:
        """Poll async image generation API until completion"""
        import asyncio
        
        for attempt in range(max_attempts):
            await asyncio.sleep(2)  # Wait 2 seconds between polls
            
            response = await client.get(result_url)
            response.raise_for_status()
            result = response.json()
            
            # Wavespeed format
            if 'data' in result:
                status = result['data'].get('status')
                
                if status in ['completed', 'succeeded']:
                    # Get image URL from outputs
                    outputs = result['data'].get('outputs', [])
                    if outputs and len(outputs) > 0:
                        image_url = outputs[0]
                        # Download the image
                        async with httpx.AsyncClient() as http_client:
                            img_response = await http_client.get(image_url)
                            return img_response.content
                            
                elif status == 'failed':
                    error = result['data'].get('error', 'Unknown error')
                    raise ImageGenerationError(f"Generation failed: {error}")
        
        raise ImageGenerationError(f"Image generation timed out after {max_attempts * 2} seconds")
    
    async def _handle_nested_response(self, data: Any) -> bytes:
        """Handle nested API responses"""
        if isinstance(data, list) and len(data) > 0:
            item = data[0]
            if isinstance(item, dict):
                if "b64_json" in item:
                    import base64
                    return base64.b64decode(item["b64_json"])
                elif "url" in item:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(item["url"])
                        return response.content
        
        raise ImageGenerationError("Could not parse image response")
    
    def get_stats(self) -> Dict:
        """Get AI Manager statistics"""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.total_cost,
            "text_models": len(self.text_models),
            "image_providers": len(self.image_clients),
            "image_stats": [
                {
                    "provider": c["provider"],
                    "successes": c["successes"],
                    "failures": c["failures"]
                }
                for c in self.image_clients
            ]
        }
    
    @classmethod
    async def cleanup(cls):
        """Cleanup resources"""
        if cls._instance:
            # Close HTTP clients
            for client_info in cls._instance.image_clients:
                client = client_info["client"]
                if isinstance(client, httpx.AsyncClient):
                    await client.aclose()
            
            if isinstance(cls._instance.text_client, httpx.AsyncClient):
                await cls._instance.text_client.aclose()
            
            logger.info("✅ AI Manager cleaned up")
            cls._instance = None