"""
Application constants and enums - Agnostic version
"""
from enum import Enum
from typing import Dict, List


class GenerationType(str, Enum):
    """Types of content that can be generated"""
    NAME = "name"
    LOGO = "logo"
    COLOR = "color"
    TAGLINE = "tagline"
    DOMAIN = "domain"
    PACKAGE = "package"


class ProjectStatus(str, Enum):
    """Status of a brand project"""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ImageFormat(str, Enum):
    """Supported image formats"""
    PNG = "png"
    JPG = "jpg"
    SVG = "svg"
    WEBP = "webp"


# Pricing constants (in USD) - Approximate costs
GENERATION_COSTS: Dict[GenerationType, float] = {
    GenerationType.NAME: 0.01,
    GenerationType.LOGO: 0.15,
    GenerationType.COLOR: 0.01,
    GenerationType.TAGLINE: 0.01,
    GenerationType.DOMAIN: 0.00,
    GenerationType.PACKAGE: 0.25
}

# Generation limits
MAX_NAMES_PER_GENERATION = 10
MAX_LOGOS_PER_GENERATION = 3
MAX_COLORS_PER_PALETTE = 6
MAX_PALETTES_PER_GENERATION = 5
MAX_TAGLINES_PER_GENERATION = 5
MAX_DOMAINS_TO_CHECK = 20

# Rate limiting
FREE_TIER_GENERATIONS = 2
PAID_TIER_GENERATIONS = 1000
RATE_LIMIT_WINDOW_HOURS = 24

# File size limits (in bytes)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_SIZE = 5 * 1024 * 1024    # 5MB

# Text length limits
MAX_DESCRIPTION_LENGTH = 2000
MAX_NAME_LENGTH = 50
MAX_TAGLINE_LENGTH = 100

# API timeouts (in seconds)
API_TIMEOUT = 30
IMAGE_GENERATION_TIMEOUT = 60
DOMAIN_CHECK_TIMEOUT = 10

# Cache TTL (in seconds)
RESEARCH_CACHE_TTL = 7 * 24 * 60 * 60  # 7 days
API_CACHE_TTL = 60 * 60                # 1 hour
USER_CACHE_TTL = 15 * 60                # 15 minutes

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Color validation
WCAG_CONTRAST_RATIO_AA = 4.5
WCAG_CONTRAST_RATIO_AAA = 7.0

# Business name patterns to avoid
BLOCKED_NAME_PATTERNS: List[str] = [
    "test",
    "demo",
    "sample",
    "example",
    "temp",
    "xxx",
    "porn",
    "drug"
]

# Generic style options
LOGO_STYLES: List[str] = [
    "minimalist",
    "modern",
    "playful",
    "elegant",
    "bold",
    "tech",
    "organic",
    "vintage"
]

COLOR_THEMES: List[str] = [
    "vibrant",
    "pastel",
    "monochrome",
    "earth",
    "ocean",
    "sunset",
    "forest",
    "corporate"
]

TAGLINE_TONES: List[str] = [
    "professional",
    "friendly",
    "bold",
    "inspirational",
    "witty",
    "serious",
    "casual",
    "technical"
]

# Error messages
ERROR_MESSAGES = {
    "rate_limit": "You've reached your generation limit. Please upgrade or wait {hours} hours.",
    "invalid_input": "Invalid input provided. Please check your data.",
    "generation_failed": "Generation failed. Please try again.",
    "auth_required": "Authentication required. Please log in.",
    "insufficient_credits": "Insufficient credits. Please upgrade your plan.",
    "service_unavailable": "Service temporarily unavailable. Please try again later.",
    "no_image_api": "No image generation API configured.",
    "no_text_api": "No text generation API configured.",
    "no_domain_api": "No domain checking API configured."
}

# Success messages
SUCCESS_MESSAGES = {
    "generation_complete": "Generation completed successfully!",
    "project_saved": "Project saved successfully.",
    "assets_downloaded": "Assets downloaded successfully."
}

# Supported API providers (για reference)
SUPPORTED_TEXT_PROVIDERS = [
    "openrouter",
    "openai",
    "anthropic",
    "google",
    "custom"
]

SUPPORTED_IMAGE_PROVIDERS = [
    "openrouter",
    "replicate",
    "together",
    "stability",
    "openai",
    "playground",
    "leonardo",
    "custom"
]

SUPPORTED_DOMAIN_PROVIDERS = [
    "whoisxml",
    "whoapi",
    "namecheap",
    "godaddy",
    "custom"
]