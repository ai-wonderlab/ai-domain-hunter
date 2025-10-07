"""
Text and Data Formatting Utilities
"""
from typing import List, Dict, Any
import re


def format_business_name(name: str) -> str:
    """Format business name for display
    
    Args:
        name: Raw business name
        
    Returns:
        Formatted name
    """
    # Title case with special handling
    words = name.split()
    formatted = []
    
    for word in words:
        if word.upper() in ['AI', 'IO', 'API', 'USA', 'UK']:
            formatted.append(word.upper())
        elif word.lower() in ['and', 'or', 'the', 'of', 'in', 'for']:
            formatted.append(word.lower())
        else:
            formatted.append(word.capitalize())
    
    return ' '.join(formatted)


def format_domain(domain: str) -> str:
    """Format domain for consistency
    
    Args:
        domain: Raw domain
        
    Returns:
        Formatted domain
    """
    domain = domain.lower().strip()
    
    # Remove protocol if present
    domain = re.sub(r'^https?://', '', domain)
    
    # Remove www
    domain = re.sub(r'^www\.', '', domain)
    
    # Remove trailing slash
    domain = domain.rstrip('/')
    
    return domain


def format_hex_color(color: str) -> str:
    """Format hex color to standard format
    
    Args:
        color: Hex color
        
    Returns:
        Formatted hex color
    """
    color = color.strip()
    
    # Add # if missing
    if not color.startswith('#'):
        color = f'#{color}'
    
    # Convert to uppercase
    return color.upper()


def format_tagline(tagline: str) -> str:
    """Format tagline for display
    
    Args:
        tagline: Raw tagline
        
    Returns:
        Formatted tagline
    """
    # Capitalize first letter
    if tagline:
        tagline = tagline[0].upper() + tagline[1:]
    
    # Ensure proper ending punctuation
    if tagline and not tagline[-1] in '.!?':
        # Don't add period if it's a question
        if '?' not in tagline:
            tagline += '.'
    
    return tagline


def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format currency amount
    
    Args:
        amount: Amount
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == 'USD':
        return f'${amount:,.2f}'
    elif currency == 'EUR':
        return f'€{amount:,.2f}'
    else:
        return f'{currency} {amount:,.2f}'


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)].rsplit(' ', 1)[0] + suffix