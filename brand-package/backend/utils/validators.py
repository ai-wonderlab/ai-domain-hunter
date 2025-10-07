"""
Input Validation Utilities
"""
import re
from typing import List, Optional, Dict, Any
from email_validator import validate_email, EmailNotValidError


def validate_business_name(name: str) -> bool:
    """Validate business name
    
    Args:
        name: Business name
        
    Returns:
        True if valid
    """
    if not name or len(name) < 2:
        return False
    
    if len(name) > 50:
        return False
    
    # Allow letters, numbers, spaces, and common symbols
    pattern = r'^[a-zA-Z0-9\s\-\.&\']+$'
    return bool(re.match(pattern, name))


def validate_domain(domain: str) -> bool:
    """Validate domain format
    
    Args:
        domain: Domain name
        
    Returns:
        True if valid format
    """
    # Basic domain validation
    pattern = r'^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, domain.lower()))


def validate_hex_color(color: str) -> bool:
    """Validate hex color code
    
    Args:
        color: Hex color code
        
    Returns:
        True if valid
    """
    pattern = r'^#?[A-Fa-f0-9]{6}$'
    return bool(re.match(pattern, color))


def validate_email_address(email: str) -> bool:
    """Validate email address
    
    Args:
        email: Email address
        
    Returns:
        True if valid
    """
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
    
    # Trim whitespace
    text = text.strip()
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    
    return text


def validate_industry(industry: str) -> bool:
    """Validate industry selection
    
    Args:
        industry: Industry name
        
    Returns:
        True if valid
    """
    valid_industries = {
        'technology', 'healthcare', 'finance', 'education',
        'retail', 'food', 'entertainment', 'travel',
        'real_estate', 'automotive', 'fashion', 'sports',
        'other'
    }
    
    return industry.lower() in valid_industries

def validate_url(url: str) -> bool:
    """Validate URL format
    
    Args:
        url: URL string
        
    Returns:
        True if valid URL
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None


def validate_file_size(size_bytes: int, max_mb: int = 10) -> bool:
    """Validate file size is within limit
    
    Args:
        size_bytes: File size in bytes
        max_mb: Maximum size in MB
        
    Returns:
        True if within limit
    """
    max_bytes = max_mb * 1024 * 1024
    return 0 < size_bytes <= max_bytes


def validate_image_type(filename: str) -> bool:
    """Validate image file extension
    
    Args:
        filename: Image filename
        
    Returns:
        True if valid image type
    """
    import os
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'}
    ext = os.path.splitext(filename.lower())[1]
    return ext in allowed_extensions


def validate_file_format(filename: str, allowed_formats: List[str] = None) -> bool:
    """Validate file format
    
    Args:
        filename: Filename to validate
        allowed_formats: List of allowed extensions (e.g., ['.pdf', '.docx'])
        
    Returns:
        True if valid format
    """
    import os
    if allowed_formats is None:
        allowed_formats = ['.pdf', '.doc', '.docx', '.txt', '.jpg', '.jpeg', '.png']
    
    ext = os.path.splitext(filename.lower())[1]
    return ext in [fmt.lower() for fmt in allowed_formats]


def validate_username(username: str) -> bool:
    """Validate username format
    
    Args:
        username: Username to validate
        
    Returns:
        True if valid
    """
    if not username or len(username) < 3 or len(username) > 30:
        return False
    
    # Alphanumeric, underscore, hyphen only
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, username))


def validate_password(password: str) -> bool:
    """Validate password strength
    
    Args:
        password: Password to validate
        
    Returns:
        True if meets requirements
    """
    if not password or len(password) < 8:
        return False
    
    # At least one uppercase, one lowercase, one digit
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    
    return has_upper and has_lower and has_digit


def validate_uuid(uuid_string: str) -> bool:
    """Validate UUID format
    
    Args:
        uuid_string: UUID string to validate
        
    Returns:
        True if valid UUID
    """
    import uuid
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, AttributeError):
        return False


class ValidationError(Exception):
    """Custom validation error"""
    pass