"""
Security Utilities
"""
import hashlib
import hmac
import secrets
import string
from typing import Optional, Tuple
from passlib.context import CryptContext
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False


def generate_token(length: int = 32) -> str:
    """Generate a secure random token
    
    Args:
        length: Token length
        
    Returns:
        Random token string
    """
    return secrets.token_urlsafe(length)


def generate_api_key() -> str:
    """Generate a secure API key
    
    Returns:
        API key string
    """
    prefix = "sk" if settings.app_env == "production" else "test"
    token = secrets.token_urlsafe(32)
    return f"{prefix}_{token}"


def generate_password(length: int = 16) -> str:
    """Generate a secure random password
    
    Args:
        length: Password length
        
    Returns:
        Random password
    """
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(alphabet) for _ in range(length))
    
    # Ensure password has at least one of each type
    if not any(c.islower() for c in password):
        password = password[:-1] + secrets.choice(string.ascii_lowercase)
    if not any(c.isupper() for c in password):
        password = password[:-1] + secrets.choice(string.ascii_uppercase)
    if not any(c.isdigit() for c in password):
        password = password[:-1] + secrets.choice(string.digits)
    if not any(c in "!@#$%^&*" for c in password):
        password = password[:-1] + secrets.choice("!@#$%^&*")
    
    return password


def create_signature(data: str, secret: Optional[str] = None) -> str:
    """Create HMAC signature for data
    
    Args:
        data: Data to sign
        secret: Secret key (uses app secret if not provided)
        
    Returns:
        Signature string
    """
    if secret is None:
        secret = settings.jwt_secret_key
    
    signature = hmac.new(
        secret.encode(),
        data.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return signature


def verify_signature(
    data: str,
    signature: str,
    secret: Optional[str] = None
) -> bool:
    """Verify HMAC signature
    
    Args:
        data: Original data
        signature: Signature to verify
        secret: Secret key
        
    Returns:
        True if signature is valid
    """
    expected_signature = create_signature(data, secret)
    return hmac.compare_digest(expected_signature, signature)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = filename.replace('/', '').replace('\\', '')
    
    # Remove dangerous characters
    dangerous_chars = '<>:"|?*'
    for char in dangerous_chars:
        filename = filename.replace(char, '')
    
    # Limit length
    name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
    if len(name) > 100:
        name = name[:100]
    
    # Reconstruct
    if ext:
        filename = f"{name}.{ext}"
    else:
        filename = name
    
    # Ensure not empty
    if not filename:
        filename = "unnamed"
    
    return filename


def is_safe_url(url: str, allowed_hosts: Optional[list] = None) -> bool:
    """Check if URL is safe for redirect
    
    Args:
        url: URL to check
        allowed_hosts: List of allowed hosts
        
    Returns:
        True if URL is safe
    """
    if not url:
        return False
    
    # Reject URLs with @ (can be used for phishing)
    if '@' in url:
        return False
    
    # Parse URL
    from urllib.parse import urlparse
    
    try:
        result = urlparse(url)
    except ValueError:
        return False
    
    # Reject non-http(s) schemes
    if result.scheme and result.scheme not in ['http', 'https']:
        return False
    
    # Check host if provided
    if allowed_hosts and result.hostname:
        if result.hostname not in allowed_hosts:
            return False
    
    return True


def mask_sensitive_data(data: str, mask_char: str = '*') -> str:
    """Mask sensitive data for logging
    
    Args:
        data: Sensitive data
        mask_char: Character to use for masking
        
    Returns:
        Masked string
    """
    if not data:
        return ""
    
    if len(data) <= 4:
        return mask_char * len(data)
    
    # Show first 2 and last 2 characters
    visible_start = data[:2]
    visible_end = data[-2:]
    masked_middle = mask_char * (len(data) - 4)
    
    return f"{visible_start}{masked_middle}{visible_end}"


def rate_limit_key(user_id: str, action: str) -> str:
    """Generate rate limit key for user action
    
    Args:
        user_id: User identifier
        action: Action being rate limited
        
    Returns:
        Rate limit key
    """
    return f"rate_limit:{user_id}:{action}"