"""
Text Parsing Utilities
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> Optional[Any]:
    """Extract JSON from text that may contain other content
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Parsed JSON or None
    """
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in markdown code blocks
    json_block_pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects or arrays
    patterns = [
        r'(\{[^{}]*\{[^{}]*\}[^{}]*\})',  # Nested objects
        r'(\{[^{}]*\})',  # Simple objects
        r'(\[[^\[\]]*\])',  # Arrays
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    return None


def parse_list_from_text(text: str) -> List[str]:
    """Parse list items from text
    
    Args:
        text: Text containing list items
        
    Returns:
        List of items
    """
    items = []
    
    # Try numbered list (1. Item, 2. Item)
    numbered_pattern = r'^\d+\.\s+(.+)$'
    for line in text.split('\n'):
        match = re.match(numbered_pattern, line.strip())
        if match:
            items.append(match.group(1).strip())
    
    # If no numbered items, try bullet points
    if not items:
        bullet_patterns = [
            r'^[-*•]\s+(.+)$',  # Bullet points
            r'^>\s+(.+)$',  # Quoted items
        ]
        
        for pattern in bullet_patterns:
            for line in text.split('\n'):
                match = re.match(pattern, line.strip())
                if match:
                    items.append(match.group(1).strip())
            
            if items:
                break
    
    # If still no items, split by common delimiters
    if not items:
        # Try comma separation
        if ',' in text:
            items = [item.strip() for item in text.split(',') if item.strip()]
        # Try semicolon separation
        elif ';' in text:
            items = [item.strip() for item in text.split(';') if item.strip()]
        # Try newline separation
        else:
            items = [line.strip() for line in text.split('\n') if line.strip()]
    
    return items


def extract_key_value_pairs(text: str) -> Dict[str, str]:
    """Extract key-value pairs from text
    
    Args:
        text: Text containing key-value pairs
        
    Returns:
        Dictionary of key-value pairs
    """
    pairs = {}
    
    # Patterns to try
    patterns = [
        r'^([^:]+):\s*(.+)$',  # Key: Value
        r'^([^=]+)=\s*(.+)$',  # Key = Value
        r'^([^-]+)-\s*(.+)$',  # Key - Value
    ]
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                pairs[key] = value
                break
    
    return pairs


def clean_ai_response(text: str) -> str:
    """Clean AI-generated text response
    
    Args:
        text: Raw AI response
        
    Returns:
        Cleaned text
    """
    # Remove common AI artifacts
    patterns_to_remove = [
        r'As an AI.*?,\s*',
        r'I understand.*?\.\s*',
        r'Here\'s.*?:\s*',
        r'Here are.*?:\s*',
        r'^\s*\*\s*',  # Remove asterisks
        r'\s*\*\s*$',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text
    
    Args:
        text: Text containing URLs
        
    Returns:
        List of URLs
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    
    # Clean URLs
    cleaned_urls = []
    for url in urls:
        # Remove trailing punctuation
        url = re.sub(r'[.,;:!?]+$', '', url)
        cleaned_urls.append(url)
    
    return list(set(cleaned_urls))


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Extract code blocks from markdown text
    
    Args:
        text: Markdown text with code blocks
        
    Returns:
        List of (language, code) tuples
    """
    code_blocks = []
    
    # Pattern for fenced code blocks
    pattern = r'```(\w*)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for lang, code in matches:
        lang = lang.strip() or 'text'
        code = code.strip()
        code_blocks.append((lang, code))
    
    return code_blocks


def split_into_chunks(
    text: str,
    max_length: int = 1000,
    separator: str = '\n'
) -> List[str]:
    """Split text into chunks of maximum length
    
    Args:
        text: Text to split
        max_length: Maximum chunk length
        separator: Preferred separator
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Try to split by separator
    parts = text.split(separator)
    
    for part in parts:
        if len(current_chunk) + len(part) + len(separator) <= max_length:
            if current_chunk:
                current_chunk += separator
            current_chunk += part
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # If part itself is too long, split it
            if len(part) > max_length:
                words = part.split()
                current_chunk = ""
                
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_length:
                        if current_chunk:
                            current_chunk += " "
                        current_chunk += word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = word
            else:
                current_chunk = part
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text
    
    Args:
        text: Text with irregular whitespace
        
    Returns:
        Normalized text
    """
    # Replace various whitespace characters with regular space
    text = re.sub(r'[\t\r\f\v]', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()