"""
Image Processing Utilities
"""
from PIL import Image, ImageDraw, ImageFont, ImageOps
from io import BytesIO
import base64
import logging
from typing import Tuple, Optional, Dict, Any
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


def process_logo_image(
    image_bytes: bytes,
    target_size: Tuple[int, int] = (512, 512),
    output_format: str = "PNG"
) -> bytes:
    """Process logo image for consistency
    
    Args:
        image_bytes: Raw image bytes
        target_size: Target dimensions
        output_format: Output format (PNG, JPEG)
        
    Returns:
        Processed image bytes
    """
    try:
        # Open image
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to RGBA for transparency support
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Resize maintaining aspect ratio
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create new image with white background
        if output_format == "PNG":
            # Keep transparency for PNG
            new_img = Image.new('RGBA', target_size, (255, 255, 255, 0))
        else:
            # White background for JPEG
            new_img = Image.new('RGB', target_size, (255, 255, 255))
            
        # Paste resized image centered
        x = (target_size[0] - img.width) // 2
        y = (target_size[1] - img.height) // 2
        
        if output_format == "PNG":
            new_img.paste(img, (x, y), img if img.mode == 'RGBA' else None)
        else:
            # For JPEG, composite over white background
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                new_img.paste(background, (x, y))
            else:
                new_img.paste(img, (x, y))
        
        # Save to bytes
        output = BytesIO()
        new_img.save(output, format=output_format, optimize=True, quality=95)
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return image_bytes


def generate_logo_variations(
    image_bytes: bytes
) -> Dict[str, bytes]:
    """Generate multiple variations of a logo
    
    Args:
        image_bytes: Original image bytes
        
    Returns:
        Dictionary of variation name to image bytes
    """
    variations = {}
    
    try:
        # Open original image
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Original
        variations['original'] = image_bytes
        
        # Square version (512x512)
        square = img.copy()
        square.thumbnail((512, 512), Image.Resampling.LANCZOS)
        square_output = BytesIO()
        square.save(square_output, format='PNG', optimize=True)
        variations['square'] = square_output.getvalue()
        
        # Icon version (128x128)
        icon = img.copy()
        icon.thumbnail((128, 128), Image.Resampling.LANCZOS)
        icon_output = BytesIO()
        icon.save(icon_output, format='PNG', optimize=True)
        variations['icon'] = icon_output.getvalue()
        
        # Favicon version (32x32)
        favicon = img.copy()
        favicon.thumbnail((32, 32), Image.Resampling.LANCZOS)
        favicon_output = BytesIO()
        favicon.save(favicon_output, format='ICO')
        variations['favicon'] = favicon_output.getvalue()
        
        # Monochrome version
        mono = img.convert('L').convert('RGBA')
        mono_output = BytesIO()
        mono.save(mono_output, format='PNG', optimize=True)
        variations['monochrome'] = mono_output.getvalue()
        
        # Inverted version
        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            rgb_img = Image.merge('RGB', (r, g, b))
            inverted = ImageOps.invert(rgb_img)
            inverted = inverted.convert('RGBA')
            inverted.putalpha(a)
        else:
            inverted = ImageOps.invert(img)
        
        inverted_output = BytesIO()
        inverted.save(inverted_output, format='PNG', optimize=True)
        variations['inverted'] = inverted_output.getvalue()
        
        return variations
        
    except Exception as e:
        logger.error(f"Failed to generate variations: {e}")
        return {'original': image_bytes}


def add_watermark(
    image_bytes: bytes,
    watermark_text: str = "DRAFT",
    position: str = "center",
    opacity: int = 128
) -> bytes:
    """Add watermark to image
    
    Args:
        image_bytes: Original image bytes
        watermark_text: Text to add as watermark
        position: Position (center, bottom-right, etc.)
        opacity: Watermark opacity (0-255)
        
    Returns:
        Watermarked image bytes
    """
    try:
        # Open image
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Create watermark layer
        watermark = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Calculate font size (5% of image width)
        font_size = max(20, img.width // 20)
        
        # Try to use a nice font, fallback to default
        try:
            font = ImageFont.truetype("Arial", font_size)
        except:
            font = ImageFont.load_default()
        
        # Get text size
        bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position
        if position == "center":
            x = (img.width - text_width) // 2
            y = (img.height - text_height) // 2
        elif position == "bottom-right":
            x = img.width - text_width - 20
            y = img.height - text_height - 20
        else:
            x, y = 20, 20
        
        # Draw text with transparency
        draw.text(
            (x, y),
            watermark_text,
            fill=(255, 255, 255, opacity),
            font=font
        )
        
        # Composite watermark over image
        watermarked = Image.alpha_composite(img, watermark)
        
        # Save to bytes
        output = BytesIO()
        watermarked.save(output, format='PNG', optimize=True)
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to add watermark: {e}")
        return image_bytes


def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string
    
    Args:
        image_bytes: Image bytes
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode('utf-8')


def base64_to_image(base64_string: str) -> bytes:
    """Convert base64 string to image bytes
    
    Args:
        base64_string: Base64 encoded string
        
    Returns:
        Image bytes
    """
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    return base64.b64decode(base64_string)


def get_image_hash(image_bytes: bytes) -> str:
    """Get hash of image for deduplication
    
    Args:
        image_bytes: Image bytes
        
    Returns:
        SHA256 hash string
    """
    return hashlib.sha256(image_bytes).hexdigest()


def optimize_image(
    image_bytes: bytes,
    max_size: int = 1024 * 1024,  # 1MB
    quality: int = 85
) -> bytes:
    """Optimize image size while maintaining quality
    
    Args:
        image_bytes: Original image bytes
        max_size: Maximum file size in bytes
        quality: JPEG quality (1-100)
        
    Returns:
        Optimized image bytes
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        
        # If already small enough, return as is
        if len(image_bytes) <= max_size:
            return image_bytes
        
        # Try progressively lower quality
        for q in range(quality, 10, -10):
            output = BytesIO()
            
            # Save with current quality
            if img.mode == 'RGBA':
                # Convert RGBA to RGB for JPEG
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                rgb_img.save(output, format='JPEG', optimize=True, quality=q)
            else:
                img.save(output, format='JPEG', optimize=True, quality=q)
            
            result = output.getvalue()
            
            # Check if size is acceptable
            if len(result) <= max_size:
                return result
        
        # If still too large, resize
        while len(result) > max_size and img.width > 100:
            # Reduce dimensions by 10%
            new_size = (int(img.width * 0.9), int(img.height * 0.9))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            output = BytesIO()
            img.save(output, format='JPEG', optimize=True, quality=60)
            result = output.getvalue()
        
        return result
        
    except Exception as e:
        logger.error(f"Image optimization failed: {e}")
        return image_bytes