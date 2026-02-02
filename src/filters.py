import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
import random

def apply_polaroid_effect(image: Image.Image, film_type: str = 'polaroid') -> Image.Image:
    """
    Apply a comprehensive Polaroid aesthetic effect to the image.
    Includes color grading, vignetting, and film grain.
    """
    # 0. Auto-Level (Pre-processing)
    # Cutoff=2 to ensure we clip highlights (overexposure) as requested
    img = _auto_level(image, cutoff=2)

    # 1. Softness / Optical Bloom (Counteract digital sharpness)
    img = _apply_softness_bloom(img)

    # 2. Add Stop Bath / Inward Shadow Border
    img = _add_stop_bath_border(img, sides='all')

    # 3. Color Grading & Tinting
    # User requested to use Polaroid vintage curves for Instax too
    img = _apply_vintage_curves(img)
    
    # 4. Lift Blacks (Fade)
    # Make sure we don't have pure black, but dark gray.
    img = _lift_blacks(img, lift_amount=25)

    # 5. Add Vignette
    vignette_intensity = 0.4 if 'polaroid' in film_type else 0.25
    img = _add_vignette(img, intensity=vignette_intensity)
    
    # 6. Add Film Grain
    img = _add_film_grain(img)
    
    # 7. Final slight desaturation
    sat_amount = 0.85 if 'polaroid' in film_type else 0.95
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(sat_amount)
    
    return img

def _lift_blacks(image: Image.Image, lift_amount: int = 25) -> Image.Image:
    """
    Lift the darkest blacks to a dark gray to simulate low dynamic range of print.
    input 0 -> lift_amount
    input 255 -> 255
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Create a lookup table
    # Linear interpolation from lift_amount to 255
    lut = []
    for i in range(256):
        # val = lift_amount + (i / 255.0) * (255 - lift_amount)
        # However, usually fades are non-linear or just a simple offset?
        # A simple linear map is easiest and effective.
        val = int(lift_amount + (i / 255.0) * (255 - lift_amount))
        lut.append(val)
        
    # Apply to all channels (RGB) same way? Or per channel?
    # Usually consistent fade.
    return image.point(lut * 3)

def _apply_softness_bloom(image: Image.Image) -> Image.Image:
    """
    Simulate the optical softness of instant film.
    1. Global very subtle Gaussian blur to kill digital sharpness.
    2. Optional: Highlight bloom (not strictly necessary if blur is good, keeping simple)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    width, height = image.size
    
    # 1. Global Diffusion
    # Blur radius proportional to size. 0.1% of min dimension?
    # At 1000px, that's 1px radius. 
    blur_radius = max(0.5, min(width, height) * 0.001)
    img_soft = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # 2. Subtle Bloom (Screen a blurred version of highlights)
    # This is expensive. Let's stick to just the optical diffusion first, 
    # as high-res output often looks too sharp purely due to pixel perfection.
    # The user said "image looks a bit too sharp". 
    
    return img_soft

def _auto_level(image: Image.Image) -> Image.Image:
    """
    Auto-level the image to fix exposure/contrast before applying effects.
    Uses generic autocontrast with a small cutoff to ignore outliers.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return ImageOps.autocontrast(image, cutoff=1)

def _add_stop_bath_border(image: Image.Image, strength: float = 0.5, sides: str = 'all') -> Image.Image:
    """
    Add an artificial inward shadow/burn from the edges.
    Implements a compound shadow:
    1. Sharp, strong layer (chemical edge) - Adjusted: Lower opacity, slightly more blur
    2. Smooth, subtle layer (vignette/depth) - Adjusted: Higher opacity
    """
    width, height = image.size
    
    # --- Layer 1: Sharp Shadow ---
    mask_sharp = Image.new('L', (width, height), 0)
    draw_sharp = ImageDraw.Draw(mask_sharp)
    
    # Very thin margin (User requested 2-4px)
    # On ~1000px img, 0.003 is 3px.
    margin_sharp = min(width, height) * 0.003
    draw_sharp.rectangle([margin_sharp, margin_sharp, width-margin_sharp, height-margin_sharp], fill=255)
    
    # Minimal blur for sharpness (matches margin roughly)
    blur_sharp = min(width, height) * 0.003
    mask_sharp = mask_sharp.filter(ImageFilter.GaussianBlur(radius=blur_sharp))
    mask_sharp_inv = ImageOps.invert(mask_sharp)
    
    # --- Layer 2: Smooth Shadow ---
    mask_smooth = Image.new('L', (width, height), 0)
    draw_smooth = ImageDraw.Draw(mask_smooth)
    
    # Margin: 2% (same)
    margin_smooth = min(width, height) * 0.02
    draw_smooth.rectangle([margin_smooth, margin_smooth, width-margin_smooth, height-margin_smooth], fill=255)
    
    # Blur: 3% (same)
    blur_smooth = min(width, height) * 0.03
    mask_smooth = mask_smooth.filter(ImageFilter.GaussianBlur(radius=blur_smooth))
    mask_smooth_inv = ImageOps.invert(mask_smooth)
    
    # --- Combine Masks ---
    # User request: 
    # "increase opacity of [smooth] by 10%" -> ~40% (was 30%)
    # "decrease opacity of sharp shadow to around 30%" -> ~30% (was 55%)
    # Latest: "soft shadow should only be 25% opacity"
    
    sharp_opacity = 0.30
    smooth_opacity = 0.25
    
    # Scale masks by their opacity and strength factor
    mask_sharp_arr = np.array(mask_sharp_inv).astype(float) * sharp_opacity
    mask_smooth_arr = np.array(mask_smooth_inv).astype(float) * smooth_opacity
    
    # Combine
    total_mask_arr = mask_sharp_arr + mask_smooth_arr
    total_mask_arr = np.clip(total_mask_arr, 0, 255) 
    
    final_mask = Image.fromarray(total_mask_arr.astype(np.uint8))
    
    # Shadow layer (Dark Warm)
    shadow_layer = Image.new('RGB', (width, height), (15, 10, 10)) 
    
    return Image.composite(shadow_layer, image, final_mask)

def _apply_instax_curves(image: Image.Image) -> Image.Image:
    """
    Apply Instax-specific color grading.
    Instax is often cooler, higher contrast, with punchy blacks compared to Polaroid's creamy vintage.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    r, g, b = image.split()
    
    # Instax: High contrast, slightly cool whites, deep blacks
    
    # RGB master curve (contrast S-curve) applied to all? 
    # Let's do components.
    
    # Red: Slight S-curve, but kept relatively restrained
    r = r.point(lambda i: i * 0.95 if i > 200 else (i*1.05 if i > 50 else i))
    
    # Green: Pretty standard
    g = g.point(lambda i: i) 
    
    # Blue: Boost in shadows (cool blacks) and slight boost in midtones
    b = b.point(lambda i: i * 1.05 + 5)
    
    return Image.merge('RGB', (r, g, b))


def _auto_level(image: Image.Image, cutoff: int = 1) -> Image.Image:
    """
    Auto-level the image to fix exposure/contrast before applying effects.
    Uses generic autocontrast with a small cutoff to ignore outliers.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return ImageOps.autocontrast(image, cutoff=cutoff)

def _apply_vintage_curves(image: Image.Image) -> Image.Image:
    """
    Apply channel-specific curve adjustments to mimic instant film.
    """
    # Split channels
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    r, g, b = image.split()
    
    # Custom curve mappings using point operations
    # Lift blacks and flatten whites somewhat, push reds/yellows in highlights
    
    # Red: slight boost in mid-highs
    r = r.point(lambda i: i * 1.05 if i < 100 else i + (255 - i) * 0.1)
    
    # Green: relatively neutral, maybe slight lift in shadows
    g = g.point(lambda i: i * 0.98 + 5)
    
    # Blue: cut in shadows (yellowing) and lift in highlights (cooling/fading)
    # This creates the classic blue/yellow split toning
    b = b.point(lambda i: i * 0.85 + 20)
    
    return Image.merge('RGB', (r, g, b))

def _add_vignette(image: Image.Image, intensity: float = 0.4) -> Image.Image:
    """
    Add a subtle darkened vignette to the corners.
    """
    width, height = image.size
    
    # Create a radial gradient mask
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distance from center, normalized
    radius = np.sqrt(X**2 + Y**2)
    
    # Create vignette mask: 1 at center, drops off towards corners
    # We want corners to be darker, so we multiply image by (1 - intensity * radius)
    # But ensuring we don't go below 0
    mask = 1 - np.clip(radius - 0.5, 0, 1) * intensity
    mask = np.clip(mask, 0, 1)
    
    # Convert image to numpy
    img_arr = np.array(image).astype(float)
    
    # Apply mask to all channels
    for c in range(3):
        img_arr[:, :, c] *= mask
        
    return Image.fromarray(np.uint8(img_arr))

def _add_film_grain(image: Image.Image, amount: float = 0.03) -> Image.Image:
    """
    Add random gaussian noise to simulate film grain.
    Adjusted to 0.03 (low/half of medium) based on feedback.
    """
    img_arr = np.array(image).astype(float)
    
    # Generate noise centered at 0
    noise = np.random.normal(0, 255 * amount, img_arr.shape)
    
    # Add noise
    noisy_img = img_arr + noise
    
    # Clip back to valid range
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_img)
