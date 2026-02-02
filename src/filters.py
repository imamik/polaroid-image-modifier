import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
import random

def apply_polaroid_effect(image: Image.Image, film_type: str = 'polaroid', 
                          chemical_distortion: bool = False, 
                          chemical_distortion_intensity: float = 1.0,
                          chemical_distortion_seed: int = None) -> Image.Image:
    """
    Apply a comprehensive Polaroid aesthetic effect to the image.
    Includes color grading, vignetting, and film grain.
    
    Args:
        image: Input PIL Image
        film_type: Type of film effect ('polaroid' or 'instax')
        chemical_distortion: Whether to apply chemical edge distortion effect
        chemical_distortion_intensity: Intensity of chemical distortion (0.0 to 2.0)
        chemical_distortion_seed: Random seed for reproducible distortion patterns
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

    # 5. Add Chemical Edge Distortion (optional)
    if chemical_distortion:
        img = _add_chemical_edge_distortion(img, 
                                            intensity=chemical_distortion_intensity,
                                            seed=chemical_distortion_seed)

    # 6. Add Vignette
    vignette_intensity = 0.4 if 'polaroid' in film_type else 0.25
    img = _add_vignette(img, intensity=vignette_intensity)
    
    # 7. Add Film Grain
    img = _add_film_grain(img)
    
    # 8. Final slight desaturation
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


def _generate_perlin_noise_2d(shape, scale=100, octaves=4, persistence=0.5, seed=None):
    """
    Generate 2D Perlin-like noise using numpy.
    This is a simplified implementation using interpolated random gradients.
    
    Args:
        shape: (height, width) tuple
        scale: Base scale of the noise (larger = smoother)
        octaves: Number of noise layers to combine
        persistence: How much each octave contributes (0-1)
        seed: Random seed for reproducibility
    
    Returns:
        2D numpy array with values in range [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    height, width = shape
    noise = np.zeros(shape)
    
    for octave in range(octaves):
        freq = 2 ** octave
        amp = persistence ** octave
        
        # Grid size for this octave
        grid_h = max(2, int(height / (scale / freq)))
        grid_w = max(2, int(width / (scale / freq)))
        
        # Generate random values at grid points
        grid = np.random.rand(grid_h + 1, grid_w + 1)
        
        # Create coordinate arrays for interpolation
        y_coords = np.linspace(0, grid_h, height, endpoint=False)
        x_coords = np.linspace(0, grid_w, width, endpoint=False)
        
        # Get integer and fractional parts
        y0 = y_coords.astype(int)
        x0 = x_coords.astype(int)
        y1 = np.clip(y0 + 1, 0, grid_h)
        x1 = np.clip(x0 + 1, 0, grid_w)
        
        # Fractional parts with smoothstep
        fy = y_coords - y0
        fx = x_coords - x0
        
        # Smoothstep for smoother interpolation
        fy = fy * fy * (3 - 2 * fy)
        fx = fx * fx * (3 - 2 * fx)
        
        # Bilinear interpolation
        for i in range(height):
            for j in range(width):
                v00 = grid[y0[i], x0[j]]
                v01 = grid[y0[i], x1[j]]
                v10 = grid[y1[i], x0[j]]
                v11 = grid[y1[i], x1[j]]
                
                v0 = v00 * (1 - fx[j]) + v01 * fx[j]
                v1 = v10 * (1 - fx[j]) + v11 * fx[j]
                
                noise[i, j] += (v0 * (1 - fy[i]) + v1 * fy[i]) * amp
    
    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def _generate_perlin_noise_2d_fast(shape, scale=100, octaves=4, persistence=0.5, seed=None):
    """
    Fast vectorized 2D Perlin-like noise generation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    height, width = shape
    noise = np.zeros(shape)
    
    for octave in range(octaves):
        freq = 2 ** octave
        amp = persistence ** octave
        
        grid_h = max(2, int(height / (scale / freq)))
        grid_w = max(2, int(width / (scale / freq)))
        
        grid = np.random.rand(grid_h + 1, grid_w + 1)
        
        y_coords = np.linspace(0, grid_h, height, endpoint=False)
        x_coords = np.linspace(0, grid_w, width, endpoint=False)
        
        y0 = y_coords.astype(int)
        x0 = x_coords.astype(int)
        y1 = np.clip(y0 + 1, 0, grid_h)
        x1 = np.clip(x0 + 1, 0, grid_w)
        
        fy = y_coords - y0
        fx = x_coords - x0
        
        fy = fy * fy * (3 - 2 * fy)
        fx = fx * fx * (3 - 2 * fx)
        
        # Vectorized bilinear interpolation using meshgrid indices
        Y0, X0 = np.meshgrid(y0, x0, indexing='ij')
        Y1, X1 = np.meshgrid(y1, x1, indexing='ij')
        FY, FX = np.meshgrid(fy, fx, indexing='ij')
        
        v00 = grid[Y0, X0]
        v01 = grid[Y0, X1]
        v10 = grid[Y1, X0]
        v11 = grid[Y1, X1]
        
        v0 = v00 * (1 - FX) + v01 * FX
        v1 = v10 * (1 - FX) + v11 * FX
        
        noise += (v0 * (1 - FY) + v1 * FY) * amp
    
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def _add_chemical_edge_distortion(image: Image.Image, intensity: float = 1.0, seed: int = None) -> Image.Image:
    """
    Simulate Polaroid chemical edge distortion effects.
    
    This emulates the uneven chemical spread from the reagent pod, light leaks,
    and color bleeding that occurs at the edges of instant film.
    
    Effects include:
    - Left edge: Blue/cyan/magenta color bleeding
    - Right edge: Subtle green/teal tones  
    - Top edge: Blue tones with occasional pink light leaks
    - Bottom edge: Vertical orange/amber streaks (reagent pod effect)
    - Corner intensification
    
    Args:
        image: Input PIL Image
        intensity: Overall effect intensity (0.0 to 1.0+)
        seed: Random seed for reproducibility
    
    Returns:
        Image with chemical edge distortion applied
    """
    if seed is not None:
        np.random.seed(seed)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    img_arr = np.array(image).astype(float)
    
    # Create distance maps from each edge (normalized 0-1, 0 at edge)
    y_coords = np.linspace(0, 1, height)
    x_coords = np.linspace(0, 1, width)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    dist_left = X
    dist_right = 1 - X
    dist_top = Y
    dist_bottom = 1 - Y
    
    # Generate organic noise masks for each edge
    noise_left = _generate_perlin_noise_2d_fast((height, width), scale=50, octaves=3, seed=seed)
    noise_right = _generate_perlin_noise_2d_fast((height, width), scale=60, octaves=3, seed=(seed + 1) if seed else None)
    noise_top = _generate_perlin_noise_2d_fast((height, width), scale=55, octaves=3, seed=(seed + 2) if seed else None)
    noise_bottom = _generate_perlin_noise_2d_fast((height, width), scale=40, octaves=4, seed=(seed + 3) if seed else None)
    
    # Edge falloff width (as fraction of image dimension) - wider for more visible effect
    edge_width = 0.12
    
    # --- LEFT EDGE: Blue/Cyan/Magenta ---
    left_mask = np.clip(1 - dist_left / edge_width, 0, 1)
    left_mask = left_mask * (0.6 + 0.4 * noise_left)  # Modulate with noise
    left_mask = left_mask ** 1.2  # Softer falloff for wider spread
    
    # Color: Mix of cyan and magenta - more saturated
    left_color_r = left_mask * (80 * noise_left + 150 * (1 - noise_left))
    left_color_g = left_mask * (120 * noise_left + 50 * (1 - noise_left))
    left_color_b = left_mask * (255 * noise_left + 220 * (1 - noise_left))
    
    # --- RIGHT EDGE: Green/Teal tones ---
    right_mask = np.clip(1 - dist_right / (edge_width * 0.8), 0, 1)
    right_mask = right_mask * (0.5 + 0.5 * noise_right)
    right_mask = right_mask ** 1.5
    
    right_color_r = right_mask * 60
    right_color_g = right_mask * (160 * noise_right + 100)
    right_color_b = right_mask * (200 * noise_right + 140)
    
    # --- TOP EDGE: Blue with pink/magenta accents ---
    top_mask = np.clip(1 - dist_top / (edge_width * 0.9), 0, 1)
    top_mask = top_mask * (0.6 + 0.4 * noise_top)
    top_mask = top_mask ** 1.3
    
    # Add occasional pink "light leak" spots
    light_leak_noise = _generate_perlin_noise_2d_fast((height, width), scale=30, octaves=2, seed=(seed + 10) if seed else None)
    light_leak_mask = (light_leak_noise > 0.65).astype(float) * top_mask * 0.7
    
    top_color_r = top_mask * (100 + 150 * light_leak_mask / (top_mask + 0.01))
    top_color_g = top_mask * 80
    top_color_b = top_mask * 240
    
    # --- BOTTOM EDGE: Vertical Orange/Amber Streaks ---
    bottom_base_mask = np.clip(1 - dist_bottom / (edge_width * 1.5), 0, 1)
    
    # Generate vertical streaks - more streaks and taller
    streak_mask = _generate_vertical_streaks(width, height, 
                                              num_streaks=int(width * 0.05),
                                              max_height_frac=0.20,
                                              seed=seed)
    
    # Combine base mask with streaks - streaks more prominent
    bottom_mask = np.maximum(bottom_base_mask * 0.4, streak_mask * 1.2)
    bottom_mask = bottom_mask * (0.7 + 0.3 * noise_bottom)
    
    # Warm orange/amber color - more saturated
    bottom_color_r = bottom_mask * 255
    bottom_color_g = bottom_mask * (160 + 50 * noise_bottom)
    bottom_color_b = bottom_mask * 30
    
    # --- CORNER INTENSIFICATION ---
    # Bottom-left corner
    corner_bl = np.sqrt((X ** 2) + ((1 - Y) ** 2))
    corner_bl = np.clip(1 - corner_bl / 0.25, 0, 1) ** 1.5
    # Bottom-right corner
    corner_br = np.sqrt(((1 - X) ** 2) + ((1 - Y) ** 2))
    corner_br = np.clip(1 - corner_br / 0.25, 0, 1) ** 1.5
    # Top-left corner - add some blue/magenta
    corner_tl = np.sqrt((X ** 2) + (Y ** 2))
    corner_tl = np.clip(1 - corner_tl / 0.2, 0, 1) ** 1.5
    
    corner_mask_warm = (corner_bl + corner_br) * (0.6 + 0.4 * noise_bottom)
    corner_mask_cool = corner_tl * (0.6 + 0.4 * noise_top)
    
    # Warm corner color (bottom corners)
    corner_color_r = corner_mask_warm * 220 + corner_mask_cool * 120
    corner_color_g = corner_mask_warm * 120 + corner_mask_cool * 60
    corner_color_b = corner_mask_warm * 50 + corner_mask_cool * 200
    
    # --- COMBINE ALL EFFECTS ---
    # Apply intensity scaling - much higher base factor
    scale = intensity * 1.0  # Full intensity scaling
    
    # Combine color layers
    overlay_r = (left_color_r + right_color_r + top_color_r + bottom_color_r + corner_color_r) * scale
    overlay_g = (left_color_g + right_color_g + top_color_g + bottom_color_g + corner_color_g) * scale
    overlay_b = (left_color_b + right_color_b + top_color_b + bottom_color_b + corner_color_b) * scale
    
    # Combined mask for blending strength
    combined_mask = np.maximum.reduce([left_mask, right_mask, top_mask, bottom_mask, 
                                        corner_mask_warm, corner_mask_cool])
    
    # Use a hybrid blending approach:
    # - Additive component for visibility on all backgrounds
    # - Screen component for natural light-like behavior
    result = img_arr.copy()
    
    # Additive blend (works on all backgrounds) with 60% weight
    # Screen blend (natural light behavior) with 40% weight
    for c, overlay_c in enumerate([overlay_r, overlay_g, overlay_b]):
        additive = result[:, :, c] + overlay_c * 0.6
        screen = result[:, :, c] + overlay_c * 0.4 * (1 - result[:, :, c] / 255)
        result[:, :, c] = np.clip(additive + screen - result[:, :, c], 0, 255)
    
    return Image.fromarray(result.astype(np.uint8))


def _generate_vertical_streaks(width: int, height: int, num_streaks: int = 20, 
                                max_height_frac: float = 0.15, seed: int = None) -> np.ndarray:
    """
    Generate vertical streaks emanating from the bottom edge.
    Simulates the reagent pod chemical spread pattern.
    
    Args:
        width: Image width
        height: Image height
        num_streaks: Number of streaks to generate
        max_height_frac: Maximum streak height as fraction of image height
        seed: Random seed
    
    Returns:
        2D numpy array mask with streak pattern
    """
    if seed is not None:
        np.random.seed(seed + 100)
    
    mask = np.zeros((height, width))
    
    for _ in range(num_streaks):
        # Random x position
        x_center = np.random.randint(0, width)
        
        # Random streak properties
        streak_height = int(height * max_height_frac * (0.3 + 0.7 * np.random.rand()))
        streak_width = np.random.randint(1, 4)
        streak_intensity = 0.4 + 0.6 * np.random.rand()
        
        # Create streak with varying intensity along height
        for dy in range(streak_height):
            y = height - 1 - dy
            if y < 0:
                break
            
            # Intensity falls off with height (stronger at bottom)
            height_factor = 1 - (dy / streak_height) ** 0.5
            
            # Add some waviness
            wave = int(2 * np.sin(dy * 0.1 + np.random.rand() * np.pi))
            
            for dx in range(-streak_width, streak_width + 1):
                x = x_center + dx + wave
                if 0 <= x < width:
                    # Width falloff
                    width_factor = 1 - abs(dx) / (streak_width + 1)
                    mask[y, x] = max(mask[y, x], 
                                     streak_intensity * height_factor * width_factor)
    
    # Apply slight blur for softness
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))
    mask = np.array(mask_img).astype(float) / 255
    
    return mask
