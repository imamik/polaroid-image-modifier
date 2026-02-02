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
    
    # Edge falloff width (as fraction of image dimension)
    edge_width = 0.08
    
    # --- LEFT EDGE: Blue/Cyan/Magenta ---
    left_mask = np.clip(1 - dist_left / edge_width, 0, 1)
    left_mask = left_mask * (0.5 + 0.5 * noise_left)  # Modulate with noise
    left_mask = left_mask ** 1.5  # Sharper falloff
    
    # Color: Mix of cyan (0, 180, 220) and magenta (180, 50, 150)
    left_color_r = left_mask * (50 * noise_left + 100 * (1 - noise_left))
    left_color_g = left_mask * (100 * noise_left + 30 * (1 - noise_left))
    left_color_b = left_mask * (220 * noise_left + 180 * (1 - noise_left))
    
    # --- RIGHT EDGE: Subtle Green/Teal ---
    right_mask = np.clip(1 - dist_right / (edge_width * 0.7), 0, 1)
    right_mask = right_mask * (0.4 + 0.6 * noise_right)
    right_mask = right_mask ** 2  # Even sharper falloff, more subtle
    
    right_color_r = right_mask * 40
    right_color_g = right_mask * (120 * noise_right + 80)
    right_color_b = right_mask * (150 * noise_right + 100)
    
    # --- TOP EDGE: Blue with pink accents ---
    top_mask = np.clip(1 - dist_top / (edge_width * 0.8), 0, 1)
    top_mask = top_mask * (0.5 + 0.5 * noise_top)
    top_mask = top_mask ** 1.8
    
    # Add occasional pink "light leak" spots
    light_leak_noise = _generate_perlin_noise_2d_fast((height, width), scale=30, octaves=2, seed=(seed + 10) if seed else None)
    light_leak_mask = (light_leak_noise > 0.7).astype(float) * top_mask * 0.5
    
    top_color_r = top_mask * (80 + 100 * light_leak_mask / (top_mask + 0.01))
    top_color_g = top_mask * 60
    top_color_b = top_mask * 200
    
    # --- BOTTOM EDGE: Vertical Orange/Amber Streaks ---
    bottom_base_mask = np.clip(1 - dist_bottom / (edge_width * 1.2), 0, 1)
    
    # Generate vertical streaks
    streak_mask = _generate_vertical_streaks(width, height, 
                                              num_streaks=int(width * 0.03),
                                              max_height_frac=0.15,
                                              seed=seed)
    
    # Combine base mask with streaks
    bottom_mask = np.maximum(bottom_base_mask * 0.3, streak_mask)
    bottom_mask = bottom_mask * (0.6 + 0.4 * noise_bottom)
    
    # Warm orange/amber color
    bottom_color_r = bottom_mask * 255
    bottom_color_g = bottom_mask * (140 + 60 * noise_bottom)
    bottom_color_b = bottom_mask * 40
    
    # --- CORNER INTENSIFICATION ---
    corner_mask = np.zeros((height, width))
    # Bottom-left corner
    corner_bl = np.sqrt((X ** 2) + ((1 - Y) ** 2))
    corner_bl = np.clip(1 - corner_bl / 0.2, 0, 1) ** 2
    # Bottom-right corner
    corner_br = np.sqrt(((1 - X) ** 2) + ((1 - Y) ** 2))
    corner_br = np.clip(1 - corner_br / 0.2, 0, 1) ** 2
    
    corner_mask = corner_bl + corner_br
    corner_mask = corner_mask * (0.5 + 0.5 * noise_bottom)
    
    # Warm corner color
    corner_color_r = corner_mask * 200
    corner_color_g = corner_mask * 100
    corner_color_b = corner_mask * 50
    
    # --- COMBINE ALL EFFECTS ---
    # Use screen blending for additive light-like effect
    def screen_blend(base, overlay, mask):
        mask_3d = mask[:, :, np.newaxis] if len(mask.shape) == 2 else mask
        return base + overlay * mask_3d * (1 - base / 255)
    
    # Apply intensity scaling
    scale = intensity * 0.4  # Base scaling factor
    
    # Combine color layers
    overlay_r = (left_color_r + right_color_r + top_color_r + bottom_color_r + corner_color_r) * scale
    overlay_g = (left_color_g + right_color_g + top_color_g + bottom_color_g + corner_color_g) * scale
    overlay_b = (left_color_b + right_color_b + top_color_b + bottom_color_b + corner_color_b) * scale
    
    # Apply using screen blend
    result = img_arr.copy()
    result[:, :, 0] = np.clip(result[:, :, 0] + overlay_r * (1 - result[:, :, 0] / 255), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] + overlay_g * (1 - result[:, :, 1] / 255), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] + overlay_b * (1 - result[:, :, 2] / 255), 0, 255)
    
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
