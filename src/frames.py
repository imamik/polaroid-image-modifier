from PIL import Image, ImageOps

# Constants for frame dimensions (300 DPI, ~11.81 px/mm)
FRAME_SPECS = {
    'polaroid_600': {
        # 88mm x 107mm -> 1039 x 1264
        'total_size': (1039, 1264),  
        # 79mm x 79mm -> 933 x 933
        'image_size': (933, 933),   
        # (88-79)/2 = 4.5mm borders L/R -> 53px
        # Top border usually slightly more than side? 
        # Standard: 4.5mm sides, ~6mm top? let's center horizontally.
        # (1039 - 933) / 2 = 53
        # Top margin: ~6mm -> 71px
        'image_pos': (53, 71)       
    },
    'instax_mini': {
        # 54mm x 86mm -> 638 x 1016
        'total_size': (638, 1016),   
        # 46mm x 62mm -> 543 x 732
        'image_size': (543, 732),   
        # Sides: (54-46)/2 = 4mm -> 47px
        # Top: ~6mm -> 71px
        'image_pos': (47, 71)       
    },
    'instax_square': {
        # 72mm x 86mm -> 850 x 1016
        'total_size': (850, 1016),   
        # 62mm x 62mm -> 732 x 732
        # Sides: (72-62)/2 = 5mm -> 59px
        # Top: ~6mm -> 71px
        'image_pos': (59, 71)
    },
    'instax_wide': {
        # 108mm x 86mm -> 1276 x 1016
        'total_size': (1276, 1016),  
        # 99mm x 62mm -> 1169 x 732
        # Sides: (108-99)/2 = 4.5mm -> 53px
        # Top: ~6mm -> 71px
        'image_pos': (53, 71)
    }
}

def add_polaroid_frame(image: Image.Image, frame_type: str = 'polaroid_600') -> Image.Image:
    """
    Resize and crop the image to fit the specified frame type, then composite it onto the frame.
    """
    if frame_type not in FRAME_SPECS:
        raise ValueError(f"Unknown frame type: {frame_type}. Available: {list(FRAME_SPECS.keys())}")
    
    spec = FRAME_SPECS[frame_type]
    total_w, total_h = spec['total_size']
    target_w, target_h = spec['image_size']
    paste_x, paste_y = spec['image_pos']
    
    # 1. Resize and Center Crop input image to fill the target image area
    # mimic ImageOps.fit but with high quality downsampling
    if image.size != (target_w, target_h):
        image_filled = ImageOps.fit(image, (target_w, target_h), method=Image.Resampling.LANCZOS)
    else:
        image_filled = image
    
    # 2. Create Frame Background
    # White background, maybe slightly off-white for realism
    frame_color = (252, 252, 250) 
    frame = Image.new('RGB', (total_w, total_h), frame_color)
    
    # Optional: Add a very subtle shadow or texture to the frame could be done here
    # For now, solid color is cleaner
    
    # 3. Paste Image
    frame.paste(image_filled, (paste_x, paste_y))
    
    return frame
