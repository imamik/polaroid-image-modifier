from PIL import Image, ImageOps
import os
from .filters import apply_polaroid_effect
from .frames import add_polaroid_frame, FRAME_SPECS

class ImagePipeline:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.image = None
        
    def load(self):
        """Load image from disk"""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        self.image = Image.open(self.input_path)
        # Ensure we are in RGB mode (handle PNGs with transparency or Grayscale)
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')
        return self

    def prepare(self, frame_type: str = 'polaroid_600'):
        """
        Pre-crop/resize the image to the target frame's inner dimensions.
        This ensures that subsequent filters (vignette, shadow) are applied
        to the visible area only.
        """
        if self.image:
            if frame_type not in FRAME_SPECS:
                raise ValueError(f"Unknown frame type: {frame_type}")
            
            spec = FRAME_SPECS[frame_type]
            target_w, target_h = spec['image_size']
            
            # Use same high-quality downsampling as frames.py
            self.image = ImageOps.fit(self.image, (target_w, target_h), method=Image.Resampling.LANCZOS)
        return self

    def apply_filter(self, film_type: str = 'polaroid'):
        """Apply color grading and effects"""
        if self.image:
            self.image = apply_polaroid_effect(self.image, film_type=film_type)
        return self

    def add_frame(self, frame_type: str = 'polaroid_600'):
        """Add the specified frame"""
        if self.image:
            self.image = add_polaroid_frame(self.image, frame_type)
        return self

    def save(self, output_path: str, quality: int = 95):
        """Save result to disk"""
        if self.image:
            self.image.save(output_path, quality=quality)
        return self
