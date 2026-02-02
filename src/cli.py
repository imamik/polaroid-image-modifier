import argparse
import sys
import os
from .pipeline import ImagePipeline
from .frames import FRAME_SPECS

def main():
    parser = argparse.ArgumentParser(description="Convert images to Polaroid/Instax aesthetic")
    parser.add_argument('--input', '-i', required=True, help="Input image path")
    parser.add_argument('--output', '-o', required=True, help="Output image path")
    parser.add_argument('--type', '-t', default='polaroid', 
                        choices=['polaroid', 'instax', 'instax_square', 'instax_wide'],
                        help="Type of frame to apply")
    parser.add_argument('--no-filter', action='store_true', help="Disable color grading/vintage effect")
    
    args = parser.parse_args()
    
    # Map CLI types to internal frame types
    type_map = {
        'polaroid': 'polaroid_600',
        'instax': 'instax_mini',
        'instax_square': 'instax_square',
        'instax_wide': 'instax_wide'
    }
    
    frame_type = type_map[args.type]
    
    print(f"Processing {args.input}...")
    try:
        pipeline = ImagePipeline(args.input).load()
        
        # Pre-crop to frame specs so filters apply to visible area
        pipeline.prepare(frame_type)
        
        if not args.no_filter:
            print("Applying vintage filters...")
            # Determine film type from frame type
            film_type = 'instax' if 'instax' in args.type else 'polaroid'
            pipeline.apply_filter(film_type=film_type)
            
        print(f"Adding {args.type} frame...")
        pipeline.add_frame(frame_type)
        
        pipeline.save(args.output)
        print(f"Saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
