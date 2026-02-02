import os
import sys
from PIL import Image, ImageDraw
from src.cli import main
from unittest.mock import patch

def create_test_image(path):
    # Create a nice colorful test image
    img = Image.new('RGB', (800, 600), color='skyblue')
    d = ImageDraw.Draw(img)
    d.rectangle([100, 400, 700, 600], fill='lightgreen', outline=None)
    d.ellipse([600, 50, 700, 150], fill='yellow', outline=None)
    img.save(path)
    print(f"Created test image at {path}")

def test_cli():
    input_img = "test_input.jpg"
    if not os.path.exists(input_img):
        create_test_image(input_img)
    else:
        print(f"Using existing {input_img}...")
    
    test_cases = [
        ("test_polaroid.jpg", "polaroid"),
        ("test_instax.jpg", "instax"),
        ("test_polaroid_nofilter.jpg", "polaroid", True),
    ]
    
    for output, type_arg, no_filter in [x if len(x)==3 else (*x, False) for x in test_cases]:
        args = ['src/cli.py', '--input', input_img, '--output', output, '--type', type_arg]
        if no_filter:
            args.append('--no-filter')
            
        print(f"\nRunning with args: {args}")
        with patch.object(sys, 'argv', args):
            try:
                main()
                if os.path.exists(output):
                    print(f"SUCCESS: {output} created.")
                else:
                    print(f"FAILURE: {output} not found.")
            except SystemExit as e:
                print(f"CLI exited with code {e}")
            except Exception as e:
                print(f"CLI crashed with {e}")

if __name__ == "__main__":
    test_cli()
