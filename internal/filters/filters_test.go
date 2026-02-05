package filters

import (
	"image"
	"image/color"
	"testing"
)

func createTestImage(width, height int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r := uint8((x * 255) / max(width, 1))  //nolint:gosec // test image generation
			g := uint8((y * 255) / max(height, 1)) //nolint:gosec // test image generation
			b := uint8(128)
			img.Set(x, y, color.NRGBA{r, g, b, 255})
		}
	}
	return img
}

func TestApply(t *testing.T) {
	img := createTestImage(100, 100)

	tests := []struct {
		name string
		opts Options
	}{
		{
			name: "polaroid film type",
			opts: Options{
				FilmType:           FilmTypePolaroid,
				ChemicalDistortion: false,
				ChemistryOverlay:   nil,
			},
		},
		{
			name: "instax film type",
			opts: Options{
				FilmType:           FilmTypeInstax,
				ChemicalDistortion: false,
				ChemistryOverlay:   nil,
			},
		},
		{
			name: "with chemical distortion",
			opts: Options{
				FilmType:           FilmTypePolaroid,
				ChemicalDistortion: true,
				ChemistryOverlay:   nil,
			},
		},
		{
			name: "with chemistry overlay",
			opts: Options{
				FilmType:           FilmTypePolaroid,
				ChemicalDistortion: false,
				ChemistryOverlay:   createTestImage(50, 50),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Apply(img, tt.opts)
			if result == nil {
				t.Error("Apply() returned nil")
				return
			}
			bounds := result.Bounds()
			if bounds.Dx() != 100 || bounds.Dy() != 100 {
				t.Errorf("Apply() changed image size: got %dx%d, want 100x100",
					bounds.Dx(), bounds.Dy())
			}
		})
	}
}

func TestFilmTypeConstants(t *testing.T) {
	if FilmTypePolaroid != "polaroid" {
		t.Errorf("FilmTypePolaroid = %q, want %q", FilmTypePolaroid, "polaroid")
	}
	if FilmTypeInstax != "instax" {
		t.Errorf("FilmTypeInstax = %q, want %q", FilmTypeInstax, "instax")
	}
}

func TestMinInt(t *testing.T) {
	tests := []struct {
		a, b, want int
	}{
		{1, 2, 1},
		{2, 1, 1},
		{5, 5, 5},
		{-1, 1, -1},
		{0, 0, 0},
	}

	for _, tt := range tests {
		got := minInt(tt.a, tt.b)
		if got != tt.want {
			t.Errorf("minInt(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}
}

func TestApplyWithEmptyImage(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 0, 0))
	opts := Options{
		FilmType:           FilmTypePolaroid,
		ChemicalDistortion: false,
	}

	result := Apply(img, opts)
	if result == nil {
		t.Error("Apply() returned nil for empty image")
	}
}

func TestApplyWithSmallImage(t *testing.T) {
	img := createTestImage(10, 10)
	opts := Options{
		FilmType:           FilmTypePolaroid,
		ChemicalDistortion: true,
	}

	result := Apply(img, opts)
	if result == nil {
		t.Error("Apply() returned nil for small image")
		return
	}
	bounds := result.Bounds()
	if bounds.Dx() != 10 || bounds.Dy() != 10 {
		t.Errorf("Apply() changed small image size: got %dx%d, want 10x10",
			bounds.Dx(), bounds.Dy())
	}
}

func TestApplyWithLargeImage(t *testing.T) {
	img := createTestImage(500, 500)
	opts := Options{
		FilmType:           FilmTypeInstax,
		ChemicalDistortion: true,
	}

	result := Apply(img, opts)
	if result == nil {
		t.Error("Apply() returned nil for large image")
		return
	}
	bounds := result.Bounds()
	if bounds.Dx() != 500 || bounds.Dy() != 500 {
		t.Errorf("Apply() changed large image size: got %dx%d, want 500x500",
			bounds.Dx(), bounds.Dy())
	}
}

func BenchmarkApply(b *testing.B) {
	img := createTestImage(200, 200)
	opts := Options{
		FilmType:           FilmTypePolaroid,
		ChemicalDistortion: false,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Apply(img, opts)
	}
}
