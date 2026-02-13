package polaroid

import (
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

func createTestImage(width, height int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, color.NRGBA{255, 128, 64, 255})
		}
	}
	return img
}

func saveTestImage(t *testing.T, img image.Image, path string) {
	t.Helper()
	f, err := os.Create(path) //nolint:gosec // test file path is controlled
	if err != nil {
		t.Fatalf("failed to create test image: %v", err)
	}
	defer func() { _ = f.Close() }()
	if err := png.Encode(f, img); err != nil {
		t.Fatalf("failed to encode test image: %v", err)
	}
}

func TestDefaultOptions(t *testing.T) {
	opts := DefaultOptions()

	if opts.FrameType != FramePolaroid600 {
		t.Errorf("DefaultOptions().FrameType = %v, want %v", opts.FrameType, FramePolaroid600)
	}
	if opts.FilmType != FilmPolaroid {
		t.Errorf("DefaultOptions().FilmType = %v, want %v", opts.FilmType, FilmPolaroid)
	}
	if opts.ChemistryDir != "" {
		t.Errorf("DefaultOptions().ChemistryDir = %v, want empty", opts.ChemistryDir)
	}
	if opts.DPI != 300 {
		t.Errorf("DefaultOptions().DPI = %v, want 300", opts.DPI)
	}
	if opts.Seed != 0 {
		t.Errorf("DefaultOptions().Seed = %v, want 0", opts.Seed)
	}
}

func TestProcess(t *testing.T) {
	tmpDir := t.TempDir()
	testImagePath := filepath.Join(tmpDir, "test.png")
	saveTestImage(t, createTestImage(100, 100), testImagePath)

	tests := []struct {
		name      string
		frameType FrameType
		filmType  FilmType
	}{
		{"polaroid_600 with polaroid film", FramePolaroid600, FilmPolaroid},
		{"instax_mini with instax film", FrameInstaxMini, FilmInstax},
		{"instax_square with polaroid film", FrameInstaxSq, FilmPolaroid},
		{"instax_wide with instax film", FrameInstaxWide, FilmInstax},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := Options{
				FrameType: tt.frameType,
				FilmType:  tt.filmType,
				DPI:       300,
			}
			outPath := filepath.Join(tmpDir, tt.name+".jpg")
			err := Process(testImagePath, outPath, opts)
			if err != nil {
				t.Errorf("Process() error = %v", err)
			}
			if _, err := os.Stat(outPath); os.IsNotExist(err) {
				t.Error("Process() did not create output file")
			}
		})
	}
}

func TestProcessWithInvalidInput(t *testing.T) {
	opts := DefaultOptions()
	err := Process("/nonexistent/image.png", "/tmp/output.jpg", opts)
	if err == nil {
		t.Error("Process() should fail with invalid input")
	}
}

func TestProcessImage(t *testing.T) {
	img := createTestImage(100, 100)

	tests := []struct {
		name      string
		frameType FrameType
		filmType  FilmType
	}{
		{"polaroid_600", FramePolaroid600, FilmPolaroid},
		{"instax_mini", FrameInstaxMini, FilmInstax},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := Options{
				FrameType: tt.frameType,
				FilmType:  tt.filmType,
			}
			result, err := ProcessImage(img, opts)
			if err != nil {
				t.Errorf("ProcessImage() error = %v", err)
				return
			}
			if result == nil {
				t.Error("ProcessImage() returned nil")
			}
		})
	}
}

func TestProcessImageWithInvalidFrameType(t *testing.T) {
	img := createTestImage(100, 100)
	opts := Options{
		FrameType: FrameType("invalid"),
		FilmType:  FilmPolaroid,
	}
	_, err := ProcessImage(img, opts)
	if err == nil {
		t.Error("ProcessImage() should fail with invalid frame type")
	}
}

func TestFrameTypes(t *testing.T) {
	types := FrameTypes()
	if len(types) != 4 {
		t.Errorf("FrameTypes() returned %d types, want 4", len(types))
	}

	expected := map[FrameType]bool{
		FramePolaroid600: true,
		FrameInstaxMini:  true,
		FrameInstaxSq:    true,
		FrameInstaxWide:  true,
	}

	for _, ft := range types {
		if !expected[ft] {
			t.Errorf("FrameTypes() contains unexpected type: %s", ft)
		}
	}
}

func TestFrameTypeConstants(t *testing.T) {
	if FramePolaroid600 != "polaroid_600" {
		t.Errorf("FramePolaroid600 = %q, want %q", FramePolaroid600, "polaroid_600")
	}
	if FrameInstaxMini != "instax_mini" {
		t.Errorf("FrameInstaxMini = %q, want %q", FrameInstaxMini, "instax_mini")
	}
	if FrameInstaxSq != "instax_square" {
		t.Errorf("FrameInstaxSq = %q, want %q", FrameInstaxSq, "instax_square")
	}
	if FrameInstaxWide != "instax_wide" {
		t.Errorf("FrameInstaxWide = %q, want %q", FrameInstaxWide, "instax_wide")
	}
}

func TestFilmTypeConstants(t *testing.T) {
	if FilmPolaroid != "polaroid" {
		t.Errorf("FilmPolaroid = %q, want %q", FilmPolaroid, "polaroid")
	}
	if FilmInstax != "instax" {
		t.Errorf("FilmInstax = %q, want %q", FilmInstax, "instax")
	}
}

func TestOptionsStruct(t *testing.T) {
	opts := Options{
		FrameType:    FramePolaroid600,
		FilmType:     FilmPolaroid,
		ChemistryDir: "/custom/path",
		DPI:          600,
		Seed:         1234,
	}

	if opts.FrameType != FramePolaroid600 {
		t.Errorf("Options.FrameType = %v, want %v", opts.FrameType, FramePolaroid600)
	}
	if opts.FilmType != FilmPolaroid {
		t.Errorf("Options.FilmType = %v, want %v", opts.FilmType, FilmPolaroid)
	}
	if opts.ChemistryDir != "/custom/path" {
		t.Errorf("Options.ChemistryDir = %v, want %v", opts.ChemistryDir, "/custom/path")
	}
	if opts.DPI != 600 {
		t.Errorf("Options.DPI = %v, want %v", opts.DPI, 600)
	}
	if opts.Seed != 1234 {
		t.Errorf("Options.Seed = %v, want %v", opts.Seed, 1234)
	}
}
