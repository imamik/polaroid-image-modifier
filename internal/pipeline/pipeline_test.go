package pipeline

import (
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/imamik/polaroid-image-modifier/internal/filters"
	"github.com/imamik/polaroid-image-modifier/internal/frames"
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

func TestNew(t *testing.T) {
	proc := New("/path/to/image.jpg")
	if proc == nil {
		t.Fatal("New() returned nil")
	}
	if proc.inputPath != "/path/to/image.jpg" {
		t.Errorf("New() inputPath = %q, want %q", proc.inputPath, "/path/to/image.jpg")
	}
}

func TestProcessor_SetChemistryDir(t *testing.T) {
	proc := New("/path/to/image.jpg")
	proc.SetChemistryDir("/custom/chemistry")
	if proc.chemistryDir != "/custom/chemistry" {
		t.Errorf("SetChemistryDir() chemistryDir = %q, want %q", proc.chemistryDir, "/custom/chemistry")
	}
}

func TestProcessor_Load(t *testing.T) {
	tmpDir := t.TempDir()
	testImagePath := filepath.Join(tmpDir, "test.png")
	saveTestImage(t, createTestImage(100, 100), testImagePath)

	tests := []struct {
		name    string
		path    string
		wantErr bool
	}{
		{"valid image", testImagePath, false},
		{"non-existent file", "/nonexistent/image.png", true},
		{"invalid path", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proc := New(tt.path)
			err := proc.Load()
			if (err != nil) != tt.wantErr {
				t.Errorf("Load() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && proc.image == nil {
				t.Error("Load() did not set image")
			}
		})
	}
}

func TestProcessor_Prepare(t *testing.T) {
	tmpDir := t.TempDir()
	testImagePath := filepath.Join(tmpDir, "test.png")
	saveTestImage(t, createTestImage(100, 100), testImagePath)

	tests := []struct {
		name      string
		frameType frames.FrameType
		wantErr   bool
	}{
		{"polaroid_600", frames.FrameTypePolaroid600, false},
		{"instax_mini", frames.FrameTypeInstaxMini, false},
		{"invalid", frames.FrameType("invalid"), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proc := New(testImagePath)
			if err := proc.Load(); err != nil {
				t.Fatalf("Load() failed: %v", err)
			}
			err := proc.Prepare(tt.frameType)
			if (err != nil) != tt.wantErr {
				t.Errorf("Prepare() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestProcessor_PrepareWithoutLoad(t *testing.T) {
	proc := New("/path/to/image.jpg")
	err := proc.Prepare(frames.FrameTypePolaroid600)
	if err == nil {
		t.Error("Prepare() should fail without Load()")
	}
}

func TestProcessor_ApplyFilter(t *testing.T) {
	tmpDir := t.TempDir()
	testImagePath := filepath.Join(tmpDir, "test.png")
	saveTestImage(t, createTestImage(100, 100), testImagePath)

	tests := []struct {
		name     string
		filmType filters.FilmType
	}{
		{"polaroid", filters.FilmTypePolaroid},
		{"instax", filters.FilmTypeInstax},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proc := New(testImagePath)
			if err := proc.Load(); err != nil {
				t.Fatalf("Load() failed: %v", err)
			}
			if err := proc.Prepare(frames.FrameTypePolaroid600); err != nil {
				t.Fatalf("Prepare() failed: %v", err)
			}
			err := proc.ApplyFilter(tt.filmType)
			if err != nil {
				t.Errorf("ApplyFilter() error = %v", err)
			}
		})
	}
}

func TestProcessor_ApplyFilterWithoutLoad(t *testing.T) {
	proc := New("/path/to/image.jpg")
	err := proc.ApplyFilter(filters.FilmTypePolaroid)
	if err == nil {
		t.Error("ApplyFilter() should fail without Load()")
	}
}

func TestProcessor_AddFrame(t *testing.T) {
	tmpDir := t.TempDir()
	testImagePath := filepath.Join(tmpDir, "test.png")
	saveTestImage(t, createTestImage(100, 100), testImagePath)

	tests := []struct {
		name      string
		frameType frames.FrameType
		wantErr   bool
	}{
		{"polaroid_600", frames.FrameTypePolaroid600, false},
		{"instax_mini", frames.FrameTypeInstaxMini, false},
		{"invalid", frames.FrameType("invalid"), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proc := New(testImagePath)
			if err := proc.Load(); err != nil {
				t.Fatalf("Load() failed: %v", err)
			}
			if err := proc.Prepare(frames.FrameTypePolaroid600); err != nil {
				t.Fatalf("Prepare() failed: %v", err)
			}
			err := proc.AddFrame(tt.frameType)
			if (err != nil) != tt.wantErr {
				t.Errorf("AddFrame() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestProcessor_AddFrameWithoutLoad(t *testing.T) {
	proc := New("/path/to/image.jpg")
	err := proc.AddFrame(frames.FrameTypePolaroid600)
	if err == nil {
		t.Error("AddFrame() should fail without Load()")
	}
}

func TestProcessor_Save(t *testing.T) {
	tmpDir := t.TempDir()
	testImagePath := filepath.Join(tmpDir, "test.png")
	outputPath := filepath.Join(tmpDir, "output.jpg")
	saveTestImage(t, createTestImage(100, 100), testImagePath)

	proc := New(testImagePath)
	if err := proc.Load(); err != nil {
		t.Fatalf("Load() failed: %v", err)
	}

	err := proc.Save(outputPath)
	if err != nil {
		t.Errorf("Save() error = %v", err)
	}

	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		t.Error("Save() did not create output file")
	}
}

func TestProcessor_SaveWithoutLoad(t *testing.T) {
	proc := New("/path/to/image.jpg")
	err := proc.Save("/tmp/output.jpg")
	if err == nil {
		t.Error("Save() should fail without Load()")
	}
}

func TestProcessor_Image(t *testing.T) {
	tmpDir := t.TempDir()
	testImagePath := filepath.Join(tmpDir, "test.png")
	saveTestImage(t, createTestImage(100, 100), testImagePath)

	proc := New(testImagePath)
	if proc.Image() != nil {
		t.Error("Image() should return nil before Load()")
	}

	if err := proc.Load(); err != nil {
		t.Fatalf("Load() failed: %v", err)
	}

	if proc.Image() == nil {
		t.Error("Image() should return image after Load()")
	}
}

func TestProcess(t *testing.T) {
	tmpDir := t.TempDir()
	testImagePath := filepath.Join(tmpDir, "test.png")
	outputPath := filepath.Join(tmpDir, "output.jpg")
	saveTestImage(t, createTestImage(100, 100), testImagePath)

	opts := Options{
		FrameType: frames.FrameTypePolaroid600,
		FilmType:  filters.FilmTypePolaroid,
		DPI:       300,
		Seed:      99,
	}

	err := Process(testImagePath, outputPath, opts)
	if err != nil {
		t.Errorf("Process() error = %v", err)
	}

	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		t.Error("Process() did not create output file")
	}
}

func TestProcessWithInvalidInput(t *testing.T) {
	opts := Options{
		FrameType: frames.FrameTypePolaroid600,
		FilmType:  filters.FilmTypePolaroid,
	}

	err := Process("/nonexistent/image.png", "/tmp/output.jpg", opts)
	if err == nil {
		t.Error("Process() should fail with invalid input")
	}
}

func TestProcessWithChemistryDir(t *testing.T) {
	tmpDir := t.TempDir()
	testImagePath := filepath.Join(tmpDir, "test.png")
	outputPath := filepath.Join(tmpDir, "output.jpg")
	chemistryDir := filepath.Join(tmpDir, "chemistry")
	saveTestImage(t, createTestImage(100, 100), testImagePath)

	if err := os.MkdirAll(chemistryDir, 0750); err != nil {
		t.Fatalf("failed to create chemistry dir: %v", err)
	}
	saveTestImage(t, createTestImage(50, 50), filepath.Join(chemistryDir, "overlay.png"))

	opts := Options{
		FrameType:    frames.FrameTypePolaroid600,
		FilmType:     filters.FilmTypePolaroid,
		ChemistryDir: chemistryDir,
		DPI:          300,
	}

	err := Process(testImagePath, outputPath, opts)
	if err != nil {
		t.Errorf("Process() with chemistry dir error = %v", err)
	}
}

func TestOptions(t *testing.T) {
	opts := Options{
		FrameType:    frames.FrameTypePolaroid600,
		FilmType:     filters.FilmTypePolaroid,
		ChemistryDir: "/path/to/chemistry",
		DPI:          300,
	}

	if opts.FrameType != frames.FrameTypePolaroid600 {
		t.Errorf("Options.FrameType = %v, want %v", opts.FrameType, frames.FrameTypePolaroid600)
	}
	if opts.FilmType != filters.FilmTypePolaroid {
		t.Errorf("Options.FilmType = %v, want %v", opts.FilmType, filters.FilmTypePolaroid)
	}
	if opts.ChemistryDir != "/path/to/chemistry" {
		t.Errorf("Options.ChemistryDir = %v, want %v", opts.ChemistryDir, "/path/to/chemistry")
	}
	if opts.DPI != 300 {
		t.Errorf("Options.DPI = %v, want %v", opts.DPI, 300)
	}
}

func TestProcessor_ApplyFilterTracksChemistryWarnings(t *testing.T) {
	tmpDir := t.TempDir()
	testImagePath := filepath.Join(tmpDir, "test.png")
	saveTestImage(t, createTestImage(100, 100), testImagePath)

	proc := New(testImagePath)
	proc.SetChemistryDir(filepath.Join(tmpDir, "missing-chemistry"))
	if err := proc.Load(); err != nil {
		t.Fatalf("Load() failed: %v", err)
	}
	if err := proc.Prepare(frames.FrameTypePolaroid600); err != nil {
		t.Fatalf("Prepare() failed: %v", err)
	}
	if err := proc.ApplyFilter(filters.FilmTypePolaroid); err != nil {
		t.Fatalf("ApplyFilter() failed: %v", err)
	}
	if len(proc.Warnings()) == 0 {
		t.Fatal("expected chemistry warning when directory is missing")
	}
}
