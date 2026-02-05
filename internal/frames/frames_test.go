package frames

import (
	"image"
	"image/color"
	"testing"
)

func TestGetSpec(t *testing.T) {
	tests := []struct {
		name      string
		frameType FrameType
		wantErr   bool
	}{
		{"polaroid_600", FrameTypePolaroid600, false},
		{"instax_mini", FrameTypeInstaxMini, false},
		{"instax_square", FrameTypeInstaxSq, false},
		{"instax_wide", FrameTypeInstaxWide, false},
		{"invalid", FrameType("invalid"), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec, err := GetSpec(tt.frameType)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetSpec() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if spec.TotalSize[0] == 0 || spec.TotalSize[1] == 0 {
					t.Error("GetSpec() returned zero TotalSize")
				}
				if spec.ImageSize[0] == 0 || spec.ImageSize[1] == 0 {
					t.Error("GetSpec() returned zero ImageSize")
				}
			}
		})
	}
}

func TestAdd(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.RGBA{255, 0, 0, 255})
		}
	}

	tests := []struct {
		name      string
		frameType FrameType
		wantErr   bool
	}{
		{"polaroid_600", FrameTypePolaroid600, false},
		{"instax_mini", FrameTypeInstaxMini, false},
		{"instax_square", FrameTypeInstaxSq, false},
		{"instax_wide", FrameTypeInstaxWide, false},
		{"invalid", FrameType("invalid"), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Add(img, tt.frameType)
			if (err != nil) != tt.wantErr {
				t.Errorf("Add() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if result == nil {
					t.Error("Add() returned nil image")
					return
				}
				spec := Specs[tt.frameType]
				bounds := result.Bounds()
				if bounds.Dx() != spec.TotalSize[0] || bounds.Dy() != spec.TotalSize[1] {
					t.Errorf("Add() image size = %dx%d, want %dx%d",
						bounds.Dx(), bounds.Dy(), spec.TotalSize[0], spec.TotalSize[1])
				}
			}
		})
	}
}

func TestValidFrameTypes(t *testing.T) {
	types := ValidFrameTypes()
	if len(types) != 4 {
		t.Errorf("ValidFrameTypes() returned %d types, want 4", len(types))
	}

	expected := map[FrameType]bool{
		FrameTypePolaroid600: true,
		FrameTypeInstaxMini:  true,
		FrameTypeInstaxSq:    true,
		FrameTypeInstaxWide:  true,
	}

	for _, ft := range types {
		if !expected[ft] {
			t.Errorf("ValidFrameTypes() contains unexpected type: %s", ft)
		}
	}
}

func TestSpecs(t *testing.T) {
	for frameType, spec := range Specs {
		t.Run(string(frameType), func(t *testing.T) {
			if spec.TotalSize[0] <= 0 || spec.TotalSize[1] <= 0 {
				t.Error("TotalSize must be positive")
			}
			if spec.ImageSize[0] <= 0 || spec.ImageSize[1] <= 0 {
				t.Error("ImageSize must be positive")
			}
			if spec.ImagePos[0] < 0 || spec.ImagePos[1] < 0 {
				t.Error("ImagePos must be non-negative")
			}
			if spec.ImagePos[0]+spec.ImageSize[0] > spec.TotalSize[0] {
				t.Error("Image exceeds frame width")
			}
			if spec.ImagePos[1]+spec.ImageSize[1] > spec.TotalSize[1] {
				t.Error("Image exceeds frame height")
			}
		})
	}
}
