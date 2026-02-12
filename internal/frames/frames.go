package frames

import (
	"fmt"
	"image"
	"image/color"

	"github.com/disintegration/imaging"
	"github.com/fogleman/gg"
)

type FrameType string

const (
	FrameTypePolaroid600 FrameType = "polaroid_600"
	FrameTypeInstaxMini  FrameType = "instax_mini"
	FrameTypeInstaxSq    FrameType = "instax_square"
	FrameTypeInstaxWide  FrameType = "instax_wide"
)

type Style struct {
	CornerRadius float64
	BorderWidth  float64
	BorderColor  color.RGBA
	PaperColor   color.RGBA
	PaperTint    color.RGBA
	PaperNoise   uint8
}

type Spec struct {
	TotalSize [2]int
	ImageSize [2]int
	ImagePos  [2]int
	Style     Style
}

var Specs = map[FrameType]Spec{
	FrameTypePolaroid600: {
		TotalSize: [2]int{1080, 1296},
		ImageSize: [2]int{956, 956},
		ImagePos:  [2]int{62, 77},
		Style:     Style{CornerRadius: 5.0, BorderWidth: 1.5, BorderColor: color.RGBA{60, 60, 60, 200}, PaperColor: color.RGBA{252, 252, 250, 255}, PaperTint: color.RGBA{10, 8, 0, 255}, PaperNoise: 4},
	},
	FrameTypeInstaxMini: {
		TotalSize: [2]int{1080, 1720},
		ImageSize: [2]int{920, 1240},
		ImagePos:  [2]int{80, 100},
		Style:     Style{CornerRadius: 4.0, BorderWidth: 1.1, BorderColor: color.RGBA{75, 75, 75, 170}, PaperColor: color.RGBA{248, 249, 247, 255}, PaperTint: color.RGBA{4, 6, 8, 255}, PaperNoise: 3},
	},
	FrameTypeInstaxSq: {
		TotalSize: [2]int{1080, 1290},
		ImageSize: [2]int{930, 930},
		ImagePos:  [2]int{75, 105},
		Style:     Style{CornerRadius: 4.5, BorderWidth: 1.2, BorderColor: color.RGBA{70, 70, 70, 180}, PaperColor: color.RGBA{249, 249, 246, 255}, PaperTint: color.RGBA{6, 6, 4, 255}, PaperNoise: 3},
	},
	FrameTypeInstaxWide: {
		TotalSize: [2]int{1080, 860},
		ImageSize: [2]int{990, 620},
		ImagePos:  [2]int{45, 100},
		Style:     Style{CornerRadius: 4.0, BorderWidth: 1.1, BorderColor: color.RGBA{70, 70, 70, 170}, PaperColor: color.RGBA{248, 249, 247, 255}, PaperTint: color.RGBA{4, 6, 8, 255}, PaperNoise: 3},
	},
}

func GetSpec(frameType FrameType) (Spec, error) {
	spec, ok := Specs[frameType]
	if !ok {
		return Spec{}, fmt.Errorf("unknown frame type: %s", frameType)
	}
	return spec, nil
}

func Add(img image.Image, frameType FrameType) (image.Image, error) {
	spec, ok := Specs[frameType]
	if !ok {
		return nil, fmt.Errorf("unknown frame type: %s", frameType)
	}

	targetW, targetH := spec.ImageSize[0], spec.ImageSize[1]

	style := spec.Style
	imgFilled := imaging.Fill(img, targetW, targetH, imaging.Center, imaging.Lanczos)

	dc := gg.NewContext(targetW, targetH)
	dc.SetColor(color.Transparent)
	dc.Clear()

	dc.DrawRoundedRectangle(0, 0, float64(targetW), float64(targetH), style.CornerRadius)
	dc.Clip()
	dc.DrawImage(imgFilled, 0, 0)
	dc.ResetClip()

	dc.DrawRoundedRectangle(0.5, 0.5, float64(targetW)-1.0, float64(targetH)-1.0, style.CornerRadius)
	dc.SetColor(style.BorderColor)
	dc.SetLineWidth(style.BorderWidth)
	dc.Stroke()

	finalImg := dc.Image()

	totalW, totalH := spec.TotalSize[0], spec.TotalSize[1]
	frame := newPaperFrame(totalW, totalH, style)

	pasteX, pasteY := spec.ImagePos[0], spec.ImagePos[1]
	result := imaging.Overlay(frame, finalImg, image.Pt(pasteX, pasteY), 1.0)

	return result, nil
}

func newPaperFrame(w, h int, style Style) *image.NRGBA {
	frame := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			fx := float64(x) / float64(maxInt(w-1, 1))
			fy := float64(y) / float64(maxInt(h-1, 1))
			shade := uint8((fx + fy) * float64(style.PaperNoise))
			frame.SetNRGBA(x, y, color.NRGBA{
				R: clamp8(int(style.PaperColor.R) + int(style.PaperTint.R)/12 - int(shade)/2),
				G: clamp8(int(style.PaperColor.G) + int(style.PaperTint.G)/12 - int(shade)/3),
				B: clamp8(int(style.PaperColor.B) + int(style.PaperTint.B)/12 - int(shade)/4),
				A: 255,
			})
		}
	}
	return frame
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func clamp8(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

func ValidFrameTypes() []FrameType {
	return []FrameType{
		FrameTypePolaroid600,
		FrameTypeInstaxMini,
		FrameTypeInstaxSq,
		FrameTypeInstaxWide,
	}
}
