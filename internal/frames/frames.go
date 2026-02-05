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

type Spec struct {
	TotalSize [2]int
	ImageSize [2]int
	ImagePos  [2]int
}

var Specs = map[FrameType]Spec{
	FrameTypePolaroid600: {
		TotalSize: [2]int{1080, 1296},
		ImageSize: [2]int{956, 956},
		ImagePos:  [2]int{62, 77},
	},
	FrameTypeInstaxMini: {
		TotalSize: [2]int{1080, 1720},
		ImageSize: [2]int{920, 1240},
		ImagePos:  [2]int{80, 100},
	},
	FrameTypeInstaxSq: {
		TotalSize: [2]int{1080, 1290},
		ImageSize: [2]int{930, 930},
		ImagePos:  [2]int{75, 105},
	},
	FrameTypeInstaxWide: {
		TotalSize: [2]int{1080, 860},
		ImageSize: [2]int{990, 620},
		ImagePos:  [2]int{45, 100},
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

	radius := 5.0
	borderWidth := 1.5
	borderColor := color.RGBA{60, 60, 60, 200}

	imgFilled := imaging.Fill(img, targetW, targetH, imaging.Center, imaging.Lanczos)

	dc := gg.NewContext(targetW, targetH)
	dc.SetColor(color.Transparent)
	dc.Clear()

	dc.DrawRoundedRectangle(0, 0, float64(targetW), float64(targetH), radius)
	dc.Clip()
	dc.DrawImage(imgFilled, 0, 0)
	dc.ResetClip()

	dc.DrawRoundedRectangle(0.5, 0.5, float64(targetW)-1.0, float64(targetH)-1.0, radius)
	dc.SetColor(borderColor)
	dc.SetLineWidth(borderWidth)
	dc.Stroke()

	finalImg := dc.Image()

	frameColor := color.RGBA{252, 252, 250, 255}
	totalW, totalH := spec.TotalSize[0], spec.TotalSize[1]
	frame := imaging.New(totalW, totalH, frameColor)

	pasteX, pasteY := spec.ImagePos[0], spec.ImagePos[1]
	result := imaging.Overlay(frame, finalImg, image.Pt(pasteX, pasteY), 1.0)

	return result, nil
}

func ValidFrameTypes() []FrameType {
	return []FrameType{
		FrameTypePolaroid600,
		FrameTypeInstaxMini,
		FrameTypeInstaxSq,
		FrameTypeInstaxWide,
	}
}
