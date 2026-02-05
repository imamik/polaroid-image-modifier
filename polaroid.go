package polaroid

import (
	"image"

	"github.com/imamik/polaroid-image-modifier/internal/filters"
	"github.com/imamik/polaroid-image-modifier/internal/frames"
	"github.com/imamik/polaroid-image-modifier/internal/pipeline"
)

type FrameType = frames.FrameType

const (
	FramePolaroid600 FrameType = frames.FrameTypePolaroid600
	FrameInstaxMini  FrameType = frames.FrameTypeInstaxMini
	FrameInstaxSq    FrameType = frames.FrameTypeInstaxSq
	FrameInstaxWide  FrameType = frames.FrameTypeInstaxWide
)

type FilmType = filters.FilmType

const (
	FilmPolaroid FilmType = filters.FilmTypePolaroid
	FilmInstax   FilmType = filters.FilmTypeInstax
)

type Options struct {
	FrameType    FrameType
	FilmType     FilmType
	ChemistryDir string
	DPI          int
}

func DefaultOptions() Options {
	return Options{
		FrameType:    FramePolaroid600,
		FilmType:     FilmPolaroid,
		ChemistryDir: "",
		DPI:          300,
	}
}

func Process(inputPath, outputPath string, opts Options) error {
	pipelineOpts := pipeline.Options{
		FrameType:    opts.FrameType,
		FilmType:     opts.FilmType,
		ChemistryDir: opts.ChemistryDir,
		DPI:          opts.DPI,
	}
	return pipeline.Process(inputPath, outputPath, pipelineOpts)
}

func ProcessImage(img image.Image, opts Options) (image.Image, error) {
	spec, err := frames.GetSpec(opts.FrameType)
	if err != nil {
		return nil, err
	}

	proc := pipeline.New("")
	proc.SetChemistryDir(opts.ChemistryDir)

	filterOpts := filters.Options{
		FilmType:           opts.FilmType,
		ChemicalDistortion: true,
		ChemistryOverlay:   nil,
	}

	filtered := filters.Apply(img, filterOpts)

	framed, err := frames.Add(filtered, opts.FrameType)
	if err != nil {
		return nil, err
	}

	_ = spec
	return framed, nil
}

func FrameTypes() []FrameType {
	return frames.ValidFrameTypes()
}
