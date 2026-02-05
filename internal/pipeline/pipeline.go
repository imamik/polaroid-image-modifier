package pipeline

import (
	"fmt"
	"image"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/disintegration/imaging"
	"github.com/imamik/polaroid-image-modifier/internal/filters"
	"github.com/imamik/polaroid-image-modifier/internal/frames"
)

type Processor struct {
	inputPath        string
	image            image.Image
	chemistryDir     string
	chemistryOverlay image.Image
}

type Options struct {
	FrameType    frames.FrameType
	FilmType     filters.FilmType
	ChemistryDir string
	DPI          int
}

func New(inputPath string) *Processor {
	return &Processor{
		inputPath: inputPath,
	}
}

func (p *Processor) SetChemistryDir(dir string) {
	p.chemistryDir = dir
}

func (p *Processor) Load() error {
	if _, err := os.Stat(p.inputPath); os.IsNotExist(err) {
		return fmt.Errorf("input file not found: %s", p.inputPath)
	}

	img, err := imaging.Open(p.inputPath)
	if err != nil {
		return fmt.Errorf("failed to load image: %w", err)
	}

	p.image = img
	return nil
}

func (p *Processor) Prepare(frameType frames.FrameType) error {
	if p.image == nil {
		return fmt.Errorf("no image loaded")
	}

	spec, err := frames.GetSpec(frameType)
	if err != nil {
		return err
	}

	targetW, targetH := spec.ImageSize[0], spec.ImageSize[1]
	p.image = imaging.Fill(p.image, targetW, targetH, imaging.Center, imaging.Lanczos)
	return nil
}

func (p *Processor) loadChemistryOverlay() {
	chemistryDir := p.chemistryDir
	if chemistryDir == "" {
		chemistryDir = "chemistry"
		if _, err := os.Stat(chemistryDir); os.IsNotExist(err) {
			chemistryDir = "../chemistry"
			if _, err := os.Stat(chemistryDir); os.IsNotExist(err) {
				chemistryDir = "../../chemistry"
			}
		}
	}

	files, err := os.ReadDir(chemistryDir)
	if err != nil {
		return
	}

	var pngFiles []string
	for _, f := range files {
		if !f.IsDir() && strings.HasSuffix(strings.ToLower(f.Name()), ".png") {
			pngFiles = append(pngFiles, f.Name())
		}
	}

	if len(pngFiles) == 0 {
		return
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	chosen := pngFiles[r.Intn(len(pngFiles))]
	overlayPath := filepath.Join(chemistryDir, chosen)

	ovImage, err := imaging.Open(overlayPath)
	if err != nil {
		return
	}

	if r.Float64() < 0.5 {
		ovImage = imaging.FlipH(ovImage)
	}
	p.chemistryOverlay = ovImage
}

func (p *Processor) ApplyFilter(filmType filters.FilmType) error {
	if p.image == nil {
		return fmt.Errorf("no image loaded")
	}

	p.loadChemistryOverlay()

	opts := filters.Options{
		FilmType:           filmType,
		ChemicalDistortion: true,
		ChemistryOverlay:   p.chemistryOverlay,
	}

	p.image = filters.Apply(p.image, opts)
	return nil
}

func (p *Processor) AddFrame(frameType frames.FrameType) error {
	if p.image == nil {
		return fmt.Errorf("no image loaded")
	}

	framed, err := frames.Add(p.image, frameType)
	if err != nil {
		return err
	}

	p.image = framed
	return nil
}

func (p *Processor) Save(outputPath string) error {
	if p.image == nil {
		return fmt.Errorf("no image to save")
	}

	err := imaging.Save(p.image, outputPath)
	if err != nil {
		return fmt.Errorf("failed to save image: %w", err)
	}

	return nil
}

func (p *Processor) Image() image.Image {
	return p.image
}

func Process(inputPath, outputPath string, opts Options) error {
	proc := New(inputPath)
	if opts.ChemistryDir != "" {
		proc.SetChemistryDir(opts.ChemistryDir)
	}

	if err := proc.Load(); err != nil {
		return fmt.Errorf("load: %w", err)
	}

	if err := proc.Prepare(opts.FrameType); err != nil {
		return fmt.Errorf("prepare: %w", err)
	}

	if err := proc.ApplyFilter(opts.FilmType); err != nil {
		return fmt.Errorf("filter: %w", err)
	}

	if err := proc.AddFrame(opts.FrameType); err != nil {
		return fmt.Errorf("frame: %w", err)
	}

	if err := proc.Save(outputPath); err != nil {
		return fmt.Errorf("save: %w", err)
	}

	return nil
}
