package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/imamik/polaroid-image-modifier/internal/filters"
	"github.com/imamik/polaroid-image-modifier/internal/frames"
	"github.com/imamik/polaroid-image-modifier/internal/pipeline"
)

var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
)

var rootCmd = &cobra.Command{
	Use:   "polaroid",
	Short: "Apply instant film effects to images",
	Long: `Polaroid Image Modifier applies authentic instant film effects to your images.

Supports Polaroid 600 and Instax (Mini, Square, Wide) frame types with
realistic color grading, vignette, grain, and chemical distortion effects.`,
	Version: fmt.Sprintf("%s (commit: %s, built: %s)", version, commit, date),
}

var processCmd = &cobra.Command{
	Use:   "process",
	Short: "Process a single image",
	Long:  `Process a single image with instant film effects.`,
	RunE:  runProcess,
}

var batchCmd = &cobra.Command{
	Use:   "batch",
	Short: "Process multiple images in a directory",
	Long:  `Process all images in a directory with instant film effects.`,
	RunE:  runBatch,
}

var (
	inputPath    string
	outputPath   string
	frameType    string
	filmType     string
	chemistryDir string
	dpi          int
)

func init() {
	processCmd.Flags().StringVarP(&inputPath, "input", "i", "", "Input image file (required)")
	processCmd.Flags().StringVarP(&outputPath, "output", "o", "", "Output image file (required)")
	processCmd.Flags().StringVarP(&frameType, "frame", "f", "polaroid_600", "Frame type: polaroid_600, instax_mini, instax_square, instax_wide")
	processCmd.Flags().StringVarP(&filmType, "film", "t", "polaroid", "Film type: polaroid, instax")
	processCmd.Flags().StringVarP(&chemistryDir, "chemistry", "c", "", "Directory containing chemistry overlay images")
	processCmd.Flags().IntVarP(&dpi, "dpi", "d", 300, "Output DPI (for reference)")
	processCmd.MarkFlagRequired("input")
	processCmd.MarkFlagRequired("output")

	batchCmd.Flags().StringVarP(&inputPath, "input", "i", "", "Input directory (required)")
	batchCmd.Flags().StringVarP(&outputPath, "output", "o", "", "Output directory (required)")
	batchCmd.Flags().StringVarP(&frameType, "frame", "f", "polaroid_600", "Frame type: polaroid_600, instax_mini, instax_square, instax_wide")
	batchCmd.Flags().StringVarP(&filmType, "film", "t", "polaroid", "Film type: polaroid, instax")
	batchCmd.Flags().StringVarP(&chemistryDir, "chemistry", "c", "", "Directory containing chemistry overlay images")
	batchCmd.Flags().IntVarP(&dpi, "dpi", "d", 300, "Output DPI (for reference)")
	batchCmd.MarkFlagRequired("input")
	batchCmd.MarkFlagRequired("output")

	rootCmd.AddCommand(processCmd)
	rootCmd.AddCommand(batchCmd)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func parseFrameType(s string) (frames.FrameType, error) {
	switch strings.ToLower(s) {
	case "polaroid_600", "polaroid":
		return frames.FrameTypePolaroid600, nil
	case "instax_mini", "mini":
		return frames.FrameTypeInstaxMini, nil
	case "instax_square", "square":
		return frames.FrameTypeInstaxSq, nil
	case "instax_wide", "wide":
		return frames.FrameTypeInstaxWide, nil
	default:
		return "", fmt.Errorf("unknown frame type: %s (valid: polaroid_600, instax_mini, instax_square, instax_wide)", s)
	}
}

func parseFilmType(s string) (filters.FilmType, error) {
	switch strings.ToLower(s) {
	case "polaroid":
		return filters.FilmTypePolaroid, nil
	case "instax":
		return filters.FilmTypeInstax, nil
	default:
		return "", fmt.Errorf("unknown film type: %s (valid: polaroid, instax)", s)
	}
}

func runProcess(cmd *cobra.Command, args []string) error {
	ft, err := parseFrameType(frameType)
	if err != nil {
		return err
	}

	film, err := parseFilmType(filmType)
	if err != nil {
		return err
	}

	start := time.Now()
	fmt.Printf("Processing: %s\n", inputPath)

	opts := pipeline.Options{
		FrameType:    ft,
		FilmType:     film,
		ChemistryDir: chemistryDir,
		DPI:          dpi,
	}

	if err := pipeline.Process(inputPath, outputPath, opts); err != nil {
		return fmt.Errorf("processing failed: %w", err)
	}

	fmt.Printf("Done: %s (%dms)\n", outputPath, time.Since(start).Milliseconds())
	return nil
}

func runBatch(cmd *cobra.Command, args []string) error {
	ft, err := parseFrameType(frameType)
	if err != nil {
		return err
	}

	film, err := parseFilmType(filmType)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(outputPath, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	files, err := os.ReadDir(inputPath)
	if err != nil {
		return fmt.Errorf("failed to read input directory: %w", err)
	}

	opts := pipeline.Options{
		FrameType:    ft,
		FilmType:     film,
		ChemistryDir: chemistryDir,
		DPI:          dpi,
	}

	processed := 0
	for _, f := range files {
		if f.IsDir() {
			continue
		}

		ext := strings.ToLower(filepath.Ext(f.Name()))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
			continue
		}

		inPath := filepath.Join(inputPath, f.Name())
		baseName := strings.TrimSuffix(f.Name(), ext)
		outName := fmt.Sprintf("%s_%s%s", baseName, frameType, ext)
		outPath := filepath.Join(outputPath, outName)

		start := time.Now()
		fmt.Printf("[%d] Processing: %s ", processed+1, f.Name())

		if err := pipeline.Process(inPath, outPath, opts); err != nil {
			fmt.Printf("FAILED: %v\n", err)
			continue
		}

		fmt.Printf("(%dms)\n", time.Since(start).Milliseconds())
		processed++
	}

	fmt.Printf("\nBatch complete: %d images processed\n", processed)
	return nil
}
