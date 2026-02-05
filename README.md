# Polaroid Image Modifier

[![CI](https://github.com/imamik/polaroid-image-modifier/actions/workflows/ci.yml/badge.svg)](https://github.com/imamik/polaroid-image-modifier/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/imamik/polaroid-image-modifier)](https://goreportcard.com/report/github.com/imamik/polaroid-image-modifier)
[![Go Reference](https://pkg.go.dev/badge/github.com/imamik/polaroid-image-modifier.svg)](https://pkg.go.dev/github.com/imamik/polaroid-image-modifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/imamik/polaroid-image-modifier)](https://github.com/imamik/polaroid-image-modifier/releases)

Apply authentic instant film effects to your images. Supports Polaroid 600 and Instax (Mini, Square, Wide) frame types with realistic color grading, vignette, grain, and chemical distortion effects.

## Features

- **Multiple Frame Types**: Polaroid 600, Instax Mini, Instax Square, Instax Wide
- **Authentic Film Effects**: Color grading, vignette, chromatic aberration, film grain
- **Chemical Distortion**: Realistic edge effects and chemistry overlays
- **Batch Processing**: Process entire directories of images
- **CLI & Library**: Use as a command-line tool or import as a Go package

## Installation

### CLI

```bash
# Using Go
go install github.com/imamik/polaroid-image-modifier/cmd/polaroid@latest

# Or download from releases
# https://github.com/imamik/polaroid-image-modifier/releases
```

### Library

```bash
go get github.com/imamik/polaroid-image-modifier
```

## Usage

### CLI

#### Process a Single Image

```bash
polaroid process -i input.jpg -o output.jpg -f polaroid_600 -t polaroid
```

#### Batch Process

```bash
polaroid batch -i ./input_dir -o ./output_dir -f instax_mini -t instax
```

#### Options

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--input` | `-i` | Input file or directory | (required) |
| `--output` | `-o` | Output file or directory | (required) |
| `--frame` | `-f` | Frame type | `polaroid_600` |
| `--film` | `-t` | Film type effect | `polaroid` |
| `--chemistry` | `-c` | Chemistry overlay directory | (auto-detect) |
| `--dpi` | `-d` | Output DPI reference | `300` |

#### Frame Types

- `polaroid_600` / `polaroid` - Classic Polaroid 600 format
- `instax_mini` / `mini` - Fujifilm Instax Mini
- `instax_square` / `square` - Fujifilm Instax Square
- `instax_wide` / `wide` - Fujifilm Instax Wide

#### Film Types

- `polaroid` - Warm, vintage Polaroid color grading
- `instax` - Cooler, modern Instax color grading

### Library

```go
package main

import (
    "log"

    polaroid "github.com/imamik/polaroid-image-modifier"
)

func main() {
    opts := polaroid.DefaultOptions()
    opts.FrameType = polaroid.FramePolaroid600
    opts.FilmType = polaroid.FilmPolaroid

    err := polaroid.Process("input.jpg", "output.jpg", opts)
    if err != nil {
        log.Fatal(err)
    }
}
```

#### Available Types

```go
// Frame types
polaroid.FramePolaroid600
polaroid.FrameInstaxMini
polaroid.FrameInstaxSq
polaroid.FrameInstaxWide

// Film types
polaroid.FilmPolaroid
polaroid.FilmInstax
```

## Chemistry Overlays

The `chemistry/` directory contains PNG overlays that simulate chemical development artifacts. These are randomly applied and flipped for variety. You can add your own overlays to this directory.

## Development

### Prerequisites

- Go 1.23+
- [pre-commit](https://pre-commit.com/) (optional, for git hooks)

### Setup

```bash
# Clone the repository
git clone https://github.com/imamik/polaroid-image-modifier.git
cd polaroid-image-modifier

# Install dependencies
go mod download

# Install pre-commit hooks (optional)
pre-commit install
pre-commit install --hook-type commit-msg

# Build
go build ./cmd/polaroid

# Run tests
go test ./...
```

### Project Structure

```
.
├── cmd/polaroid/       # CLI application
├── internal/           # Internal packages
│   ├── filters/        # Image filter effects
│   ├── frames/         # Frame specifications
│   └── pipeline/       # Processing pipeline
├── chemistry/          # Chemistry overlay images
├── polaroid.go         # Public API
└── go.mod
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes using [Conventional Commits](https://www.conventionalcommits.org/)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.
