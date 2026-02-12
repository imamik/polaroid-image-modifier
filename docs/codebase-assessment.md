# Codebase Assessment and Improvement Plan

## Current architecture: what is working

- Clear package boundaries: `internal/filters`, `internal/frames`, and `internal/pipeline` map well to the image-processing flow.
- Public API is small and approachable via `Process`, `ProcessImage`, and simple option types.
- Test coverage exists across all major internal packages.

## Architectural improvements

### 1) Introduce a configurable, deterministic processing context

**Current state**
- Randomness is seeded from `time.Now()` in multiple places (`film grain`, `chemical distortion`, `chemistry overlay selection`).
- This makes output non-reproducible and harder to test.

**Recommendation**
- Add `Seed int64` (or `RNG *rand.Rand`) to public `Options` and thread it through pipeline and filters.
- Default behavior can keep non-deterministic output by seeding once when `Seed == 0`.
- Add snapshot tests that lock seed and compare stable image hashes.

**Benefits**
- Reproducible renders, easier regression testing, easier bug reports.

### 2) Convert monolithic filter pipeline into composable stages

**Current state**
- `filters.Apply` does many responsibilities at once (contrast, local tone mapping, aberration, bloom, curves, vignette, chemistry effects, grain, final polish).

**Recommendation**
- Model each pass as a stage type:
  - `type Stage interface { Name() string; Apply(*image.NRGBA64, Context) *image.NRGBA64 }`
- Build named stage pipelines per film stock (`Polaroid600Pipeline`, `InstaxPipeline`).
- Allow optional stage toggles and parameter presets for CLI experimentation.

**Benefits**
- Better extensibility for new film profiles and effects.
- Easier profiling and A/B comparison per stage.

### 3) Split frame geometry from frame rendering style

**Current state**
- `frames.Spec` mixes layout geometry with a single hard-coded frame look.

**Recommendation**
- Keep `Spec` as geometry only.
- Add style/paper profile struct (`paper color`, `border stroke`, `corner radius`, `paper texture`, `aging tint`).
- Allow film-dependent frame styles so Polaroid and Instax can look materially different.

**Benefits**
- More realistic output and easier addition of special editions.

### 4) Improve error visibility around chemistry overlays

**Current state**
- Overlay discovery/loads fail silently in `loadChemistryOverlay`.

**Recommendation**
- Return warning-rich errors (or structured diagnostics) from overlay loading.
- Expose a debug mode to print selected overlay and whether fallback distortion was used.

**Benefits**
- Easier troubleshooting, especially in CLI batch mode.

### 5) Add performance instrumentation and avoid excess allocations

**Current state**
- Many full-frame conversions and allocations (`Clone`, `NewNRGBA64`) between nearly every pass.

**Recommendation**
- Add benchmarks for representative resolutions.
- Reuse image buffers when possible (ping-pong buffers).
- Consider parallelizing independent pixel loops with worker chunks.

**Benefits**
- Better throughput for batch processing and large images.

## Making polaroids look more real

### 1) Physically motivated grain model

- Replace purely per-pixel Gaussian additive grain with luma-dependent grain:
  - stronger in shadows, subtler in highlights,
  - slight channel decorrelation,
  - optional larger clumps at low frequencies.
- Add separate controls for `grain_size`, `grain_amount`, and `grain_chroma`.

### 2) Add halation and highlight bloom tied to bright regions

- Real instant film often exhibits warm halation around highlights.
- Build a thresholded highlight mask, blur it asymmetrically, and add warm tint (R > G > B).
- Keep intensity film-specific: stronger for vintage Polaroid, milder for Instax.

### 3) Film-stock-specific tone response curves

- Current curves are global and relatively simple.
- Add measured/fit LUTs per stock with subtle toe/shoulder behavior.
- Consider separate curves for each RGB channel and optional cross-talk matrix.

### 4) Better optical artifacts

- Current chromatic aberration is uniform shift.
- Prefer radial aberration increasing with distance from center.
- Add slight barrel distortion and soft corner focus falloff.

### 5) Chemistry realism from flow maps (not pure edge masks)

- Instead of only edge-weighted random overlays, generate or load low-frequency flow maps that create streaks, pooling, and uneven reagent spread.
- Blend with local image luminance so artifacts appear chemically plausible, not uniformly synthetic.

### 6) Frame/paper realism

- Add subtle paper texture and tiny print-surface gloss variation.
- Slightly warm paper whites and add gentle edge darkening/soiling in old-film presets.
- Introduce small per-render border offsets/rotation to mimic scanner or print placement imperfections.

## Suggested phased roadmap

### Phase 1 (quick wins)
- Deterministic seed support.
- Stage breakdown for `filters.Apply`.
- Better overlay diagnostics.

### Phase 2 (quality)
- Halation, improved grain, stock LUTs.
- Radial chromatic aberration + mild lens distortion.

### Phase 3 (polish + scale)
- Paper texture/material profiles.
- Performance pass with benchmarks and buffer reuse.
- Golden-image regression tests for each stock/frame profile.
