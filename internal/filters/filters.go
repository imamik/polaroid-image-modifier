package filters

import (
	"image"
	"image/color"
	"math"
	"math/rand"
	"time"

	"github.com/disintegration/imaging"
	"github.com/fogleman/gg"
)

type FilmType string

const (
	FilmTypePolaroid FilmType = "polaroid"
	FilmTypeInstax   FilmType = "instax"
)

type Options struct {
	FilmType           FilmType
	ChemicalDistortion bool
	ChemistryOverlay   image.Image
}

func Apply(img image.Image, opts Options) image.Image {
	src := imaging.Clone(img)
	src = applyAutocontrast8Asymmetric(src, 0.5, 0.0)

	bounds := src.Bounds()
	work := image.NewNRGBA64(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			work.Set(x, y, src.At(x, y))
		}
	}

	work = applyLocalShadowsHighlights16(work, 0.6, 0.4)
	work = applyChromaticAberration16(work, 1.5)
	work = applySoftnessBloom16(work)

	vignetteIntensity := 0.4
	if opts.FilmType == FilmTypeInstax {
		vignetteIntensity = 0.25
	}
	work = applyVignette16(work, vignetteIntensity)

	if opts.FilmType == FilmTypeInstax {
		work = applyInstaxCurves16(work)
	} else {
		work = applyVintageCurves16(work)
	}
	work = applySaturation16(work, -15.0)

	if opts.ChemistryOverlay != nil {
		work = applyChemistryOverlay16(work, opts.ChemistryOverlay)
	} else if opts.ChemicalDistortion {
		work = applyChemicalDistortion16(work, 1.0)
	}
	work = applyInboundShadow16(work, 0.03, 0.045, 0.35)
	work = applyInboundShadow16(work, 0.003, 0.003, 0.30)

	work = applyFilmGrain16(work, 0.021)
	work = applyUnifiedHDRPolish16(work, 0.5, 0.0, 1.25, 18, 0.85)

	return imaging.Clone(work)
}

func applyUnifiedHDRPolish16(img *image.NRGBA64, lowCutoff, highCutoff, contrastFactor, liftAmount, gammaExp float64) *image.NRGBA64 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w*h == 0 {
		return img
	}

	var hR, hG, hB [256]int
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := img.NRGBA64At(x, y)
			hR[c.R>>8]++
			hG[c.G>>8]++
			hB[c.B>>8]++
		}
	}

	findRange := func(hist [256]int, lowCut, highCut float64) (float64, float64) {
		lowThr := int(float64(w*h) * lowCut / 100.0)
		highThr := int(float64(w*h) * highCut / 100.0)
		low, high := 0, 255
		sum := 0
		for i := 0; i < 256; i++ {
			sum += hist[i]
			if sum > lowThr {
				low = i
				break
			}
		}
		sum = 0
		for i := 255; i >= 0; i-- {
			sum += hist[i]
			if sum > highThr {
				high = i
				break
			}
		}
		return float64(low << 8), float64(high << 8)
	}

	lowR, highR := findRange(hR, lowCutoff, highCutoff)
	lowG, highG := findRange(hG, lowCutoff, highCutoff)
	lowB, highB := findRange(hB, lowCutoff, highCutoff)

	var lutR, lutG, lutB [65536]uint16
	lift := liftAmount * 257.0

	for i := 0; i < 65536; i++ {
		process := func(val float64, low, high float64) uint16 {
			if high > low {
				val = (val - low) * 65535.0 / (high - low)
			}
			val = math.Max(0, math.Min(65535, val))

			if val < 16384 {
				f := val / 16384.0
				boost := (1.0 - f) * 4000.0
				val += boost
			}
			if val > 55700 {
				f := (val - 55700.0) / (65535.0 - 55700.0)
				val = 55700.0 + (val-55700.0)*(1.0-f*0.3)
			}

			val = (val-32768.0)*contrastFactor + 32768.0
			val = math.Max(0, math.Min(65535, val))

			val = lift + (val/65535.0)*(65535.0-lift)
			val = math.Pow(val/65535.0, gammaExp) * 65535.0

			return uint16(math.Max(0, math.Min(65535, val)))
		}
		lutR[i] = process(float64(i), lowR, highR)
		lutG[i] = process(float64(i), lowG, highG)
		lutB[i] = process(float64(i), lowB, highB)
	}

	res := image.NewNRGBA64(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := img.NRGBA64At(x, y)
			res.SetNRGBA64(x, y, color.NRGBA64{lutR[c.R], lutG[c.G], lutB[c.B], c.A})
		}
	}
	return res
}

func applyAutocontrast8Asymmetric(img *image.NRGBA, lowCutoff, highCutoff float64) *image.NRGBA {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	var hist [256]int
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.NRGBAAt(x+bounds.Min.X, y+bounds.Min.Y)
			lum := int(0.299*float64(c.R) + 0.587*float64(c.G) + 0.114*float64(c.B))
			hist[lum]++
		}
	}
	lowThr := int(float64(w*h) * lowCutoff / 100.0)
	highThr := int(float64(w*h) * highCutoff / 100.0)
	low, high := 0, 255
	sum := 0
	for i := 0; i < 256; i++ {
		sum += hist[i]
		if sum > lowThr {
			low = i
			break
		}
	}
	sum = 0
	for i := 255; i >= 0; i-- {
		sum += hist[i]
		if sum > highThr {
			high = i
			break
		}
	}
	if high <= low {
		return img
	}
	return imaging.AdjustFunc(img, func(c color.NRGBA) color.NRGBA {
		r := (float64(c.R) - float64(low)) * 255.0 / float64(high-low)
		g := (float64(c.G) - float64(low)) * 255.0 / float64(high-low)
		b := (float64(c.B) - float64(low)) * 255.0 / float64(high-low)
		return color.NRGBA{uint8(math.Max(0, math.Min(255, r))), uint8(math.Max(0, math.Min(255, g))), uint8(math.Max(0, math.Min(255, b))), c.A}
	})
}

func applyVintageCurves16(img *image.NRGBA64) *image.NRGBA64 {
	bounds := img.Bounds()
	res := image.NewNRGBA64(bounds)
	t := float64(25700)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := img.NRGBA64At(x, y)
			r, g, b := float64(c.R), float64(c.G), float64(c.B)
			if r < t-1280 {
				r *= 1.05
			} else if r > t+1280 {
				r = r + (65535-r)*0.1
			} else {
				v1, v2 := r*1.05, r+(65535-r)*0.1
				f := (r - (t - 1280)) / 2560.0
				r = v1*(1-f) + v2*f
			}
			g = g*0.98 + 1280
			b = b*0.85 + 5120
			res.SetNRGBA64(x, y, color.NRGBA64{uint16(math.Max(0, math.Min(65535, r))), uint16(math.Max(0, math.Min(65535, g))), uint16(math.Max(0, math.Min(65535, b))), c.A})
		}
	}
	return res
}

func applyInstaxCurves16(img *image.NRGBA64) *image.NRGBA64 {
	bounds := img.Bounds()
	res := image.NewNRGBA64(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := img.NRGBA64At(x, y)
			r, g, b := float64(c.R), float64(c.G), float64(c.B)
			if r > 51400 {
				r *= 0.95
			} else if r > 12850 {
				r *= 1.05
			}
			b = b*1.05 + 1280
			res.SetNRGBA64(x, y, color.NRGBA64{uint16(math.Max(0, math.Min(65535, r))), uint16(math.Max(0, math.Min(65535, g))), uint16(math.Max(0, math.Min(65535, b))), c.A})
		}
	}
	return res
}

func applyVignette16(img *image.NRGBA64, intensity float64) *image.NRGBA64 {
	bounds := img.Bounds()
	wf, hf := float64(bounds.Dx()), float64(bounds.Dy())
	cx, cy := wf/2.0, hf/2.0
	maxR := math.Sqrt(cx*cx + cy*cy)
	res := image.NewNRGBA64(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := img.NRGBA64At(x, y)
			dx, dy := float64(x-bounds.Min.X)-cx, float64(y-bounds.Min.Y)-cy
			mask := 1.0 - math.Max(0, math.Min(1, math.Sqrt(dx*dx+dy*dy)/maxR-0.5))*intensity
			res.SetNRGBA64(x, y, color.NRGBA64{uint16(float64(c.R) * mask), uint16(float64(c.G) * mask), uint16(float64(c.B) * mask), c.A})
		}
	}
	return res
}

func applyInboundShadow16(img *image.NRGBA64, marginF, blurF, intensity float64) *image.NRGBA64 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	margin, blur := float64(minInt(w, h))*marginF, float64(minInt(w, h))*blurF
	dc := gg.NewContext(w, h)
	dc.SetColor(color.Black)
	dc.Clear()
	dc.SetColor(color.White)
	dc.DrawRectangle(margin, margin, float64(w)-2*margin, float64(h)-2*margin)
	dc.Fill()
	mask := imaging.Blur(dc.Image(), blur)
	shadow := color.NRGBA64{3855, 2570, 2570, 65535}
	res := image.NewNRGBA64(bounds)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.NRGBA64At(x+bounds.Min.X, y+bounds.Min.Y)
			mc := color.NRGBA64Model.Convert(mask.At(x, y)).(color.NRGBA64)
			a := (65535.0 - float64(mc.R)) / 65535.0 * intensity
			res.SetNRGBA64(x+bounds.Min.X, y+bounds.Min.Y, color.NRGBA64{uint16(float64(c.R)*(1-a) + float64(shadow.R)*a), uint16(float64(c.G)*(1-a) + float64(shadow.G)*a), uint16(float64(c.B)*(1-a) + float64(shadow.B)*a), c.A})
		}
	}
	return res
}

func applySoftnessBloom16(img *image.NRGBA64) *image.NRGBA64 {
	img8 := imaging.Clone(img)
	diff8 := imaging.Blur(img8, 1.0)
	thresh := uint8(180)
	high8 := imaging.AdjustFunc(diff8, func(c color.NRGBA) color.NRGBA {
		if c.R < thresh && c.G < thresh && c.B < thresh {
			return color.NRGBA{0, 0, 0, c.A}
		}
		m := 255.0 / (255.0 - float64(thresh))
		return color.NRGBA{uint8(math.Max(0, (float64(c.R)-float64(thresh))*m)), uint8(math.Max(0, (float64(c.G)-float64(thresh))*m)), uint8(math.Max(0, (float64(c.B)-float64(thresh))*m)), c.A}
	})
	bloom8 := imaging.Blur(high8, float64(minInt(img8.Bounds().Dx(), img8.Bounds().Dy()))*0.015)
	bounds := img.Bounds()
	res := image.NewNRGBA64(bounds)
	intens := 0.495
	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			c := diff8.NRGBAAt(x, y)
			bc := bloom8.NRGBAAt(x, y)
			r := 65535.0 - ((65535.0 - float64(c.R)*257.0) * (65535.0 - float64(bc.R)*257.0*intens) / 65535.0)
			g := 65535.0 - ((65535.0 - float64(c.G)*257.0) * (65535.0 - float64(bc.G)*257.0*intens) / 65535.0)
			b := 65535.0 - ((65535.0 - float64(c.B)*257.0) * (65535.0 - float64(bc.B)*257.0*intens) / 65535.0)
			res.SetNRGBA64(x+bounds.Min.X, y+bounds.Min.Y, color.NRGBA64{uint16(r), uint16(g), uint16(b), uint16(c.A) << 8})
		}
	}
	return res
}

func applyChromaticAberration16(img *image.NRGBA64, shift float64) *image.NRGBA64 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	res := image.NewNRGBA64(bounds)

	getPixel := func(x, y float64) (float64, float64, float64) {
		x0 := math.Floor(x)
		x1 := x0 + 1
		y0 := math.Floor(y)
		y1 := y0 + 1

		wx1 := x - x0
		wx0 := 1.0 - wx1
		wy1 := y - y0
		wy0 := 1.0 - wy1

		clampX := func(v float64) int {
			iv := int(v)
			if iv < 0 {
				return 0
			}
			if iv >= w {
				return w - 1
			}
			return iv
		}
		clampY := func(v float64) int {
			iv := int(v)
			if iv < 0 {
				return 0
			}
			if iv >= h {
				return h - 1
			}
			return iv
		}

		c00 := img.NRGBA64At(clampX(x0)+bounds.Min.X, clampY(y0)+bounds.Min.Y)
		c01 := img.NRGBA64At(clampX(x0)+bounds.Min.X, clampY(y1)+bounds.Min.Y)
		c10 := img.NRGBA64At(clampX(x1)+bounds.Min.X, clampY(y0)+bounds.Min.Y)
		c11 := img.NRGBA64At(clampX(x1)+bounds.Min.X, clampY(y1)+bounds.Min.Y)

		r := (float64(c00.R)*wx0*wy0 + float64(c01.R)*wx0*wy1 + float64(c10.R)*wx1*wy0 + float64(c11.R)*wx1*wy1)
		g := (float64(c00.G)*wx0*wy0 + float64(c01.G)*wx0*wy1 + float64(c10.G)*wx1*wy0 + float64(c11.G)*wx1*wy1)
		b := (float64(c00.B)*wx0*wy0 + float64(c01.B)*wx0*wy1 + float64(c10.B)*wx1*wy0 + float64(c11.B)*wx1*wy1)

		return r, g, b
	}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			gc := img.NRGBA64At(x+bounds.Min.X, y+bounds.Min.Y)
			rr, _, _ := getPixel(float64(x)+shift, float64(y)+shift)
			_, _, bb := getPixel(float64(x)-shift, float64(y)-shift)

			res.SetNRGBA64(x+bounds.Min.X, y+bounds.Min.Y, color.NRGBA64{
				uint16(math.Max(0, math.Min(65535, rr))),
				gc.G,
				uint16(math.Max(0, math.Min(65535, bb))),
				gc.A,
			})
		}
	}
	return res
}

func applySaturation16(img *image.NRGBA64, amount float64) *image.NRGBA64 {
	f := (100.0 + amount) / 100.0
	bounds := img.Bounds()
	res := image.NewNRGBA64(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := img.NRGBA64At(x, y)
			lum := 0.299*float64(c.R) + 0.587*float64(c.G) + 0.114*float64(c.B)
			res.SetNRGBA64(x, y, color.NRGBA64{uint16(math.Max(0, math.Min(65535, lum+(float64(c.R)-lum)*f))), uint16(math.Max(0, math.Min(65535, lum+(float64(c.G)-lum)*f))), uint16(math.Max(0, math.Min(65535, lum+(float64(c.B)-lum)*f))), c.A})
		}
	}
	return res
}

func applyFilmGrain16(img *image.NRGBA64, amount float64) *image.NRGBA64 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	src := rand.NewSource(time.Now().UnixNano())
	r := rand.New(src)
	res := image.NewNRGBA64(bounds)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.NRGBA64At(x+bounds.Min.X, y+bounds.Min.Y)
			n := r.NormFloat64() * 65535 * amount
			res.SetNRGBA64(x+bounds.Min.X, y+bounds.Min.Y, color.NRGBA64{uint16(math.Max(0, math.Min(65535, float64(c.R)+n))), uint16(math.Max(0, math.Min(65535, float64(c.G)+n))), uint16(math.Max(0, math.Min(65535, float64(c.B)+n))), c.A})
		}
	}
	return res
}

func applyChemistryOverlay16(img *image.NRGBA64, overlay image.Image) *image.NRGBA64 {
	bounds := img.Bounds()
	ov := imaging.Resize(overlay, bounds.Dx(), bounds.Dy(), imaging.Lanczos)
	ov = imaging.AdjustContrast(ov, 63.35)
	ov = imaging.AdjustBrightness(ov, -9.75)
	res := image.NewNRGBA64(bounds)
	op := 0.45
	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			c := img.NRGBA64At(x+bounds.Min.X, y+bounds.Min.Y)
			oc := color.NRGBA64Model.Convert(ov.At(x, y)).(color.NRGBA64)
			sr := 65535.0 - ((65535.0 - float64(c.R)) * (65535.0 - float64(oc.R)) / 65535.0)
			sg := 65535.0 - ((65535.0 - float64(c.G)) * (65535.0 - float64(oc.G)) / 65535.0)
			sb := 65535.0 - ((65535.0 - float64(c.B)) * (65535.0 - float64(oc.B)) / 65535.0)
			res.SetNRGBA64(x+bounds.Min.X, y+bounds.Min.Y, color.NRGBA64{uint16(float64(c.R)*(1-op) + sr*op), uint16(float64(c.G)*(1-op) + sg*op), uint16(float64(c.B)*(1-op) + sb*op), c.A})
		}
	}
	return res
}

func applyChemicalDistortion16(img *image.NRGBA64, intensity float64) *image.NRGBA64 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	sr := rand.New(rand.NewSource(time.Now().UnixNano()))
	grid := image.NewNRGBA(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			grid.SetNRGBA(x, y, color.NRGBA{uint8(sr.Intn(256)), uint8(sr.Intn(256)), uint8(sr.Intn(256)), 255})
		}
	}
	noise := imaging.Resize(grid, w, h, imaging.Lanczos)
	res := image.NewNRGBA64(bounds)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.NRGBA64At(x+bounds.Min.X, y+bounds.Min.Y)
			nc := color.NRGBA64Model.Convert(noise.At(x, y)).(color.NRGBA64)
			dl, dr, dt, db := float64(x)/float64(w), 1.0-float64(x)/float64(w), float64(y)/float64(h), 1.0-float64(y)/float64(h)
			ew := 0.12
			mask := math.Max(math.Max(0, 1.0-dl/ew), math.Max(math.Max(0, 1.0-dr/ew), math.Max(math.Max(0, 1.0-dt/ew), math.Max(0, 1.0-db/ew))))
			oR, oG, oB := mask*(38550*float64(nc.R)/65535.0+25700), mask*(30855*float64(nc.G)/65535.0+12850), mask*(51415*float64(nc.B)/65535.0+25700)
			fr := 65535.0 - ((65535.0 - float64(c.R)) * (65535.0 - oR*intensity) / 65535.0)
			fg := 65535.0 - ((65535.0 - float64(c.G)) * (65535.0 - oG*intensity) / 65535.0)
			fb := 65535.0 - ((65535.0 - float64(c.B)) * (65535.0 - oB*intensity) / 65535.0)
			res.SetNRGBA64(x+bounds.Min.X, y+bounds.Min.Y, color.NRGBA64{uint16(math.Min(65535, fr)), uint16(math.Min(65535, fg)), uint16(math.Min(65535, fb)), c.A})
		}
	}
	return res
}

func applyLocalShadowsHighlights16(img *image.NRGBA64, amountShadows, amountHighlights float64) *image.NRGBA64 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	lum8 := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.NRGBA64At(x+bounds.Min.X, y+bounds.Min.Y)
			lum := 0.299*float64(c.R) + 0.587*float64(c.G) + 0.114*float64(c.B)
			lum8.SetGray(x, y, color.Gray{uint8(lum / 257.0)})
		}
	}

	radius := 100.0
	blurredLum8 := imaging.Blur(lum8, radius)

	res := image.NewNRGBA64(bounds)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.NRGBA64At(x+bounds.Min.X, y+bounds.Min.Y)
			localAvg := float64(color.GrayModel.Convert(blurredLum8.At(x, y)).(color.Gray).Y) / 255.0

			r, g, b := float64(c.R)/65535.0, float64(c.G)/65535.0, float64(c.B)/65535.0
			pixLum := 0.299*r + 0.587*g + 0.114*b

			shadowMask := math.Max(0, 1.0-localAvg)
			shadowLift := amountShadows * shadowMask * (1.0 - pixLum)

			highlightMask := math.Max(0, localAvg)
			highlightCompress := amountHighlights * highlightMask * pixLum

			factor := 1.0 + shadowLift - highlightCompress

			fr := r * factor
			fg := g * factor
			fb := b * factor

			res.SetNRGBA64(x+bounds.Min.X, y+bounds.Min.Y, color.NRGBA64{
				uint16(math.Max(0, math.Min(1, fr)) * 65535.0),
				uint16(math.Max(0, math.Min(1, fg)) * 65535.0),
				uint16(math.Max(0, math.Min(1, fb)) * 65535.0),
				c.A,
			})
		}
	}
	return res
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
