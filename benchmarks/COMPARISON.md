# Rust OCR and Tesseract CPU comparison

The configured Tesseract pipeline matched Rust compact at seven character edits
in 3,184 reference characters and had a lower observed median batch time. Rust
compact used a smaller model bundle. Rust balanced made four character edits.

## Native macOS ARM64 results

Both systems ran natively on the same Mac with RustO and OpenMP thread settings
of two. Three fresh-process trials per configuration processed the same ordered
12 synthetic images. Execution order changed between trials. Each process reused
one initialized engine for its batch. All character edit counts were identical
across the three trials for a given configuration.

| Engine | Median 12-page batch | Range across 3 batches | Median page | CER |
| --- | ---: | ---: | ---: | ---: |
| Rust balanced | 37.11 s | 8.54–44.92 s | 3.01 s | 0.13% |
| Rust compact | 18.50 s | 5.25–34.52 s | 1.22 s | 0.22% |
| Tesseract 5.5.4, PSM 1 / Otsu | 11.68 s | 4.52–19.27 s | 0.67 s | 14.64% |
| Tesseract 5.5.4, document preset | 9.97 s | 6.03–16.64 s | 0.51 s | 0.22% |

Batch time includes process startup, model verification/loading, file reading,
preprocessing, OCR, and result output. Page time includes reading and preprocessing,
with model loading measured separately. CER collapses whitespace and preserves
case, punctuation, and reading order.

The timing ranges overlap substantially. These exploratory runs do not establish
an idle-machine latency guarantee or a general speed ranking. The configured
Tesseract runs used 5.07–6.47 CPU-seconds per batch, compared with 8.30–13.73 for
Rust compact. CPU time sums user and system time over the process's threads.

## Footprint

| Engine | Model files | Median peak process RSS | Median CPU time per batch |
| --- | ---: | ---: | ---: |
| Rust balanced | 12.49 MiB | 265.1 MiB | 26.23 CPU-seconds |
| Rust compact | 9.46 MiB | 125.0 MiB | 12.65 CPU-seconds |
| Tesseract 5.5.4, PSM 1 / Otsu | 15.07 MiB | 136.1 MiB | 6.85 CPU-seconds |
| Tesseract 5.5.4, document preset | 15.07 MiB | 138.2 MiB | 5.90 CPU-seconds |

Model size includes the detection, recognition, dictionary, and rotation assets
for Rust and eng, ind, and osd traineddata for Tesseract. Runtime binaries and
shared libraries are excluded from that column. Tesseract RSS includes the small
Python ctypes measurement driver. Rust RSS covers the compiled measurement
executable using the public library. These are measured process footprints.

## Tesseract document preset

Every image receives the same data-dependent rules, without using its transcript
or filename to choose settings.

- Stretch grayscale luminance between its 0.5th and 99.5th percentiles when the
  observed range spans 12–105 levels, using the same contrast-restoration strategy
  as the Rust pipeline.
- Upscale images with a longest side below 1,000 pixels to 1,000 pixels.
- Estimate skew within ±15 degrees using Leptonica. Rotate with white fill when
  confidence is at least 2 and the angle magnitude is at least 0.5 degrees.
- Set image resolution to 300 DPI and run Tesseract with eng+ind, PSM 1, and
  Sauvola thresholding (`thresholding_method=2`). PSM 1 handles page orientation.

The preset was developed and checked on these same synthetic examples. This is
a development-set comparison. The image set contains repeated source documents
with different degradations and provides limited evidence about real documents.
[Tesseract image-quality guidance](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)

## Character edits by difficult image

| Image | Rust balanced | Rust compact | Tesseract Otsu | Tesseract document |
| --- | ---: | ---: | ---: | ---: |
| faded.jpg | 3 | 6 | 0 | 2 |
| low-resolution.jpg | 0 | 0 | 25 | 5 |
| shadowed.jpg | 0 | 1 | 218 | 0 |
| skewed.png | 1 | 0 | 223 | 0 |

The earlier 14.64% Tesseract CER remains reproducible for the PSM 1 / Otsu
configuration. Adding contrast recovery, scaling, deskewing, and adaptive
thresholding substantially changes that result. The earlier general claim that
the Rust prototype is more accurate than Tesseract was too broad.

## Evidence and reproduction

[Native macOS measurements, input/model hashes, and all transcripts](cpu-comparison-macos-arm64.json)
contain every run. Tesseract 5.5.4 and Leptonica 1.87.0 came from the pinned
`tesserocr==2.13.0` macOS ARM64 wheel. The original eng, ind, and osd model files
were reused from the Linux setup and checked by SHA-256.
[Benchmark wheel distribution](https://pypi.org/project/tesserocr/2.13.0/)

The [Linux experiment](cpu-comparison-linux-arm64.json) separately records three
trials of both Rust profiles and Tesseract 5.5.0 configurations. Its document
preset predates contrast restoration. Timings were highly variable and a CPU
pressure snapshot showed 52.03% recent scheduler pressure. Its results should be
interpreted within that Linux environment.

Build the public-library measurement executable, then run the comparison harness
with the appropriate native library paths and a local copy of the three traineddata
files. The harness supports an explicit engine list, preserves failed runs, and
records the complete commands.

```sh
cargo build --release --locked --example cpu_latency
python3 benchmarks/compare_cpu.py \
  --tessdata models/tessdata \
  --tesseract-library /path/to/libtesseract.5.5.4.dylib \
  --leptonica-library /path/to/libleptonica.6.0.0.dylib \
  --engines rust-balanced rust-compact tesseract-otsu tesseract-document-contrast \
  --trials 3 --threads 2 --output /tmp/ocr-comparison.json
```
