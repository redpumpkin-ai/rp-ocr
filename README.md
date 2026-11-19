# RP OCR

RP OCR is a standalone Rust library and command-line tool for printed document
scans. It runs on the CPU using the RustO inference core and pretrained
PP-OCRv6 Tiny detection and recognition models. Page orientation uses a
PaddleOCR classifier. A contrast adjustment helps recover washed-out scans.
[Inference core](https://github.com/byrizki/rusto-rs) · [Model bundle](https://huggingface.co/byrizki/rusto-models/tree/34812c324fa258436fbcf1959eaf4ace7af6e5d4)

The CLI reads JPEG and PNG images. It can return plain text or JSON with
reading order, confidence scores, and line polygons. The first release
targets printed Latin-script documents, with English and Indonesian in the
benchmark. Use rasterized pages for PDF input.

## Run

Build with Rust 1.96.1 and Cargo. Model setup uses `curl` to download about
14 MB of weights and verifies each file by SHA-256. Run `prepare` on a
connected machine, then copy the verified model directory alongside the
binary for offline use. Scanning only uses local files.

```sh
cargo build --release --locked
target/release/rp-ocr prepare
target/release/rp-ocr scan benchmarks/fixtures/clean-letter.png
target/release/rp-ocr scan --json benchmarks/fixtures/rotated-90.png
```

Use `--models /path/to/models` to point to the model directory. For multiple
inputs, text output separates pages with a form feed and JSON returns one
object per input, in input order. Polygon coordinates are in pixels from the
top left, after EXIF correction. Automatic page rotation still returns
polygons in the original coordinate space of that EXIF-corrected image.
Confidence scores are model scores. Images can have up to 32 million pixels
and 8,192 pixels per side.

## Rust API

```rust
use std::path::Path;
use rp_ocr::Engine;

fn main() -> anyhow::Result<()> {
    let mut engine = Engine::open(Path::new("models"))?;
    let page = engine.scan_file(Path::new("scan.png"))?;
    println!("{}", page.text());
    Ok(())
}
```

Use `scan_rgb` for an in-memory `image::RgbImage`.

## Verification

```sh
cargo test --locked
cargo test --release --locked --test offline -- --ignored
```

The offline test uses the prepared model bundle and checks the transcript,
reading order, and coordinates of a rotated Indonesian scan.

## Benchmark

`benchmarks/fixtures` contains synthetic scans and their reference transcripts.
Each engine processes the 12 images in one CLI invocation. The benchmark
records character and word error rates and the elapsed time for the batch,
including startup. Text normalization collapses whitespace and preserves
case, punctuation, and reading order. Tesseract 5.5.0 runs with `eng+ind` in
modes 3 and 1. Mode 1 enables orientation detection.
[Tesseract command-line reference](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html)

```sh
python3 benchmarks/compare.py --output /tmp/rp-ocr-results.json
```

Python is used by the benchmark harness. OCR and model verification run in
the Rust binary. The harness can collect separate runs with `--only rp-ocr`
or `--only tesseract` when the engines run on different platforms.

### Recorded run

The 12 images come from five synthetic English and Indonesian documents.

| Engine | Runtime platform | Character error rate | Word error rate | Batch time |
| --- | --- | ---: | ---: | ---: |
| RP OCR 0.1.0 | macOS ARM64 | 0.13% | 1.09% | 48.0 s |
| Tesseract 5.5.0, PSM 3 | Linux ARM64 | 30.21% | 34.64% | 159.3 s |
| Tesseract 5.5.0, PSM 1 | Linux ARM64 | 14.64% | 18.08% | 208.0 s |

RP OCR made four character errors in 3,184 reference characters. It tied
Tesseract with orientation detection on the three rotated scans and had fewer
errors on the low-resolution, shadowed, and skewed scans. Tesseract read the
faded scan exactly, where RP OCR made three character errors.

The offline Linux smoke test passed. The complete Linux RP OCR batch hit a
300-second timeout, so the accuracy comparison uses macOS output for RP OCR
and Linux output for Tesseract. The batch times describe their individual
platforms. Recognition results cover this synthetic suite.
[RP OCR transcripts](benchmarks/macos-arm64.json) · [Tesseract transcripts](benchmarks/linux-arm64.json)
