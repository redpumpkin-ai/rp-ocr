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

## Benchmark

`benchmarks/fixtures` contains synthetic scans and their reference transcripts.
The comparison records character error rate, word error rate, and median cold
CLI latency across three runs per image. Text normalization collapses
whitespace and preserves case, punctuation, and reading order. Tesseract runs
with `eng+ind` in modes 3 and 1. Mode 1 enables orientation detection.
[Tesseract command-line reference](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html)

```sh
python3 benchmarks/compare.py --output /tmp/rp-ocr-results.json
```

Python is used by the benchmark harness. OCR and model verification run in
the Rust binary.
