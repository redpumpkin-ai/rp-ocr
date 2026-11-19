# RP OCR

Standalone Rust OCR for printed English and Indonesian document scans. The CLI
and library run on CPU, using the RustO document pipeline, RTen inference and
pretrained PaddleOCR PP-OCRv6 Tiny models. No GPU, Tesseract or PDFium runtime is
required. Recognition was not trained from scratch.
[Inference core](https://github.com/byrizki/rusto-rs) ·
[Pinned model source](https://huggingface.co/byrizki/rusto-models/tree/34812c324fa258436fbcf1959eaf4ace7af6e5d4)

Inputs are PNG and JPEG. Output is plain text or JSON with page rotation,
confidence scores and line polygons. Rasterize PDF pages before supplying them.
Reuse one `Engine` when processing a batch.

## CPU profiles

| Profile | Detection | Recognition | Runtime model files |
| --- | --- | --- | ---: |
| `balanced` (default) | Original resolution | Full precision, multilingual | 12.49 MiB |
| `fast` (experimental) | Reduced resolution for clear, sparse pages; original otherwise | INT8, English/Indonesian/Latin alphabet | 10.36 MiB |
| `compact` | Maximum longest side 640 pixels | Half-precision weights, multilingual | 9.46 MiB |

Fast retains 692 output classes, including the CTC blank, Latin letters and
accents, numbers, currency, punctuation, mathematical symbols and space. Its
reader head is derived reproducibly from the pinned multilingual INT8 model.
Other alphabetic scripts require balanced or compact. Pruning changes the
available labels and confidence scores; it is not a replacement for language
training.

Fast uses a 640-pixel detection raster only when the source is at least 640
pixels on each side, has a white border, and has at most 24 dark text bands on
each axis. Other pages retain balanced detection. Recognition always reads
crops from the source image. All profiles include orientation correction,
contrast recovery and reading-order processing.

The [CPU optimization report](benchmarks/OPTIMIZATION.md) contains repeated
Tesseract comparisons, accuracy limits and footprint measurements. These
profiles were developed on synthetic scans. Dense-table reading order remains
inaccurate; real-document accuracy is not yet established.

## Build and run

Build with Rust 1.96.1 and Cargo. Model preparation uses `curl`, verifies pinned
SHA-256 hashes and, for fast, generates the smaller reader and dictionary locally.
Scanning uses local files only.

```sh
cargo build --release --locked
target/release/rp-ocr --profile fast prepare
RUSTO_NUM_THREADS=2 RTEN_NUM_THREADS=2 target/release/rp-ocr \
  --profile fast scan --json scan.png
```

Use `--models /path/to/models` to select a model directory. An offline fast
installation needs the executable and just these four files:

- `det.rten`
- `rec-latin.rten`
- `dict-latin.txt`
- `orient.rten`

The downloaded parent `rec-int8.rten` and `dict.txt` are preparation caches and
can be omitted from a fast-only deployment. Balanced needs `det.rten`, `rec.rten`,
`dict.txt` and `orient.rten`. Compact needs `det-compact.mnn`, `rec-compact.mnn`,
`dict.txt` and `orient.rten`. Tesseract libraries and traineddata are comparison
harness dependencies only.

For multiple inputs, text separates pages with a form feed and JSON returns
one object per input, in input order. Polygon coordinates refer to the supplied
image after EXIF correction, including when OCR corrects page rotation.
Confidence scores are model scores. Inputs are limited to 32 million pixels
and 8,192 pixels per side.

## Rust API

```rust
use std::path::Path;
use rp_ocr::{Engine, Profile};

fn main() -> anyhow::Result<()> {
    let mut engine = Engine::open_profile(Path::new("models"), Profile::Fast)?;
    let page = engine.scan_file(Path::new("scan.png"))?;
    println!("{}", page.text());
    Ok(())
}
```

`Engine::open` selects balanced. `scan_rgb` accepts an in-memory
`image::RgbImage`.

## Verify and benchmark

```sh
cargo test --locked
RUSTO_NUM_THREADS=2 RTEN_NUM_THREADS=2 cargo test --release --locked \
  --test accuracy --test offline -- --ignored --test-threads=1
cargo build --release --locked --example cpu_latency
RUSTO_NUM_THREADS=2 RTEN_NUM_THREADS=2 target/release/examples/cpu_latency \
  --profile fast benchmarks/fixtures/*.png benchmarks/fixtures/*.jpg
```

The accuracy tests compare every original development page with the frozen
balanced baseline. The offline test also checks rotation, transcript order and
source-image coordinates. Six additional synthetic scans and their accuracy
comparison are in `benchmarks/validation` and `benchmarks/extra-accuracy.json`.

[Current results and reproduction](benchmarks/OPTIMIZATION.md) ·
[Original matched comparison](benchmarks/COMPARISON.md) ·
[Earlier cross-platform experiment](benchmarks/macos-arm64.json) ·
[Tesseract configuration audit](benchmarks/tesseract-settings-audit.jsonl)

The original 14.64% Tesseract CER applied to PSM 1 with Otsu thresholding. The
configured Tesseract document pipeline scores 0.22% on the same 12 scans, so
14.64% should not be used as its general accuracy baseline.
