# CPU optimization, English and Indonesian

The `fast` profile is smaller and faster than the updated balanced profile on
the original 12 development scans, with no per-page increase in character errors
across all 18 development scans. It has **not demonstrated a reliable speed win
over configured Tesseract**. Median batch times were 19.66 vs 16.09 seconds at
two-thread settings, and 38.44 vs 27.67 seconds at one-thread settings. Large
timing variation limits these measurements.

This is an experimental English/Indonesian profile. Accuracy is checked against
the original balanced engine, and Tesseract uses its stronger document preset
rather than the earlier Otsu-only configuration.

## Implementation

- Reuse one engine and one explicitly sized RTen CPU thread pool. Share immutable
  model weights between inference sessions. Keep one recognition worker and one
  line per batch, avoiding competing recognition pools.
- Run the balanced and fast RTen models through RTen 0.28.0 with prepacked weights
  and an operator registry limited to the models' operators. Compact retains the
  earlier RustO inference path. RTen 0.28 also contains integer-operator
  correctness fixes. [RTen changelog](https://github.com/robertknight/rten/blob/main/CHANGELOG.md),
  [RTen performance guidance](https://github.com/robertknight/rten/blob/main/docs/performance.md)
- Keep full-precision detection. Integer detection introduced per-page accuracy
  regressions and was rejected. Fast uses the pinned INT8 recognition backbone.
- Derive a 692-class head from the 6,906-class recognizer. Keep blank, space,
  Latin letters/accents, numbers, punctuation, currency and other symbols, plus
  Greek mathematical letters. Repack the head's quantized weights, bias, scales
  and zero points; preserve the backbone tensors. The reader shrinks from
  2,827,104 to 2,274,112 bytes. This is a deterministic transformation of a
  pretrained model, without retraining.
- Reduce detection to a maximum side of 640 pixels only for pages at least 640
  pixels on each side with a white border and at most 24 dark text bands on
  each axis. Low-resolution, faint/shadowed-border and dense pages keep original
  detection. Recognition uses source-image crops. These heuristics are
  experimental; they cannot guarantee that small or faint text is preserved on
  other documents.
- Strip release symbols. Thin LTO and several margin/line-routing experiments
  did not provide a useful speed/size tradeoff and were removed.

The detector, recognizer and orientation weights originate from the
[pinned RustO model bundle](https://huggingface.co/byrizki/rusto-models/tree/34812c324fa258436fbcf1959eaf4ace7af6e5d4).
The library wraps the [RustO 0.4.1 pipeline](https://github.com/byrizki/rusto-rs).
There is no Tesseract, PDFium or GPU dependency in the OCR executable. Tesseract
is used only by the benchmark harness.

## Accuracy and limits

The original 12 synthetic images represent five source documents with rotations
and degradations, totaling 3,184 normalized reference characters. Whitespace is
collapsed and text is normalized to NFC; case, punctuation and order remain
significant. This same set was used during optimization and Tesseract-preset
development, so it is development data, not an independent accuracy test.

The standalone language-head probe preserved the full-alphabet INT8 transcripts
exactly on all 18 development images before changing detector resolution.
[Pruning check](experiments/latin-head-validation.json)

Six additional synthetic images cover a serif agreement with a footer, an
Indonesian purchase order and a dense monospace register, each upright and
rotated. They contain 2,514 reference characters. They also became development
data when their results informed the dense-page fallback. No candidate page may
have more character edits than its frozen original balanced baseline.
[Additional images and references](validation/), [complete output](extra-accuracy.json)

The additional set has 478 edits (19.01% CER) in the original balanced baseline
and 474 (18.85% CER) in fast. Every page is equal or better by character edit
count; the agreement improves 3→2 edits per image and the register 92→91 and
380→379. The purchase orders remain exact.

The register exposes a large reading-order limitation: columns and row fields
are interleaved incorrectly. A low CER on the original 12 images does not
represent this layout or establish accuracy on production scans. No guarantee
of equal accuracy across real English/Indonesian documents has been established.
Other alphabetic scripts need balanced or compact.

## Measurement method

Three fresh-process trials per engine and thread setting reuse one initialized
engine per ordered 12-image batch. Execution order is shuffled with a fixed seed.
Both engines receive identical image bytes and matching thread-budget environment
settings on native macOS ARM64. The Mac is not CPU-pinned or an isolated benchmark
host; ranges and process CPU time should be considered with wall time.

Batch time includes process startup, hash verification, model loading, image
reading, preprocessing, OCR and output. Per-page latency excludes model loading.
Peak RSS includes the collector: compiled Rust for RP OCR and Python/ctypes for
Tesseract. Model sizes exclude executable/shared-library size. The native release
CLI uses symbol stripping and no LTO, compiled with Rust 1.96.1.

Tesseract 5.5.4 and Leptonica 1.87.0 are from the
[`tesserocr==2.13.0` native wheel](https://pypi.org/project/tesserocr/2.13.0/).
The fixed document preset applies the same rules to every image: conditional
0.5/99.5 percentile contrast stretching, upscaling short inputs to 1,000 pixels,
Leptonica deskew, 300 DPI, eng+ind, PSM 1 and Sauvola thresholding. All preparation
time is counted. This is the stronger 0.22% CER configuration documented in the
[original matched comparison](COMPARISON.md), not the 14.64% CER Otsu preset.
[Tesseract image-quality guidance](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)

## Native macOS ARM64, two-thread settings

| Engine | Median 12-page batch | Range (3 trials) | Median page | CER | Median peak RSS | Median process CPU time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rust-balanced | 34.02 s | 27.39–37.55 s | 2792 ms | 0.13% | 227.0 MiB | 16.58 CPU-s |
| rust-fast | 19.66 s | 12.96–24.30 s | 1625 ms | 0.09% | 164.3 MiB | 9.40 CPU-s |
| tesseract-document-contrast | 16.09 s | 9.17–16.48 s | 909 ms | 0.22% | 141.9 MiB | 9.49 CPU-s |

Fast is 1.73× faster than the updated balanced profile by the observed median,
but takes 22% longer than configured Tesseract under these two-thread settings.
Ranges overlap and the measurements do not establish a stable latency guarantee.
Each trial has 4, 3 and 7 character edits respectively, in 3,184 reference
characters. Fast never increases the original balanced baseline edit count on
any of the 12 pages.

[Raw two-thread runs, hashes and transcripts](cpu-optimized-macos-arm64.json)

## Native macOS ARM64, one-thread settings

| Engine | Median 12-page batch | Range (3 trials) | Median page | CER | Median process CPU time |
| --- | ---: | ---: | ---: | ---: | ---: |
| rust-fast | 38.44 s | 16.47–44.89 s | 2204 ms | 0.09% | 8.93 CPU-s |
| tesseract-document-contrast | 27.67 s | 26.89–30.83 s | 2025 ms | 0.22% | 9.35 CPU-s |

[Raw one-thread runs, hashes and transcripts](cpu-optimized-single-macos-arm64.json)

The fast profile has not demonstrated a reliable speed advantage over configured
Tesseract at either tested thread setting. The optimization target remains open.


## Footprint

| Component | Size |
| --- | ---: |
| Balanced model bundle | 12.49 MiB |
| Fast model bundle | 10.36 MiB (10,865,788 bytes) |
| Tesseract eng + ind + osd models | 15.07 MiB |
| Native stripped RP OCR CLI | 4.88 MiB (5,112,072 bytes) |
| Native fast CLI + required model files | 15.24 MiB |

Fast reduces model storage by 17% versus balanced and 31% versus these Tesseract
model files. Its measured process RSS is lower than balanced and higher than
Tesseract. This does not claim a smaller total Tesseract installation: its native
library sizes are excluded. `otool -L` lists only system `libSystem` and
`libiconv` dependencies for the native OCR executable.


## Offline Linux verification

The new Linux ARM64 CLI and collector compile as stripped, statically linked
musl executables. The CLI is 5,104,296 bytes (4.87 MiB).

**The final offline Linux smoke remains unverified.** A container run restricted
to two CPUs and 2 GiB, with networking disabled and only the four runtime models
mounted, exceeded its 300-second client timeout without any OCR events. Docker's
local `/_ping` returned `OK`, but its container-list API also timed out after 10
seconds without response bytes; `docker ps` stalled as well. This does not isolate
an OCR-engine timeout. Docker was not restarted, and container creation/completion
could not be confirmed. [Failure record and binary/model hashes](optimized-linux-smoke.json)

All six focused native Rust tests, Clippy with warnings denied, formatting,
Python lint and the fresh-cache model-derivation check passed. The stripped native
CLI also read an Indonesian page using only the four fast runtime model files.
The current feature branch is retained for review; the Docker smoke gate has not
passed, so this iteration is not merged into main.

The old balanced Linux batch also exceeded a 300-second harness timeout. Its
cause remains unestablished. Earlier successful Linux checks describe their
older builds, not this final English/Indonesian profile. RPPDF-12 retains the
Linux verification and timeout diagnostic work.

## Reproduce

```sh
cargo build --release --locked --example cpu_latency
cargo test --release --locked --test accuracy --test offline -- --ignored --test-threads=1
python3 benchmarks/compare_cpu.py \
  --tessdata models/tessdata \
  --tesseract-library /path/to/libtesseract.5.5.4.dylib \
  --leptonica-library /path/to/libleptonica.6.0.0.dylib \
  --engines rust-balanced rust-fast tesseract-document-contrast \
  --trials 3 --threads 2 --output /tmp/ocr-cpu.json
```

Repeat with `--threads 1` for the single-thread comparison. The JSON records
input/model/source/binary hashes, commands, all transcripts, errors, loading
latency, CPU time, wall time and peak RSS. Keep other builds and benchmarks idle
while measuring. To validate the additional scans, build the frozen baseline
collector from commit `8950a42` and use `benchmarks/validate_extra.py` with its
`--baseline-collector`, `--candidate-collector` and `--output` options.

Earlier attempts remain in [experiments/](experiments/README.md) and
[int8-variants.json](int8-variants.json). Their timings were collected with
different implementations or uncontrolled overlap and must not be pooled with
the current matched runs. Both successful and unsuccessful attempts are retained.
