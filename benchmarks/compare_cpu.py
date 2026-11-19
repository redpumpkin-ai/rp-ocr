"""Compare persistent Rust and Tesseract engines on one CPU environment."""

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from compare import edit_distance, normalize

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def worker(command):
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=300, check=False
        )
        status = "ok" if completed.returncode == 0 else "error"
        text = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        status = "timeout"
        text = (error.stdout or b"").decode("utf-8", errors="replace")
        stderr = (error.stderr or b"").decode("utf-8", errors="replace")
        returncode = None
    elapsed = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    rss_scale = 1 if platform.system() == "Darwin" else 1024
    return {
        "status": status,
        "returncode": returncode,
        "total_seconds": elapsed,
        "peak_rss_bytes": usage.ru_maxrss * rss_scale,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "stdout": text,
        "stderr": stderr,
    }


def measure(command, images, env):
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            json.dumps(command),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=330,
        env=env,
    )
    result = json.loads(completed.stdout)
    result["command"] = command
    events = [json.loads(line) for line in result.pop("stdout").splitlines()]
    result["model_load"] = next(
        (row for row in events if row["event"] == "model_load"), None
    )
    result["pages"] = [row for row in events if row["event"] == "page"]
    if result["status"] != "ok":
        return result
    if [row["image"] for row in result["pages"]] != [str(path) for path in images]:
        raise ValueError("OCR output page identities differ from the input sequence")
    for page, image in zip(result["pages"], images, strict=True):
        expected = normalize(image.with_suffix(".txt").read_text(encoding="utf-8"))
        actual = normalize(page["text"])
        page["reference"] = image.with_suffix(".txt").read_text(encoding="utf-8")
        page["reference_characters"] = len(expected)
        page["character_edits"] = edit_distance(expected, actual)
        page["word_edits"] = edit_distance(expected.split(), actual.split())
    return result


def summary(runs):
    valid = [run for run in runs if run["status"] == "ok"]
    if not valid:
        return {"completed_trials": 0, "failures": len(runs)}
    pages = [page for run in valid for page in run["pages"]]
    characters = sum(page["reference_characters"] for page in pages)
    errors = sum(page["character_edits"] for page in pages)
    latencies = [page["milliseconds"] for page in pages]
    return {
        "completed_trials": len(valid),
        "failures": len(runs) - len(valid),
        "median_batch_seconds": statistics.median(
            run["total_seconds"] for run in valid
        ),
        "batch_seconds": [run["total_seconds"] for run in valid],
        "median_page_ms": statistics.median(latencies),
        "p95_page_ms": statistics.quantiles(latencies, n=20, method="inclusive")[18],
        "median_first_page_ms": statistics.median(
            run["pages"][0]["milliseconds"] for run in valid
        ),
        "median_model_load_ms": statistics.median(
            run["model_load"]["milliseconds"] for run in valid
        ),
        "median_peak_rss_bytes": statistics.median(
            run["peak_rss_bytes"] for run in valid
        ),
        "cer": errors / characters,
        "character_edits_per_trial": [
            sum(page["character_edits"] for page in run["pages"]) for run in valid
        ],
        "reference_characters_per_trial": sum(
            page["reference_characters"] for page in valid[0]["pages"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rust-collector",
        type=Path,
        default=Path("target/release/examples/cpu_latency"),
    )
    parser.add_argument(
        "--rust-engine", type=Path, default=Path("target/release/rp-ocr")
    )
    parser.add_argument("--baseline-collector", type=Path)
    parser.add_argument(
        "--tessdata", type=Path, default=Path("/usr/share/tesseract-ocr/5/tessdata")
    )
    parser.add_argument("--tesseract-library", type=Path)
    parser.add_argument("--leptonica-library", type=Path)
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=[
            "rust-balanced",
            "rust-fast",
            "rust-compact",
            "rust-baseline",
            "tesseract-otsu",
            "tesseract-sauvola",
            "tesseract-document",
            "tesseract-document-contrast",
        ],
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or args.threads < 1:
        parser.error("trials and threads must be positive")
    images = sorted(
        path for path in FIXTURES.iterdir() if path.suffix in {".jpg", ".png"}
    )
    env = {
        **os.environ,
        "OMP_THREAD_LIMIT": str(args.threads),
        "OMP_NUM_THREADS": str(args.threads),
        "RUSTO_NUM_THREADS": str(args.threads),
        "RTEN_NUM_THREADS": str(args.threads),
        "RAYON_NUM_THREADS": str(args.threads),
        "TESSDATA_PREFIX": str(args.tessdata),
    }
    library_arguments = []
    for flag, path in [
        ("--tesseract-library", args.tesseract_library),
        ("--leptonica-library", args.leptonica_library),
    ]:
        if path:
            library_arguments.extend([flag, str(path.resolve())])
    commands = {
        **{
            f"rust-{profile}": [
                str(args.rust_collector.resolve()),
                "--profile",
                profile,
                *map(str, images),
            ]
            for profile in ("balanced", "fast", "compact")
        },
        **{
            f"tesseract-{preset}": [
                sys.executable,
                str(FIXTURES.parent / "tesseract_api_latency.py"),
                "--preset",
                preset,
                *library_arguments,
                *map(str, images),
            ]
            for preset in ("otsu", "sauvola", "document", "document-contrast")
        },
    }
    if args.baseline_collector:
        commands["rust-baseline"] = [
            str(args.baseline_collector.resolve()),
            "--profile",
            "balanced",
            *map(str, images),
        ]
    if args.engines:
        if "rust-baseline" in args.engines and not args.baseline_collector:
            parser.error("rust-baseline requires --baseline-collector")
        commands = {name: commands[name] for name in args.engines}
    names = list(commands)
    rng = random.Random(20261119)
    runs = {name: [] for name in names}
    orders = []
    for trial in range(args.trials):
        order = names.copy()
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            if any(run["status"] != "ok" for run in runs[name]):
                continue
            result = measure(commands[name], images, env)
            runs[name].append(result)
            print(
                f"Trial {trial + 1}/{args.trials} {name}: {result['status']}, "
                f"{result['total_seconds']:.2f}s",
                file=sys.stderr,
                flush=True,
            )
    model_files = {
        "rust-balanced": [
            Path("models") / name
            for name in ("det.rten", "rec.rten", "dict.txt", "orient.rten")
        ],
        "rust-compact": [
            Path("models") / name
            for name in (
                "det-compact.mnn",
                "rec-compact.mnn",
                "dict.txt",
                "orient.rten",
            )
        ],
        "rust-fast": [
            Path("models") / name
            for name in (
                "det.rten",
                "rec-latin.rten",
                "dict-latin.txt",
                "orient.rten",
            )
        ],
        "tesseract": [
            args.tessdata / (name + ".traineddata") for name in ("eng", "ind", "osd")
        ],
    }
    required_models = {
        "rust-balanced"
        if name == "rust-baseline"
        else "tesseract"
        if name.startswith("tesseract-")
        else name
        for name in commands
    }
    model_files = {
        name: files for name, files in model_files.items() if name in required_models
    }
    report = {
        "system": platform.system(),
        "machine": platform.machine(),
        "rust_compiler": subprocess.check_output(
            ["rustc", "--version"], text=True
        ).strip(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path("Cargo.toml"),
                Path("Cargo.lock"),
                *sorted(Path("src").rglob("*.rs")),
            ]
        },
        "fast_profile_settings": {
            "recognizer": "INT8, 692 English/Indonesian/Latin classes",
            "detector": "FP32",
            "batch_size": 1,
            "detector_resolution": "max 640 for white-border sparse pages >=640px; original otherwise",
        },
        "thread_limit": args.threads,
        "trials_requested": args.trials,
        "execution_orders": orders,
        "methodology": "Fresh process and one initialized engine per trial. Same ordered image bytes. "
        "Per-page time includes reading and preprocessing. Total batch includes process startup, "
        "model loading, and OCR. One fixed preset per run, without per-page oracle selection. These presets were developed on the same synthetic inputs. No GPU. "
        "Rust uses the public library through cpu_latency. Tesseract uses its C API through ctypes. "
        "Tesseract process RSS includes the Python ctypes driver. "
        "A failed or timed-out configuration is retained and skipped in later trials.",
        "presets": {
            "tesseract-document-contrast": "Document preset plus 0.5/99.5 percentile luminance stretching when the range spans 12..105 levels, using the contrast-restoration strategy of the Rust engine.",
            "tesseract-otsu": "eng+ind, automatic orientation and segmentation (PSM 1), Otsu",
            "tesseract-sauvola": "eng+ind, PSM 1, Sauvola (thresholding_method=2)",
            "tesseract-document": "eng+ind, PSM 1, Sauvola. Before OCR, upscale images below 1000px on "
            "the longest side to 1000px, estimate and correct skew within +/-15 degrees "
            "using Leptonica with confidence >= 2 and angle magnitude >= 0.5 degrees, and set resolution to 300dpi. "
            "The identical rule is applied to every image and its time is included.",
        },
        "model_bytes": {
            name: sum(path.stat().st_size for path in files)
            for name, files in model_files.items()
        },
        "model_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for files in model_files.values()
            for path in files
        },
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in images
        },
        "tesseract_library": str(args.tesseract_library)
        if args.tesseract_library
        else None,
        "leptonica_library": str(args.leptonica_library)
        if args.leptonica_library
        else None,
        "rust_version": subprocess.check_output(
            [str(args.rust_engine.resolve()), "--version"], text=True
        ).strip(),
        "collector_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [args.rust_collector, args.baseline_collector]
            if path is not None
        },
        "tesseract_cli_version": subprocess.check_output(
            ["tesseract", "--version"], text=True
        ).splitlines()[0]
        if shutil.which("tesseract")
        else None,
        "summary": {name: summary(data) for name, data in runs.items()},
        "runs": runs,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output == Path("-"):
        sys.stdout.write(encoded)
    else:
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        print(json.dumps(worker(json.loads(sys.argv[2]))))
    else:
        main()
