"""Compare the same image batch across OCR engines. Uses the Python standard library."""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def run(command, threads):
    env = {
        **os.environ,
        "OMP_THREAD_LIMIT": str(threads),
        "RAYON_NUM_THREADS": str(threads),
    }
    start = time.perf_counter()
    completed = subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=300, env=env
    )
    return {
        "seconds": round(time.perf_counter() - start, 4),
        "text": completed.stdout,
        "stderr": completed.stderr,
    }


def page_outputs(text, expected_pages):
    pages = text.split("\f")
    if len(pages) == expected_pages + 1 and not pages[-1].strip():
        pages.pop()
    if len(pages) != expected_pages:
        raise ValueError(f"Expected {expected_pages} OCR pages, received {len(pages)}")
    return pages


def measure(samples, commands, threads):
    cases = [
        {
            "image": image.name,
            "reference": image.with_suffix(".txt").read_text(encoding="utf-8"),
            "engines": {},
        }
        for image in samples
    ]
    batches = {}
    for name, command in commands.items():
        batch = run(command, threads)
        pages = page_outputs(batch.pop("text"), len(samples))
        batches[name] = batch
        print(f"{name} completed in {batch['seconds']:.2f}s", file=sys.stderr, flush=True)
        for case, prediction in zip(cases, pages, strict=True):
            expected = normalize(case["reference"])
            actual = normalize(prediction)
            case["engines"][name] = {
                "text": prediction,
                "character_edits": edit_distance(expected, actual),
                "word_edits": edit_distance(expected.split(), actual.split()),
            }
    return cases, batches


def summarize(cases, engines):
    expected = [normalize(case["reference"]) for case in cases]
    character_count = sum(map(len, expected))
    word_count = sum(len(text.split()) for text in expected)
    return {
        engine: {
            "character_error_rate": sum(
                case["engines"][engine]["character_edits"] for case in cases
            )
            / character_count,
            "word_error_rate": sum(
                case["engines"][engine]["word_edits"] for case in cases
            )
            / word_count,
        }
        for engine in engines
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="target/release/rp-ocr")
    parser.add_argument("--models", default="models")
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--tess-lang", default="eng+ind")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--only", choices=("rp-ocr", "tesseract"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")

    versions = {}
    if args.only != "tesseract":
        versions["rp-ocr"] = run([args.engine, "--version"], args.threads)["text"].strip()
    if args.only != "rp-ocr":
        languages = run([args.tesseract, "--list-langs"], args.threads)["text"].splitlines()[1:]
        required = set(args.tess_lang.split("+")) | {"osd"}
        if missing := required.difference(languages):
            parser.error(f"missing Tesseract data files: {', '.join(sorted(missing))}")
        versions["tesseract"] = run([args.tesseract, "--version"], args.threads)["text"].splitlines()[0]

    samples = sorted(
        path for path in FIXTURES.iterdir() if path.suffix in {".jpg", ".png"}
    )
    images = [str(path) for path in samples]
    with tempfile.TemporaryDirectory(prefix="rp-ocr-benchmark-") as temp:
        filelist = Path(temp) / "inputs.txt"
        filelist.write_text("\n".join(images) + "\n", encoding="utf-8")
        commands = {
            "rp-ocr": [args.engine, "--models", args.models, "scan", *images],
            **{
                f"tesseract-psm{mode}": [
                    args.tesseract, str(filelist), "stdout", "-l", args.tess_lang,
                    "--psm", str(mode),
                ]
                for mode in (3, 1)
            },
        }
        if args.only:
            commands = {
                name: command for name, command in commands.items()
                if name.startswith(args.only)
            }
        cases, batches = measure(samples, commands, args.threads)

    rotations = [case for case in cases if case["image"].startswith("rotated")]
    other = [case for case in cases if case not in rotations]
    report = {
        "machine": platform.machine(),
        "system": platform.system(),
        "versions": versions,
        "tesseract_languages": args.tess_lang if args.only != "rp-ocr" else None,
        "thread_limit": args.threads,
        "batch_runs": batches,
        "all_images": summarize(cases, commands),
        "rotated": summarize(rotations, commands),
        "other_images": summarize(other, commands),
        "cases": cases,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output == Path("-"):
        sys.stdout.write(encoded)
    else:
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
