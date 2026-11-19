"""Compare cold CLI runs on the same images. Uses only the Python standard library."""

import argparse
import json
import os
import platform
import statistics
import subprocess
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
        command, capture_output=True, text=True, check=True, timeout=90, env=env
    )
    return {
        "seconds": round(time.perf_counter() - start, 4),
        "text": completed.stdout,
        "stderr": completed.stderr,
    }


def measure(samples, trials, commands, threads):
    results = []
    for image in samples:
        expected = image.with_suffix(".txt").read_text(encoding="utf-8")
        reference = normalize(expected)
        case = {"image": image.name, "reference": expected, "engines": {}}
        for trial in range(trials):
            order = list(commands)
            if trial % 2:
                order.reverse()
            for name in order:
                sample = run(commands[name](image), threads)
                actual = normalize(sample["text"])
                sample["character_edits"] = edit_distance(reference, actual)
                sample["word_edits"] = edit_distance(reference.split(), actual.split())
                case["engines"].setdefault(name, []).append(sample)
        results.append(case)
    return results


def summarize(cases, engines):
    expected = [normalize(case["reference"]) for case in cases]
    character_count = sum(map(len, expected))
    word_count = sum(len(text.split()) for text in expected)
    summary = {}
    for engine in engines:
        timings = [
            trial["seconds"] for case in cases for trial in case["engines"][engine]
        ]
        errors = [case["engines"][engine][0] for case in cases]
        summary[engine] = {
            "character_error_rate": sum(row["character_edits"] for row in errors)
            / character_count,
            "word_error_rate": sum(row["word_edits"] for row in errors) / word_count,
            "median_cold_seconds": statistics.median(timings),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="target/release/rp-ocr")
    parser.add_argument("--models", default="models")
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--tess-lang", default="eng+ind")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or args.threads < 1:
        parser.error("trials and threads must be positive")

    def tesseract(image, psm):
        return [
            args.tesseract, str(image), "stdout",
            "-l", args.tess_lang, "--psm", str(psm),
        ]

    commands = {
        "rp-ocr": lambda image: [
            args.engine, "--models", args.models, "scan", str(image)
        ],
        "tesseract-psm3": lambda image: tesseract(image, 3),
        "tesseract-psm1": lambda image: tesseract(image, 1),
    }
    samples = sorted(
        path for path in FIXTURES.iterdir() if path.suffix in {".jpg", ".png"}
    )
    cases = measure(samples, args.trials, commands, args.threads)
    rotations = [case for case in cases if case["image"].startswith("rotated")]
    upright = [case for case in cases if case not in rotations]
    report = {
        "machine": platform.machine(),
        "system": platform.system(),
        "rp_ocr_version": run([args.engine, "--version"], args.threads)["text"].strip(),
        "tesseract_version": run([args.tesseract, "--version"], args.threads)["text"]
        .splitlines()[0],
        "tesseract_languages": args.tess_lang,
        "thread_limit": args.threads,
        "trials_per_image": args.trials,
        "all_images": summarize(cases, commands),
        "rotated": summarize(rotations, commands),
        "other_images": summarize(upright, commands),
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
