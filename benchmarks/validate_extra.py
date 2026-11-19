"""Check additional synthetic documents against a frozen Rust baseline binary."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from compare import edit_distance, normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-collector", type=Path, required=True)
    parser.add_argument(
        "--candidate-collector",
        type=Path,
        default=Path("target/release/examples/cpu_latency"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    args = parser.parse_args()
    images = sorted((Path(__file__).resolve().parent / "validation").glob("*.png"))
    env = dict(
        os.environ, RUSTO_NUM_THREADS="2", RAYON_NUM_THREADS="2", RTEN_NUM_THREADS="2"
    )
    report = {
        "purpose": "Accuracy validation on six additional synthetic images, including dense text, serif/monospace fonts, a footer and rotation. Not a latency benchmark.",
        "input_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in images
        },
        "runs": {},
        "collector_sha256": {},
        "candidate_profile": "fast",
        "candidate_model_sha256": {
            name: hashlib.sha256((Path("models") / name).read_bytes()).hexdigest()
            for name in ("det.rten", "rec-latin.rten", "dict-latin.txt", "orient.rten")
        },
    }
    cached = (
        json.loads(args.baseline_report.read_text()) if args.baseline_report else None
    )
    if cached:
        assert cached["input_sha256"] == report["input_sha256"], (
            "Baseline input hashes changed"
        )
        assert (
            cached["collector_sha256"]["baseline"]
            == hashlib.sha256(args.baseline_collector.read_bytes()).hexdigest()
        ), "Baseline binary changed"
    for name, collector, profile in [
        ("baseline", args.baseline_collector, "balanced"),
        ("candidate", args.candidate_collector, "fast"),
    ]:
        if name == "baseline" and cached:
            rows = cached["runs"]["baseline"]
            report["baseline_reused"] = str(args.baseline_report)
        else:
            result = subprocess.run(
                [str(collector.resolve()), "--profile", profile, *map(str, images)],
                env=env,
                text=True,
                capture_output=True,
                check=True,
                timeout=300,
            )
            rows = [json.loads(line) for line in result.stdout.splitlines()]
        pages = [row for row in rows if row["event"] == "page"]
        assert [row["image"] for row in pages] == list(map(str, images))
        for page, path in zip(pages, images, strict=True):
            expected = path.with_suffix(".txt").read_text(encoding="utf-8")
            page["reference"] = expected
            page["reference_characters"] = len(normalize(expected))
            page["character_edits"] = edit_distance(
                normalize(expected), normalize(page["text"])
            )
        report["runs"][name] = rows
        report["collector_sha256"][name] = hashlib.sha256(
            collector.read_bytes()
        ).hexdigest()
        print(name, "edits", sum(p["character_edits"] for p in pages), flush=True)
    baseline = [row for row in report["runs"]["baseline"] if row["event"] == "page"]
    candidate = [row for row in report["runs"]["candidate"] if row["event"] == "page"]
    regressions = [
        Path(old["image"]).name
        for old, new in zip(baseline, candidate, strict=True)
        if new["character_edits"] > old["character_edits"]
    ]
    report["regressions"] = regressions
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    assert not regressions, f"Accuracy regressed: {regressions}"


if __name__ == "__main__":
    main()
