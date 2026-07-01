#!/usr/bin/env python3
"""flux-eot-lab front door: point it at your audio, get a recommendation.

One command that chains the two steps the lab already exposes separately:
  1. bench.py               -- sweep the turn knobs over your clips -> raw JSONL
  2. flux_eot_lab.analysis  -- classify + emit per-class recommendations.md

Quick start (no key, no audio -- verifies wiring only):
    python recommend.py --mock

Real run (needs DEEPGRAM_API_KEY; see the README):
    python recommend.py --input audio/manifest.json --output results/

Mock numbers are wiring fixtures, never measurements. See README honesty notes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bench
from flux_eot_lab import analysis

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "audio" / "manifest.json"
DEFAULT_OUTPUT = PROJECT_DIR / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="flux-bench",
        description=(
            "Point Flux EOT Lab at your audio and get per-class threshold "
            "recommendations. Chains the sweep (bench.py) and the analysis."
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"audio manifest to sweep (default: {DEFAULT_MANIFEST})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"directory for the raw sweep JSONL (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="deterministic fake sources: no API key, no WAV files (wiring only)",
    )
    p.add_argument(
        "--class",
        dest="audio_class",
        default=None,
        help="restrict to one audio class (clean_short/clean_long/noisy_single/crosstalk)",
    )
    p.add_argument(
        "--tolerance-ms",
        type=float,
        default=200.0,
        help="classification tolerance window in ms (default 200)",
    )
    p.add_argument(
        "--operating-point-ms",
        type=float,
        default=600.0,
        help="stated detection operating point in ms (default 600)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        sys.exit(
            f"input manifest not found: {args.input}\n"
            "Generate the audio classes first (see the README), or pass --mock "
            "to verify the wiring without a key or audio."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "raw.jsonl"

    # Step 1: sweep -> raw JSONL. Reuses bench.py verbatim through its argv main,
    # so the front door can never drift from the harness it wraps.
    bench_argv = ["--manifest", str(args.input), "--out", str(raw_path)]
    if args.mock:
        bench_argv.append("--mock")
    if args.audio_class:
        bench_argv += ["--class", args.audio_class]

    print("=== step 1/2: sweep ===", flush=True)
    rc = bench.main(bench_argv)
    if rc != 0:
        return rc

    # Step 2: classify -> recommendations(_mock).md. Same analysis the README
    # tables are built from; mock runs write recommendations_mock.md and stay
    # clearly tagged as wiring, never measurements.
    print("\n=== step 2/2: analysis ===", flush=True)
    return analysis.main(
        [
            str(raw_path),
            "--tolerance-ms", str(args.tolerance_ms),
            "--operating-point-ms", str(args.operating_point_ms),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
