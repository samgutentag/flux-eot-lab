#!/usr/bin/env python3
"""Flux EOT Lab benchmark harness.

Replays the four audio classes through the instrumented Flux agent, sweeps the
real turn knobs (eot_threshold / eager_eot_threshold / eot_timeout_ms), and
writes RAW per-turn records to JSONL. Classification and tolerance are applied
later by flux_eot_lab.analysis, NOT here.

Run without any API key or audio files:
    python bench.py --mock

Real run (needs DEEPGRAM_API_KEY, see the README):
    python bench.py

Mock numbers are wiring fixtures only. Every mock record is flagged mock=True;
they must never be reported as measured. See README "why now" / honesty notes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from flux_eot_lab.agent import InstrumentedFluxAgent
from flux_eot_lab.config import (
    AUDIO_CLASSES,
    DEFAULT_FLUX_MODEL,
    load_env,
    sweep_grid,
)
from flux_eot_lab.records import ClipSpec, write_jsonl

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "audio" / "manifest.json"


def _load_dotenv() -> None:
    """Load .env with a zero-dependency fallback and a warning on failure."""
    load_env(PROJECT_DIR / ".env")


def load_clips(manifest_path: Path) -> list[ClipSpec]:
    """Resolve audio/manifest.json into a list of ClipSpec via from_sidecar.

    Accepts a manifest that is either a JSON list or an object with a "clips"
    key. Each entry may be a bare sidecar path string or an object carrying a
    "sidecar"/"path" field. Paths resolve relative to the manifest directory.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found: {manifest_path}\n"
            "Generate or provide the audio classes first (see the README "
            "and audio/spec.md)."
        )

    with manifest_path.open() as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        entries = data.get("clips", [])
    else:
        entries = data

    base = manifest_path.parent
    clips: list[ClipSpec] = []
    for entry in entries:
        if isinstance(entry, str):
            rel = entry
        elif isinstance(entry, dict):
            rel = entry.get("sidecar") or entry.get("path")
            if rel is None:
                raise ValueError(f"manifest entry missing sidecar/path: {entry!r}")
        else:
            raise ValueError(f"unsupported manifest entry: {entry!r}")

        sidecar_path = (base / rel).resolve()
        clips.append(ClipSpec.from_sidecar(sidecar_path))

    return clips


def build_sources(use_mock: bool):
    """Return a Sources bundle: deterministic mocks, or real API-backed clients."""
    if use_mock:
        from flux_eot_lab.mock import build_mock_sources

        return build_mock_sources()

    from flux_eot_lab.clients import build_real_sources

    dg_key = os.getenv("DEEPGRAM_API_KEY")
    if not dg_key:
        sys.exit(
            "Missing required API key: DEEPGRAM_API_KEY\n"
            "Set it in your environment or .env, or run with --mock.\n"
            "See the README for the full checklist."
        )

    return build_real_sources(dg_key=dg_key, flux_model=DEFAULT_FLUX_MODEL)


def run_ping(count: int = 15) -> int:
    """--ping: sample Deepgram round-trip latency `count` times and report the
    distribution, then exit. Needs a real key. A single ping is noisy, so the
    median and p95 over several pings is what belongs in the README disclosure."""
    dg_key = os.getenv("DEEPGRAM_API_KEY")
    if not dg_key:
        sys.exit(
            "--ping needs DEEPGRAM_API_KEY (it hits the real Deepgram endpoint).\n"
            "See the README."
        )
    from flux_eot_lab.clients import measure_rtt
    from flux_eot_lab.analysis import _percentile

    samples_ms: list[float] = []
    for i in range(count):
        rtt_ms = measure_rtt(dg_key) * 1000.0
        samples_ms.append(rtt_ms)
        print(f"  ping {i + 1}/{count}: {rtt_ms:.1f} ms", flush=True)

    median = _percentile(samples_ms, 50)
    p95 = _percentile(samples_ms, 95)
    print(
        f"\nDeepgram RTT over {count} pings: median {median:.0f} ms, "
        f"p95 {p95:.0f} ms (min {min(samples_ms):.0f} ms, max {max(samples_ms):.0f} ms)"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench.py",
        description="Flux EOT latency-vs-false-cutoff sweep harness.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="run with deterministic fake sources (no API key, no WAV files)",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="add finer sweep steps near the knee (0.6-0.8)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSONL path (default: results/raw_<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--class",
        dest="audio_class",
        default=None,
        help=(
            "restrict the run to a single audio class (a built-in one or a "
            f"bring-your-own condition; built-ins are {', '.join(AUDIO_CLASSES)})"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"audio manifest path (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="sample Deepgram round-trip latency and exit",
    )
    parser.add_argument(
        "--ping-count",
        type=int,
        default=15,
        help="number of pings to sample for the RTT distribution (default 15)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _load_dotenv()

    if args.ping:
        return run_ping(args.ping_count)

    clips = load_clips(args.manifest)
    if args.audio_class:
        clips = [c for c in clips if c.audio_class == args.audio_class]
        if not clips:
            sys.exit(f"no clips for audio class {args.audio_class!r} in {args.manifest}")

    # Fail fast (once) if a real run is missing its WAVs, e.g. a fresh clone where the
    # gitignored audio has not been regenerated yet. Mock never reads audio, so skip it.
    if not args.mock:
        missing = [c.clip_id for c in clips if c.audio_file and not Path(c.audio_file).exists()]
        if missing:
            sys.exit(
                f"audio not found for {len(missing)} clip(s) (e.g. {missing[0]}).\n"
                "WAVs are gitignored, so generate them first:\n"
                "    python audio/generate_audio.py\n"
                "or pass --mock to run on the committed sidecars with no audio and no key."
            )

    points = sweep_grid(refine=args.refine)

    out_path = args.out
    if out_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_DIR / "results" / f"raw_{ts}.jsonl"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sources = build_sources(args.mock)
    agent = InstrumentedFluxAgent(sources)

    mode = "MOCK" if args.mock else "REAL"
    total_steps = len(clips) * len(points)
    print(
        f"[{mode}] {len(clips)} clip(s) x {len(points)} sweep point(s) "
        f"= {total_steps} run(s) -> {out_path}"
    )
    if args.mock:
        print("[MOCK] synthetic timing - wiring fixture only, NOT measured data.")

    records = []
    step = 0
    for clip in clips:
        for point in points:
            step += 1
            eager = (
                "off"
                if point.eager_eot_threshold is None
                else f"{point.eager_eot_threshold:.2f}"
            )
            print(
                f"[{step}/{total_steps}] {clip.clip_id} "
                f"({clip.audio_class}) eot={point.eot_threshold:.2f} "
                f"eager={eager} timeout={point.eot_timeout_ms}ms",
                flush=True,
            )
            try:
                turn_records = agent.run_clip(clip, point)
            except Exception as exc:  # keep the sweep going; surface the failure
                print(f"    ! run_clip failed: {exc}", file=sys.stderr, flush=True)
                continue
            records.extend(turn_records)

    write_jsonl(out_path, records)
    print(f"Wrote {len(records)} turn record(s) to {out_path}")
    if not args.mock:
        print(
            "Next: python -m flux_eot_lab.analysis "
            f"{out_path} --tolerance-ms 200 --operating-point-ms 600"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
