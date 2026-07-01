#!/usr/bin/env python3
"""End-to-end smoke test for the mock path. No API keys, no WAV files.

Runs the whole wiring without a real key:
  1. bench.py --mock  -> results/raw_mock.jsonl
  2. records.read_jsonl round-trips the file
  3. every TurnRecord.mock is True (honesty guardrail)
  4. across the run all four classify() categories appear
     (clean, hard_cutoff, near_miss, late)
  5. recommend() yields a row per audio class, each with a band assigned
  6. the analysis CLI writes results/recommendations_mock.md containing a table

Exit 0 on success, nonzero with a clear message on the first failure.

Run it:  python tests/smoke_test.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# --- locate project root and make the package importable ------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
RAW_PATH = RESULTS_DIR / "raw_mock.jsonl"
# Mock analysis writes to a SEPARATE path so the smoke test can never clobber the
# committed real recommendations.md (analysis routes mock output to *_mock.md).
RECS_PATH = RESULTS_DIR / "recommendations_mock.md"

BAND_VALUES = {"green", "yellow", "red"}
REQUIRED_CATEGORIES = {"clean", "hard_cutoff", "near_miss", "late"}


class SmokeFailure(Exception):
    """A smoke-test assertion failed, with a human-readable message."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _import_lab():
    """Import the lab modules, turning missing-dependency builds into a clear error."""
    try:
        from flux_eot_lab import config, records, analysis
    except ImportError as exc:  # a sibling builder's module is missing/broken
        raise SmokeFailure(
            "could not import flux_eot_lab modules "
            "(config / records / analysis). This smoke test depends on the "
            f"full build being present. Underlying import error: {exc}"
        ) from exc
    return config, records, analysis


def _extract_band(entry) -> str | None:
    """Pull a band string out of a recommendation entry, tolerant of shape.

    recommend() returns a dict keyed by audio class. Each value may be a plain
    band string, a dict carrying a "band" key, or a dataclass with a .band attr.
    """
    if entry is None:
        return None
    if isinstance(entry, str):
        return entry if entry in BAND_VALUES else None
    if isinstance(entry, dict):
        band = entry.get("band")
        return band if isinstance(band, str) else None
    band = getattr(entry, "band", None)
    return band if isinstance(band, str) else None


def run_bench_mock() -> None:
    """Invoke the real bench.py CLI in --mock mode as a subprocess."""
    # Clear stale artifacts so a prior run can never produce a false pass.
    for stale in (RAW_PATH, RECS_PATH):
        if stale.exists():
            stale.unlink()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "bench.py",
        "--mock",
        "--out",
        str(RAW_PATH),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    check(
        proc.returncode == 0,
        "bench.py --mock exited nonzero "
        f"(code {proc.returncode}).\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
    )
    check(
        RAW_PATH.exists() and RAW_PATH.stat().st_size > 0,
        f"bench.py --mock did not produce a non-empty {RAW_PATH}",
    )


def run_analysis_cli() -> None:
    """Invoke the analysis CLI so it writes results/recommendations_mock.md."""
    cmd = [
        sys.executable,
        "-m",
        "flux_eot_lab.analysis",
        str(RAW_PATH),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    check(
        proc.returncode == 0,
        "flux_eot_lab.analysis exited nonzero "
        f"(code {proc.returncode}).\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
    )


def main() -> int:
    print("Flux EOT Lab :: mock smoke test")
    print(f"  project root: {PROJECT_ROOT}")

    config, records, analysis = _import_lab()

    # 1. run the mock harness end to end via the real CLI -------------------
    print("[1/6] running bench.py --mock ...")
    run_bench_mock()
    print(f"      wrote {RAW_PATH}")

    # 2. read it back; assert the JSONL round-trips -------------------------
    print("[2/6] reading raw records back ...")
    recs = records.read_jsonl(RAW_PATH)
    check(len(recs) > 0, f"{RAW_PATH} read back as zero records")
    print(f"      {len(recs)} TurnRecord rows")

    # 3. every mock record is flagged mock=True (honesty guardrail) ---------
    print("[3/6] checking mock flag on every record ...")
    not_mock = [
        (r.clip_id, r.turn_index)
        for r in recs
        if getattr(r, "mock", False) is not True
    ]
    check(
        not not_mock,
        "every TurnRecord from --mock must have mock=True; "
        f"{len(not_mock)} did not, e.g. {not_mock[:3]}",
    )

    # 4. all four classification categories appear across the run ----------
    print("[4/6] checking all four classify() categories appear ...")
    tol_ms = config.DEFAULT_TOLERANCE_MS
    seen: dict[str, int] = {}
    for r in recs:
        label = analysis.classify(r, tol_ms)
        seen[label] = seen.get(label, 0) + 1
    missing = REQUIRED_CATEGORIES - set(seen)
    check(
        not missing,
        "mock run did not exercise every classification category at "
        f"tolerance {tol_ms}ms. missing: {sorted(missing)}. "
        f"seen: {seen}",
    )
    print(f"      categories: {seen}")

    # 5. recommend() -> a row per audio class, each with a band ------------
    print("[5/6] checking per-class recommendations + bands ...")
    op_ms = config.DEFAULT_OPERATING_POINT_MS
    analyzed = analysis.analyze(recs, tol_ms, op_ms)
    recs_by_class = analysis.recommend(analyzed, op_ms)
    check(
        isinstance(recs_by_class, dict),
        f"recommend() must return a dict keyed by audio class, got {type(recs_by_class)}",
    )
    for audio_class in config.AUDIO_CLASSES:
        check(
            audio_class in recs_by_class,
            f"recommend() missing a row for audio class '{audio_class}'. "
            f"got keys: {sorted(recs_by_class)}",
        )
        band = _extract_band(recs_by_class[audio_class])
        check(
            band in BAND_VALUES,
            f"recommendation for '{audio_class}' has no valid band "
            f"(expected one of {sorted(BAND_VALUES)}), got entry: "
            f"{recs_by_class[audio_class]!r}",
        )
    print(f"      bands: " + ", ".join(
        f"{c}={_extract_band(recs_by_class[c])}" for c in config.AUDIO_CLASSES
    ))

    # 6. analysis CLI writes recommendations.md with a table ---------------
    print("[6/6] checking results/recommendations_mock.md is written w/ a table ...")
    run_analysis_cli()
    check(
        RECS_PATH.exists(),
        f"analysis CLI did not write {RECS_PATH}",
    )
    md = RECS_PATH.read_text(encoding="utf-8")
    # a markdown table needs a header separator row of pipes and dashes
    has_table = any(
        line.count("|") >= 2 and set(line.replace("|", "").strip()) <= set("-: ")
        and "-" in line
        for line in md.splitlines()
    )
    check(
        has_table,
        f"{RECS_PATH} exists but contains no markdown table "
        "(no `|---|` separator row found)",
    )
    print(f"      {RECS_PATH} OK ({len(md)} chars)")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeFailure as failure:
        print(f"\nSMOKE TEST FAILED: {failure}", file=sys.stderr)
        sys.exit(1)
    except Exception as unexpected:  # noqa: BLE001 - surface anything cleanly
        print(
            f"\nSMOKE TEST FAILED (unexpected error): "
            f"{type(unexpected).__name__}: {unexpected}",
            file=sys.stderr,
        )
        sys.exit(2)
