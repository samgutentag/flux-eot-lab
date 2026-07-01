"""Analysis for the Flux End-of-Turn Detection Tuning Lab.

Classification and the tolerance window are applied HERE, at analysis time, not
baked into capture. The harness records RAW per-turn data once; this module
slices it many ways. Because tolerance is a parameter, you can re-classify a
finer or coarser window without re-running a single audio class.

Headline metric: DETECTION latency = first end-of-turn event (EagerEndOfTurn or
EndOfTurn) minus the user's annotated true end of speech. This is the part the
eot_threshold knob actually moves. The downstream large-language-model and
text-to-speech pipeline is out of scope; its latency varies independently of the
threshold and is the integrator's choice, so it is not measured here.

CLI:
    python -m flux_eot_lab.analysis RAW.jsonl [--tolerance-ms 200] [--operating-point-ms 600]

Writes results/analysis_<ts>.json (per-class curve points) and
results/recommendations.md (the README tables, computed from the data). If the
input is mock wiring data, a loud banner makes clear the numbers are NOT
measurements.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

from flux_eot_lab.config import AUDIO_CLASSES, BUDGET_BANDS, DEFAULT_OPERATING_POINT_MS, DEFAULT_TOLERANCE_MS
from flux_eot_lab.records import TurnRecord, read_jsonl

# Human-facing labels for the four audio classes, in canonical order.
CLASS_LABELS = {
    "clean_short": "Clean / short turns",
    "clean_long": "Clean / long-form turns",
    "noisy_single": "Noisy / single speaker",
    "crosstalk": "Crosstalk (secondary voice)",
    "cafe_crosstalk": "Cafe (crosstalk + noise)",
}

# Where generated artifacts land. Resolved off the package location so the output
# path is identical no matter what directory the CLI is invoked from.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# --------------------------------------------------------------------------- #
# Per-record primitives
# --------------------------------------------------------------------------- #

# Detection latency below this is physically impossible (Flux cannot detect the
# end of a turn a full second before the speaker stops). It only happens when an
# extra Flux segment shifts the time-based turn matching and a committed EndOfTurn
# is paired to the wrong annotated true_end.
_IMPLAUSIBLE_DETECTION_MS = -1000.0


def _is_nan(x: float | None) -> bool:
    return x is not None and isinstance(x, float) and math.isnan(x)


def is_suspect(rec: TurnRecord) -> bool:
    """A turn whose timing is impossible or unmatched, excluded from aggregation.

    Two causes, both downstream of Flux segmenting a clip into a different number
    of turns than were annotated (it split or merged a turn):
      - an extra Flux segment beyond the annotated turns -> true_end_ms is NaN
        (the agent fills NaN when a Flux turn_index has no annotated counterpart);
      - the time matching paired the wrong events -> an impossibly negative
        detection delta.
    Excluding these keeps a mis-segmented run from poisoning the published curve
    with negative-thousands detection and NaN percentiles. Valid turns
    are untouched.
    """
    if rec.true_end_ms is None or _is_nan(rec.true_end_ms):
        return True
    d = rec.detection_delta_ms
    if d is not None and not _is_nan(d) and d < _IMPLAUSIBLE_DETECTION_MS:
        return True
    return False


def classify(rec: TurnRecord, tol_ms: float) -> str:
    """Bucket a single raw turn against a tolerance window.

    Returns one of: "clean", "hard_cutoff", "near_miss", "late".

    - hard_cutoff: a committed EndOfTurn landed before the user actually finished
      (end_of_turn_ms < true_end - tol). This is the real false cutoff the user
      hears -- the agent would have talked over them. Worst outcome, evaluated first.
    - near_miss: EagerEndOfTurn fired early (before true_end - tol) but the model
      recovered via TurnResumed, so it did NOT become a hard cutoff. This is the
      speculative-waste category -- an early start that got walked back.
    - late: a committed EndOfTurn landed well after the user finished
      (end_of_turn_ms > true_end + tol). Safe but sluggish.
    - clean: everything else -- the boundary landed inside the tolerance window.
    """
    true_end = rec.true_end_ms
    eot = rec.end_of_turn_ms
    eager = rec.eager_eot_ms

    if eot is not None and eot < true_end - tol_ms:
        return "hard_cutoff"
    if rec.eager_fired and eager is not None and eager < true_end - tol_ms and rec.turn_resumed_fired:
        return "near_miss"
    if eot is not None and eot > true_end + tol_ms:
        return "late"
    return "clean"


def detection_ms(rec: TurnRecord) -> float | None:
    """HEADLINE metric: detection latency, true_end -> first end-of-turn event.

    The signed gap between the annotated true end of speech and the first
    end-of-turn event Flux fired. This is the only span the eot_threshold knob
    moves. None for an errored or unmatched turn, so it is excluded from the
    distribution rather than poisoning it.
    """
    if rec.detection_delta_ms is not None and not _is_nan(rec.detection_delta_ms):
        return rec.detection_delta_ms
    if rec.first_eot_event_ms is None or rec.true_end_ms is None:
        return None
    if _is_nan(rec.first_eot_event_ms) or _is_nan(rec.true_end_ms):
        return None
    return rec.first_eot_event_ms - rec.true_end_ms


def head_start_ms(rec: TurnRecord) -> float | None:
    """Eager head-start: how many ms earlier EagerEndOfTurn fired than the
    committed EndOfTurn. This is the lead time eager hands downstream work. None
    unless eager fired and both events are present."""
    if not rec.eager_fired:
        return None
    eager = rec.eager_eot_ms
    eot = rec.end_of_turn_ms
    if eager is None or eot is None or _is_nan(eager) or _is_nan(eot):
        return None
    return eot - eager


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #


def _percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile. Robust for any sample size, including n=1."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(s[int(k)])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def _band(false_cutoff_rate: float) -> str:
    """Tag a false-cutoff rate green / yellow / red per the locked budget bands."""
    if false_cutoff_rate <= BUDGET_BANDS["green"]:
        return "green"
    if false_cutoff_rate <= BUDGET_BANDS["yellow"]:
        return "yellow"
    return "red"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def analyze(records: Iterable[TurnRecord], tol_ms: float, op_ms: float) -> dict:
    """Aggregate raw turns into per-(class, sweep-point) curve points.

    Returns a dict with the curve data and run metadata. Each curve point carries
    p50/p95 DETECTION latency, the eager head-start p50, false_cutoff_rate
    (hard_cutoff / total) and waste_ratio (near_miss / total).
    """
    records = list(records)
    n_records = len(records)
    n_errors = sum(1 for r in records if r.error)
    # Mis-segmented / unmatched turns, dropped so they cannot poison the curve.
    n_suspect = sum(1 for r in records if not r.error and is_suspect(r))
    is_mock = any(r.mock for r in records)

    # group key: (audio_class, eot_threshold, eager_eot_threshold)
    groups: dict[tuple, list[TurnRecord]] = {}
    for rec in records:
        if rec.error or is_suspect(rec):
            continue
        key = (rec.audio_class, rec.eot_threshold, rec.eager_eot_threshold)
        groups.setdefault(key, []).append(rec)

    curve: list[dict] = []
    for (audio_class, eot, eager), recs in groups.items():
        total = len(recs)
        cats = [classify(r, tol_ms) for r in recs]
        hard = cats.count("hard_cutoff")
        near = cats.count("near_miss")
        late = cats.count("late")
        clean = cats.count("clean")

        detections = [detection_ms(r) for r in recs]
        head_starts = [head_start_ms(r) for r in recs]

        curve.append(
            {
                "class": audio_class,
                "eot": eot,
                "eager": eager,
                "n": total,
                # p50/p95 are DETECTION latency (the headline metric).
                "p50": _round(_percentile(detections, 50)),
                "p95": _round(_percentile(detections, 95)),
                "head_start_p50": _round(_percentile(head_starts, 50)),
                "false_cutoff": _round(hard / total if total else None, 4),
                "waste": _round(near / total if total else None, 4),
                "clean": clean,
                "late": late,
                "counts": {"clean": clean, "hard_cutoff": hard, "near_miss": near, "late": late},
            }
        )

    # stable ordering: class (canonical), then eot, then eager-off before eager-on
    curve.sort(key=lambda p: (_class_order(p["class"]), p["eot"], (p["eager"] is not None, p["eager"] or 0)))

    return {
        "tolerance_ms": tol_ms,
        "operating_point_ms": op_ms,
        "mock": is_mock,
        "n_records": n_records,
        "n_errors": n_errors,
        "n_suspect": n_suspect,
        "budget_bands": BUDGET_BANDS,
        "curve": curve,
    }


def observed_classes(analysis: dict) -> list[str]:
    """Report order for classes: the built-in AUDIO_CLASSES first, then any extra
    class present in the swept data (a bring-your-own condition from the
    scaffold-test-clips skill), sorted. Mock and built-in runs see no extras, so
    the existing tables are byte-identical; a new condition shows up on its own
    once it has records, with no edit to AUDIO_CLASSES.
    """
    seen = {p["class"] for p in analysis.get("curve", [])}
    return list(AUDIO_CLASSES) + sorted(c for c in seen if c not in AUDIO_CLASSES)


def recommend(analysis: dict, op_ms: float) -> dict:
    """Per class: the lowest-DETECTION-latency sweep point whose false-cutoff rate
    stays inside the budget band, tagged green / yellow / red.

    Returns a dict keyed by audio class (one of config.AUDIO_CLASSES); each value
    is either None (no data for that class) or a dict with eot / eager / p50 / p95 /
    false_cutoff / waste / band / within_budget. The operating point itself lives on
    the analysis dict (analysis["operating_point_ms"]), so callers read it from there.

    Selection rule: among a class's swept points, keep those whose
    false_cutoff_rate is acceptable (<= yellow band), then pick the one with the
    lowest p50 DETECTION latency. If no point is acceptable (all red), fall back to
    the lowest-false-cutoff point and tag it red, so the table is honest about
    there being no safe operating point rather than hiding the class.

    The ~operating point (op_ms) is the stated detection budget the recommendation
    is reported AT; `within_budget` flags whether the chosen p50 lands inside it.
    """
    yellow = BUDGET_BANDS["yellow"]
    by_class: dict[str, dict | None] = {}

    for cls in observed_classes(analysis):
        points = [p for p in analysis["curve"] if p["class"] == cls and p["p50"] is not None and p["false_cutoff"] is not None]
        if not points:
            by_class[cls] = None
            continue

        acceptable = [p for p in points if p["false_cutoff"] <= yellow]
        pool = acceptable if acceptable else points
        # lowest detection latency wins; if none acceptable, surface the least-bad false-cutoff
        if acceptable:
            chosen = min(pool, key=lambda p: p["p50"])
        else:
            chosen = min(pool, key=lambda p: p["false_cutoff"])

        by_class[cls] = {
            "eot": chosen["eot"],
            "eager": chosen["eager"],
            "p50": chosen["p50"],
            "p95": chosen["p95"],
            "false_cutoff": chosen["false_cutoff"],
            "waste": chosen["waste"],
            "band": _band(chosen["false_cutoff"]),
            "within_budget": chosen["p50"] is not None and chosen["p50"] <= op_ms,
        }

    return by_class


def eager_cost(analysis: dict) -> dict:
    """Per class: the eager-on speculative-waste ratio and the head-start it buys.

    Picks the best eager-on point (acceptable false-cutoff, lowest detection p50).
    `head_start` is how many ms earlier EagerEndOfTurn fires than the committed
    EndOfTurn there -- the lead time eager hands downstream work, paid for in the
    wasted-speculation rate (eager fires walked back by TurnResumed).
    """
    yellow = BUDGET_BANDS["yellow"]
    out: dict[str, dict | None] = {}

    for cls in observed_classes(analysis):
        pts = [p for p in analysis["curve"] if p["class"] == cls and p["p50"] is not None]
        on = [p for p in pts if p["eager"] is not None]

        def _best(group: list[dict]) -> dict | None:
            if not group:
                return None
            ok = [p for p in group if p["false_cutoff"] is not None and p["false_cutoff"] <= yellow]
            return min(ok or group, key=lambda p: p["p50"])

        best_on = _best(on)
        if best_on is None:
            out[cls] = None
            continue

        out[cls] = {
            "waste": best_on["waste"],
            "head_start": best_on.get("head_start_p50"),
            "eager_eot": best_on["eager"],
        }

    return out


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def _round(x: float | None, ndigits: int = 1) -> float | None:
    return None if x is None else round(x, ndigits)


def _class_order(cls: str) -> int:
    return AUDIO_CLASSES.index(cls) if cls in AUDIO_CLASSES else len(AUDIO_CLASSES)


def _ms(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0f} ms"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _eager_cell(x: float | None) -> str:
    return "off" if x is None else f"{x:.2f}"


def render_recommendations_md(analysis: dict, rec: dict, cost: dict) -> str:
    """Build results/recommendations.md: the per-class detection recommendation,
    the eager head-start/cost table, and the full curve, filled from the data. A
    mock banner fires when the input is wiring data.
    """
    mock = analysis["mock"]
    op = analysis["operating_point_ms"]
    tol = analysis["tolerance_ms"]
    lines: list[str] = []

    lines.append("# Flux end-of-turn detection recommendations")
    lines.append("")
    if mock:
        lines.append("> ## :warning: MOCK WIRING NUMBERS -- NOT MEASUREMENTS")
        lines.append("> ")
        lines.append("> These numbers were computed from `--mock` fixture data. The mock Flux")
        lines.append("> source synthesizes event timing deterministically to exercise the analysis")
        lines.append("> code; it does not touch Deepgram and measures nothing. **Do not paste")
        lines.append("> these into the README.** The README's measured cells stay `PENDING_REAL_RUN`")
        lines.append("> until the harness runs against a real key. Re-run with real sources to get")
        lines.append("> numbers you can report.")
        lines.append("")
    else:
        lines.append("> Generated from a real harness run against real Deepgram sources.")
        lines.append("")
    lines.append(
        f"Detection latency = true end of speech to the first end-of-turn event. "
        f"Tolerance window &plusmn;{tol:.0f} ms. Operating point ~{op:.0f} ms. "
        f"Budget band: green &le; {BUDGET_BANDS['green'] * 100:.0f}% / "
        f"yellow &le; {BUDGET_BANDS['yellow'] * 100:.0f}% / red &gt; {BUDGET_BANDS['yellow'] * 100:.0f}% "
        f"false-cutoff. {analysis['n_records']} raw turns, {analysis['n_errors']} errored, "
        f"{analysis.get('n_suspect', 0)} excluded as suspect (mis-segmented)."
    )
    lines.append("")

    # Table 1: per-class recommendation
    lines.append("## Per-class threshold recommendations")
    lines.append("")
    lines.append(
        "| Audio class | Recommended `eot_threshold` | Eager (`eager_eot_threshold`) "
        "| Detection p50 | Detection p95 | False-cutoff rate | Band |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for cls in observed_classes(analysis):
        label = CLASS_LABELS.get(cls, cls)
        r = rec.get(cls)
        if r is None:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | `{r['eot']}` | `{_eager_cell(r['eager'])}` | {_ms(r['p50'])} "
            f"| {_ms(r['p95'])} | {_pct(r['false_cutoff'])} | {r['band']} |"
        )
    lines.append("")

    # Table 2: eager-EOT head-start vs cost
    lines.append("## The eager-EOT head-start and cost")
    lines.append("")
    lines.append("At the best eager-on point per class: the head-start (how much earlier `EagerEndOfTurn` fires than the committed `EndOfTurn`, the lead time eager hands downstream work) and the wasted-speculation rate (eager fires walked back by `TurnResumed`).")
    lines.append("")
    lines.append("| Audio class | Head-start (p50) | Wasted-speculation rate, eager on |")
    lines.append("|---|---|---|")
    for cls in observed_classes(analysis):
        label = CLASS_LABELS.get(cls, cls)
        c = cost.get(cls)
        if c is None:
            lines.append(f"| {label} | n/a | n/a |")
            continue
        lines.append(f"| {label} | {_ms(c['head_start'])} | {_pct(c['waste'])} |")
    lines.append("")

    # Table 3: full curve (the data behind the tradeoff plot)
    lines.append("## Full curve data")
    lines.append("")
    lines.append("Every swept point. The hero plot is detection p50 against false-cutoff rate, one curve per class.")
    lines.append("")
    lines.append("| Class | `eot` | Eager | n | Detection p50 | Detection p95 | Head-start p50 | False-cutoff | Wasted-spec |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in analysis["curve"]:
        label = CLASS_LABELS.get(p["class"], p["class"])
        lines.append(
            f"| {label} | `{p['eot']}` | `{_eager_cell(p['eager'])}` | {p['n']} "
            f"| {_ms(p['p50'])} | {_ms(p['p95'])} | {_ms(p['head_start_p50'])} "
            f"| {_pct(p['false_cutoff'])} | {_pct(p['waste'])} |"
        )
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _write_outputs(analysis: dict, rec: dict, cost: dict) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Mock runs write to a SEPARATE path so a --mock or quickstart run can never
    # clobber the committed real tables with wiring-fixture numbers.
    suffix = "_mock" if analysis["mock"] else ""

    json_path = RESULTS_DIR / f"analysis{suffix}_{ts}.json"
    payload = {
        "generated_at": ts,
        "tolerance_ms": analysis["tolerance_ms"],
        "operating_point_ms": analysis["operating_point_ms"],
        "mock": analysis["mock"],
        "n_records": analysis["n_records"],
        "n_errors": analysis["n_errors"],
        "n_suspect": analysis.get("n_suspect", 0),
        "budget_bands": analysis["budget_bands"],
        "curve": analysis["curve"],
        "recommendations": rec,
        "eager_cost": cost,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    md_name = "recommendations_mock.md" if analysis["mock"] else "recommendations.md"
    md_path = RESULTS_DIR / md_name
    md_path.write_text(render_recommendations_md(analysis, rec, cost))
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify raw Flux turn records and emit per-class detection tradeoff curves + recommendations."
    )
    parser.add_argument("raw", help="Path to the raw per-turn JSONL produced by bench.py")
    parser.add_argument("--tolerance-ms", type=float, default=DEFAULT_TOLERANCE_MS,
                        help=f"Tolerance window for classification (default {DEFAULT_TOLERANCE_MS}).")
    parser.add_argument("--operating-point-ms", type=float, default=DEFAULT_OPERATING_POINT_MS,
                        help=f"Stated detection operating point (default {DEFAULT_OPERATING_POINT_MS}).")
    args = parser.parse_args(argv)

    raw_path = Path(args.raw)
    if not raw_path.exists():
        parser.error(f"raw file not found: {raw_path}")

    records = read_jsonl(raw_path)
    if not records:
        print(f"No records in {raw_path}; nothing to analyze.")
        return 1

    analysis = analyze(records, args.tolerance_ms, args.operating_point_ms)
    rec = recommend(analysis, args.operating_point_ms)
    cost = eager_cost(analysis)

    if analysis["mock"]:
        print("=" * 72)
        print("  MOCK WIRING DATA -- these are NOT measurements.")
        print("  The analysis code is exercised end-to-end, but every number")
        print("  below comes from synthetic fixture timing. Do not report it.")
        print("  README measured cells stay PENDING_REAL_RUN until a real run.")
        print("=" * 72)

    json_path, md_path = _write_outputs(analysis, rec, cost)

    print(f"\nAnalyzed {analysis['n_records']} turns "
          f"({analysis['n_errors']} errored, {analysis.get('n_suspect', 0)} suspect excluded) "
          f"at tol={args.tolerance_ms:.0f}ms, op={args.operating_point_ms:.0f}ms.")
    print("\nPer-class recommendation (detection latency):")
    for cls in observed_classes(analysis):
        r = rec.get(cls)
        label = CLASS_LABELS.get(cls, cls)
        if r is None:
            print(f"  {label:42s} no data")
        else:
            print(f"  {label:42s} eot={r['eot']} eager={_eager_cell(r['eager'])} "
                  f"detection_p50={_ms(r['p50'])} false_cutoff={_pct(r['false_cutoff'])} [{r['band']}]")

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    if analysis["mock"]:
        print("\nReminder: mock numbers. Re-run against real keys for reportable results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
