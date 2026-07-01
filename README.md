# Flux End-of-Turn Detection Tuning Lab

Your Flux agent works but does not feel good yet: it talks over people, or lags after they stop.
This measures turn **detection** (true end of speech to the first end-of-turn event, the span
`eot_threshold` controls) so you can tune it for your own audio. 

## Recommendations

These are one run's results (3 clips per class): each row is the lowest detection latency whose
false-cutoff rate stays in budget at a ~600 ms operating point. Lower latency is paid for in interruptions.

| Audio class | `eot_threshold` | Eager | Detection p50 | Detection p95 | False-cutoff | Band |
|---|---|---|---|---|---|---|
| Clean / short turns | `0.6` | `0.40` | -81 ms | 48 ms | 0.0% | green |
| Clean / long-form turns | `0.6` | `0.40` | 41 ms | 614 ms | 0.0% | green |
| Noisy / single speaker | `0.9` | `0.70` | 349 ms | 5054 ms | 0.0% | green |
| Crosstalk (secondary voice) | `0.8` | `0.60` | 220 ms | 671 ms | 0.0% | green |

**Read these as one run, not fixed truth.** The latency-vs-false-cutoff tradeoff is real: false cutoffs
rise at low thresholds, latency (and a fat tail) rise at high ones. But at this sample size (3 clips,
~16 turns per setting) the exact optimal threshold is not pinned. Bootstrapping this run 300 times, every
class's recommended threshold spreads across two or three values (noisy the most, 0.9 under half the
time). The honest takeaway is the shape, not the digit: **there is no single right threshold, it depends
on your audio, so measure instead of guessing.** Run it on your own audio, and add clips per class for a
tighter answer (see [METHODOLOGY.md](METHODOLOGY.md), "Sample size").

- **A high threshold buys a fat tail:** on noisy audio at 0.9, p95 stretches to ~5 s (Flux waiting out the silence timeout). Worth knowing before you crank the knob "to be safe."

> Every cell is a p50 and a p95, never an average. The sweep design, the honesty guardrails, and the
sources are in [METHODOLOGY.md](METHODOLOGY.md).

> Working with an AI agent (Claude Code, Cursor, and the like)? Point it at `METHODOLOGY.md`. It is
> written as a domain model an agent can ground on: the knob semantics, the latency-vs-cutoff tradeoff,
> and a glossary. Hand it the repo and it can reason about tuning, not just run the command.

## Install

- Python 3.10+, then `pip install -r requirements.txt && pip install -e .` (the second installs the `flux-bench` command).
- `libsndfile` (macOS: `brew install libsndfile`; Debian/Ubuntu: `apt-get install libsndfile1`).
- A real run needs `DEEPGRAM_API_KEY` with Flux + Aura access. Mock mode needs no key and no audio.

## Run

```bash
# No key, no audio (committed fixtures):
flux-bench --mock

# Real run: generate the clips (needs the key), then sweep and recommend:
python audio/generate_audio.py
flux-bench --input audio/manifest.json --output results/
```

> Bringing your own audio instead? `audio/annotate_real.py` drafts the sidecars; see `audio/annotate-your-audio.md`.

## The skill: `scaffold-test-clips`

Describe a condition in plain language ("a loud cafe with a second speaker") and it writes the
dialogue, turn by turn. Aura synthesizes it, so `true_end_ms` is exact by construction. The model
invents what to test; it never touches the ground truth or a measured number.

```bash
python audio/generate_audio.py --defs audio/generated/<condition>_defs.json
flux-bench --class <condition>
```

Example output: `audio/generated/cafe_crosstalk_defs.json`. The flip side of that choice: AI stays
out of the measurement, the harness is deterministic, with no model near a reported number.

## Who it's for

Solo and small-team builders who shipped a Flux agent off the quickstart, and teams evaluating Flux on
their own conditions. Independent benchmarks exist, but they are one operating point on someone else's
audio. Measure yours.

## Next

Same harness, next condition: barge-in (an interruption class), then multilingual
(`flux-general-multi`), then long-session resilience.
