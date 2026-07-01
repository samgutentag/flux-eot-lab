# Methodology and rationale

The [README](README.md) is the quickstart: what this lab does and how to run it. This
document is the how and the why, how the numbers are measured, why the design choices,
and where the tool fits.

## The eager-EOT head-start and cost

When eager is on, `EagerEndOfTurn` fires before the committed `EndOfTurn`, handing
downstream work a head-start, paid for in wasted speculation (eager fires the speaker
walks back via `TurnResumed`). At the best eager-on point per class:

| Audio class | Head-start (p50) | Wasted-speculation rate, eager on |
|---|---|---|
| Clean / short turns | 0 ms | 7.7% |
| Clean / long-form | 0 ms | 0.0% |
| Noisy / single speaker | 387 ms | 7.7% |
| Crosstalk (secondary voice) | 209 ms | 6.7% |

The head-start is conditional: on clean audio at the low recommended threshold eager
buys ~0 ms (the events nearly coincide), while on noisy and crosstalk it buys 200 to
390 ms of lead at a single-digit wasted-speculation rate, well under Deepgram's launch
blog "50-70% more calls" figure.

## How the measurement works

**Headline metric: detection latency.** From the user's true end of speech (annotated
`true_end` ground truth) to the first end-of-turn event (`EagerEndOfTurn` or
`EndOfTurn`). This is the one span `eot_threshold` moves.

What this lab tunes is detection; the LLM and text-to-speech pipeline is a separate,
variable clock it does not measure:

```
true_end --detection--> EagerEndOfTurn / EndOfTurn   ||   LLM --> TTS --> first audio
         (tunable; measured here)                    ||   (your choice; variable; out of scope)
```

The downstream pipeline is deliberately not measured: its latency swings run-to-run with
your model and network and has nothing to do with the threshold being tuned, so bundling
it in would only add variance the knob cannot move.

**Why p50 and p95, never a single average.** Deepgram's own
[Aura 2 latency video](https://www.youtube.com/watch?v=u5RJ-3WtFKo) makes the point out
loud: "average latency isn't the real problem. Variability is, those long-tail spikes.
That's the awkward pause your users remember." The median tells you how it usually feels.
The p95 is the turn your user actually remembers.

On the pipeline clock: that same video puts Aura's time-to-first-byte at sub-100ms and
cites Coval's independent benchmark ranking Aura 2 first on first-byte time. Coval's
[TTS benchmark](https://benchmarks.coval.ai/tts) is public, so you can check the standings
yourself. Your measured pipeline should land near that anchor. If it does not, the gap is
in your integration, not the model.

**Measure raw once, slice many ways.** The harness captures raw per-turn data: the full
event timeline plus the signed delta between every turn-boundary event and the annotated
true end. Classification (clean / hard-cutoff / near-miss recovered via `TurnResumed` /
late) and the tolerance window (default &plusmn;200 ms) are applied at analysis time, not
baked into capture, so the tolerance is sweepable without re-running.

**A guard on the data.** Flux segments a clip its own way, and on hard audio it can split
or merge a turn, leaving an extra Flux segment with no annotated counterpart. The analysis
flags those turns as suspect (a turn count that does not match the annotation, or an
impossible negative detection delta) and excludes them, so a mis-segmented turn cannot
poison the published curve. In this run, 105 of 597 turns (about 18%) were excluded that
way; the noisy class is the main contributor, and tightening it further is a known follow-up.

**The sweep.** `eot_threshold` over {0.5, 0.6, 0.7, 0.8, 0.9}, each run with eager off
(`eager_eot_threshold` unset) and eager on (a few steps below `eot_threshold`). One fast
pass by default; `--refine` adds finer passes around the knee. Verified ranges:
`eot_threshold` 0.5-0.9 (default 0.7), `eager_eot_threshold` 0.3-0.9 (must be &le;
`eot_threshold`, unset = eager disabled), `eot_timeout_ms` 500-10000 (default 5000).
The third knob, `eot_timeout_ms`, is the silence backstop; it is held at the default and
not swept, because it only fires when confidence never reaches the bar, not as a feel dial.

**Budget band.** green &le; 2%, yellow &le; 5%, red &gt; 5% false-cutoff, at a ~600 ms
operating point. No published industry standard for an acceptable false-cutoff rate
exists (verified June 2026). The 5% line is where best-in-class measured detectors land
in LiveKit's open `eot-bench` (LiveKit v1 4.5%, Soniox 5.5% @ 600 ms); the 2% line
matches the academic early-cut floor (~2.2-2.4%) and the one published practitioner
budget. See sources at the bottom.

**The four audio classes.** (1) clean / short turns (control), (2) clean / long-form
turns (long single-utterance turns, detection on extended speech), (3) noisy / single
speaker (acoustic robustness), (4) crosstalk with a secondary voice (false-trigger
robustness). The speaker axis is crosstalk robustness, not diarization. See `audio/spec.md`.

**Sample size.** This run is 3 clips per class, about 9 to 15 valid turns per swept
point after the suspect guard. Enough to show the knee and the per-class shifts; for
tight p95 confidence intervals, generate more clips and re-sweep (the harness scales by
dropping in more clips). At this size the exact per-class threshold is not statistically pinned:
bootstrapping the run (resampling turns and re-recommending 300 times) moves the recommended
`eot` across two or three values for every class, noisy the most (0.9 under half the time). The
stable finding is the tradeoff itself, not the specific digit, which is why the honest output is a
recommendation plus the guidance to scale up or measure your own audio. The full per-class curve (detection p50 against false-cutoff
rate, swept across `eot_threshold` with eager on and off) is in
`results/recommendations.md`; regenerate it any time with
`python -m flux_eot_lab.analysis results/raw_real.jsonl`.

### Disclosure

Run on: Apple Silicon macOS in Santa Barbara, California, US (residential 2 Gbps symmetric
fiber, measured over WiFi). Endpoint round-trip via an Aura ping, request to first audio byte,
so it includes TLS and synthesis, not pure network latency: median 328 ms, p95 352 ms
(min 237, max 352) over 15 pings. It varies ~100 ms ping to ping; a deployment co-located
near the Deepgram endpoint will see lower, steadier numbers. Measure your own with
`bench.py --ping`.

## Why now

Flux shipped in October 2025. It is mature, not new. Deepgram publishes plenty: threshold
ranges and defaults, four use-case config presets in the [Flux configuration docs](https://developers.deepgram.com/docs/flux/configuration)
(keyed to a use case like medical or RAG, with no measured numbers behind them), and
self-reported launch percentiles (p90 1 s, p95 1.5 s on an undisclosed test set). A competitor,
LiveKit's `eot-bench`, benchmarked Flux once (9.9% false-cutoff @ 600 ms) on its own audio.
What nobody publishes is measured end-of-turn latency broken out by acoustic condition
(clean / noisy / crosstalk) that a builder can reproduce on their own audio. That is the gap
this fills, and it doubles as the evidence base if Deepgram's own numbers age.

Deepgram is putting out latency content right now: the [Flux launch blog](https://deepgram.com/learn/introducing-flux-conversational-speech-recognition)
and a [June 2026 Aura 2 latency explainer](https://www.youtube.com/watch?v=u5RJ-3WtFKo).
Both make the case that latency matters. Neither tells you where to set `eot_threshold` for
your audio. That is the gap this fills, the practical follow-up to their own posts.

## Why measure detection under the loop instead of the Voice Agent API

This is a pre-production tuning lab, not a production framework and not a rejection of
Deepgram's SDKs. You go under the managed layer for raw event access (`StartOfTurn`,
`EagerEndOfTurn`, `TurnResumed`, `EndOfTurn`) because you cannot tune what you cannot
see. Measure the detection tradeoff for your audio, find your thresholds, then graduate
to production with earned trust. The tuned values transfer directly: the three knobs
(`eot_threshold`, `eager_eot_threshold`, `eot_timeout_ms`) are connection-string params
on the raw Flux API and the same names under `agent.listen.provider` (v2) on the managed
Voice Agent API. What you leave behind moving to the managed agent is only the
event-handling plumbing. Verify the layer, then trust it.

## Glossary

| Term | Meaning |
|---|---|
| **end-of-turn (EOT)** | the moment Flux decides the speaker is done and it is the agent's turn |
| **detection latency** | true end of speech to the first end-of-turn event; the tunable headline metric |
| **`eot_threshold`** | confidence Flux needs to commit an `EndOfTurn` (0.5 to 0.9, default 0.7) |
| **`eager_eot_threshold`** | lower bar for a speculative early `EagerEndOfTurn` (0.3 to 0.9, must be at or below `eot_threshold`; unset = off) |
| **`eot_timeout_ms`** | silence fallback that forces an `EndOfTurn` (500 to 10000, default 5000) |
| **false cutoff** | a committed `EndOfTurn` that landed before the speaker actually finished |
| **wasted speculation** | an `EagerEndOfTurn` walked back by a `TurnResumed` because the speaker kept going |
| **head-start** | how much earlier `EagerEndOfTurn` fires than the committed `EndOfTurn` |
| **p50 / p95** | 50th and 95th percentile; the typical turn and the near-worst turn |
| **STT / TTS** | speech-to-text (Flux) / text-to-speech (Aura) |

## Sources

- Deepgram, "Introducing Flux" launch blog (self-reported percentiles, eager cost).
- Deepgram Flux docs: [configuration](https://developers.deepgram.com/docs/flux/configuration) (threshold ranges plus four use-case config presets, opinionated starting points with no measured numbers behind them), state machine, eager-EOT, Voice Agent settings.
- LiveKit `eot-bench` (open, competitor-run; Flux 9.9% @ 600 ms).
- Stivers et al. 2009 (PNAS) human turn-taking baseline; arXiv 2401.08916 early-cut floor.
- Each claim above links inline to its source.
