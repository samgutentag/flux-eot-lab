---
name: scaffold-test-clips
description: >-
  Author new test-clip definitions for a target conversational or acoustic
  condition (phone support, cafe crosstalk, impatient caller, and so on) in the
  schema audio/generate_audio.py consumes. The LLM writes the conversational
  material; Aura synthesizes it; the true_end_ms ground truth is exact by
  construction. Use when a condition your existing classes don't cover keeps
  surfacing and you want to grow the test corpus for it.
---

# Scaffold Test Clips

Generate clip definitions for a new test condition. This skill *writes language*:
realistic, varied, condition-appropriate conversation. That is the part an LLM is
actually good at and a config section can never do, and it is the honest place to
put a model in this project.

## Where this sits, and why it is honest

The credibility spine of the lab is that no model touches a measurement. This
skill respects that line exactly. The model sits at the very front of the
pipeline and writes *test stimuli* (what is said, how the turns are shaped). Then
everything downstream is deterministic:

1. **This skill** authors clip defs (conversational material + timing). Non-deterministic, and that is a feature: more variety means better coverage.
2. **`audio/generate_audio.py`** synthesizes each utterance with Aura and concatenates it with measured silences, so `true_end_ms` is *exact by construction*. It is the running sample position at the end of each turn's speech. No hand-labeling, no detector, no model in the loop.
3. **`bench.py` + `analysis`** measure and recommend, deterministically.

So the model expands *what you test*. It never touches the ground truth or a
reported number. Note this is a different use of Aura than the LLM-and-TTS
response pipeline that was deliberately cut from the measurement: there, TTS
latency polluted the numbers; here, Aura mints fixed stimuli with known truth.

## Input

A plain-language brief:

| Field | Notes |
| --- | --- |
| `condition` | what you're simulating, e.g. "cafe support call with a second speaker cutting in" |
| `audio_class` | a short slug for the class, e.g. `cafe_crosstalk`. New slugs are fine (see Registering a new class). |
| `clips` | how many distinct clips (distinct scripts, same shape). 3 is the norm, enough turns for a stable p50/p95. |
| `turns_per_clip` | usually 4 to 5. |
| features | which schema knobs the condition needs: `distractors` (a second voice), `noise_snr_db` (a broadband noise bed). |

## Output

One JSON file, shape `{audio_class: [clip_def, ...]}`, that
`generate_audio.py --defs <file>` renders directly. Each `clip_def`:

```json
{
  "notes": "one line describing the clip",
  "noise_snr_db": 15.0,
  "turns": [
    {"lead_ms": 300, "text": "first utterance", "gap_ms": 2500,
     "distractors": [{"offset_in_turn_ms": 1200, "text": "second voice phrase"}]},
    {"lead_ms": 0, "text": "next utterance", "gap_ms": 2500, "distractors": []}
  ]
}
```

- `lead_ms`: silence before the turn's speech. Use `300` on the first turn, `0` after.
- `text`: the primary utterance. Aura's primary voice synthesizes it. One utterance per turn, no intra-turn pauses, so Flux segments 1:1.
- `gap_ms`: silence after the turn (where an agent would reply). **Keep this at 2500.** Shorter gaps make Flux merge or split turns, which the analysis then flags suspect and discards.
- `distractors` (optional): secondary-voice phrases overlaid inside the turn at `offset_in_turn_ms` from the turn's speech start. Keep them short and keep the offset comfortably inside the utterance.
- `noise_snr_db` (optional, clip-level): adds a broadband noise bed over the speech spans at the given signal-to-noise ratio. Lower is louder. 20 is mild, 12 is loud.

## Writing rules

- **Real, varied dialogue.** Distinct scenarios per clip, natural phrasing, the kind of turns your target persona actually says. This is the whole value; do not produce filler.
- **One utterance per turn**, declarative or a single question. No mid-turn pauses (that is the retired `pauses` path).
- **Gaps stay at 2500.** Non-negotiable for 1:1 segmentation.
- **Distractor offsets sit inside the utterance**, not past its end, or the overlay lands in the gap and reads as a separate turn.
- **No em dashes** in any text.

## Honesty guardrail

This skill authors *stimuli only*. It never writes `true_end_ms`, a detection
latency, a false-cutoff rate, or any measured number. Those are computed by
`generate_audio.py` at synthesis (ground truth) and by `analysis` after a real
run (measurements). If you catch yourself putting a number that looks like a
measurement into the output, stop; that is not this skill's job.

## Procedure

1. Read the brief. Pick the schema features the condition needs.
2. Write `clips` clip defs of `turns_per_clip` turns each, following the writing rules.
3. Emit the JSON file, shape `{audio_class: [clip_def, ...]}`, under `audio/generated/<audio_class>_defs.json`.
4. **Register a new class (only if `audio_class` is new).** So the condition shows up in the recommendation table, add the slug to `AUDIO_CLASSES` and a human label to `CLASS_LABELS`:
   - `flux_eot_lab/config.py`: append the slug to `AUDIO_CLASSES`.
   - `flux_eot_lab/analysis.py`: add `"<slug>": "Human label"` to `CLASS_LABELS`.
   Existing slugs need no change.
5. Render and run (needs `DEEPGRAM_API_KEY`):
   ```
   python audio/generate_audio.py --defs audio/generated/<audio_class>_defs.json
   python recommend.py --input audio/manifest.json --output results/
   ```
   The first command synthesizes the clips and appends them to the manifest; the second sweeps and recommends. The new condition appears alongside the base classes.
