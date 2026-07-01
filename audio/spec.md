# Audio Class Spec + Annotation Format

This lab measures Flux turn-taking latency vs false-cutoff across **four audio
classes**. Each clip is a single WAV (waveform audio) file plus a JSON **sidecar** that carries the
annotated ground truth. The harness never guesses where a turn ends: it reads
`true_end_ms` from the sidecar. Classification and the tolerance window are
applied at *analysis* time (see `flux_eot_lab/analysis.py`), so the same raw run
can be re-scored at different tolerances without re-recording.

> Numbers in the committed sidecars are **input ground truth** (where speech
> actually stops in a synthetic clip), not measured results. Measured latency /
> false-cutoff values only ever come from a real `bench.py` run and live in
> `results/`. They are never written here.

---

## The four classes

| `audio_class`   | What it stresses                              | Shape                                                                 |
| --------------- | --------------------------------------------- | --------------------------------------------------------------------- |
| `clean_short`   | Baseline turn-taking on tidy, short turns     | 4-6 short turns, clean speech, normal inter-turn gaps                 |
| `clean_long`    | Detection on long single-utterance turns      | 4 long single-utterance turns, clean speech, no intra-turn pauses      |
| `noisy_single`  | Robustness to a noise bed, one speaker        | 4-6 turns over a low-level broadband noise bed (single speaker)       |
| `crosstalk`     | A second voice talking over the primary       | 4-6 turns, a secondary voice overlaps the primary on `distractor_spans` |

The four `audio_class` values are fixed in `flux_eot_lab/config.AUDIO_CLASSES`
(`["clean_short", "clean_long", "noisy_single", "crosstalk"]`). The speaker axis
in `crosstalk` is **crosstalk robustness, not diarization**: we only care
whether the distractor voice causes a premature end-of-turn (EOT), not who said what.

### Why these four

- `clean_short` is the floor: if turn-taking is bad here, nothing else matters.
- `clean_long` tests detection on long, extended turns of clean speech (single
  utterances, no intra-turn pauses), where the speaker talks for a while before the
  true end. It checks that detection latency holds up as turns get longer.
- `noisy_single` separates "noise" from "another talker" so we don't blame
  diarization for what is really a signal-to-noise ratio (SNR) problem.
- `crosstalk` is the adversarial case: a real second voice over annotated spans.

---

## Sidecar schema (`audio/clips/<clip_id>.json`)

One JSON object per clip. Field names map **1:1** to the `ClipSpec` / `TurnSpec`
dataclasses in `flux_eot_lab/records.py`. `ClipSpec.from_sidecar(path)` parses
this file.

```jsonc
{
  "clip_id": "clean_short_1",          // unique id; matches the filename stem
  "audio_class": "clean_short",        // one of config.AUDIO_CLASSES
  "audio_file": "clean_short_1.wav",   // path to the WAV (mock mode ignores it)
  "sample_rate": 16000,                // Flux input rate (config.INPUT_SAMPLE_RATE)
  "encoding": "linear16",              // Flux input encoding
  "duration_ms": 16200.0,              // total clip length in ms
  "notes": "Short clean single-utterance turns ...",  // free text; provenance, SNR, etc.
  "turns": [
    {
      "turn_index": 0,                 // 0-based, monotonic within the clip
      "true_end_ms": 2500.0,           // GROUND TRUTH: ms from clip start to the
                                        //   moment the speaker truly stops this turn
      "transcript": "Hey, what's the weather looking like today?",
      "speaker": "primary",            // "primary" | "secondary"; default "primary"
      "distractor_spans": []           // crosstalk only; see below
    }
  ]
}
```

### Field reference

| Field                 | Type            | Meaning                                                                                          |
| --------------------- | --------------- | ------------------------------------------------------------------------------------------------ |
| `clip_id`             | string          | Unique clip id. Convention: matches the sidecar filename stem.                                    |
| `audio_class`         | string          | One of `config.AUDIO_CLASSES`.                                                                    |
| `audio_file`          | string \| null  | Relative path to the WAV. Mock mode ignores it (reads no audio); a real run replays it.          |
| `sample_rate`         | int             | Always `16000` for Flux input (`config.INPUT_SAMPLE_RATE`).                                       |
| `encoding`            | string          | Always `"linear16"`.                                                                              |
| `duration_ms`         | float           | Total clip duration in ms.                                                                        |
| `notes`               | string          | Free text. Provenance, noise SNR, voice ids used, etc.                                            |
| `turns`               | array           | Ordered list of `TurnSpec` objects (see below). 4-6 per clip here.                               |

### `TurnSpec`

| Field              | Type   | Meaning                                                                                                    |
| ------------------ | ------ | ---------------------------------------------------------------------------------------------------------- |
| `turn_index`       | int    | 0-based, strictly increasing within the clip.                                                              |
| `true_end_ms`      | float  | **The headline ground truth.** ms from clip start to the moment the primary speaker truly stops this turn. |
| `transcript`       | string | What the primary speaker says this turn.                                                                    |
| `speaker`          | string | `"primary"` (default) or `"secondary"`. Distractor turns are not listed as turns; see `distractor_spans`.  |
| `distractor_spans` | array  | `crosstalk` only. List of `{"start_ms", "end_ms"}` where a secondary voice overlaps this turn. `[]` otherwise. |

#### `distractor_spans` entry

```jsonc
{ "start_ms": 1400.0, "end_ms": 2100.0 }   // secondary voice audible from 1400ms to 2100ms (clip-relative)
```

Spans are **clip-relative** (same clock as `true_end_ms`), not turn-relative.
They describe *where the distractor voice plays*, which is what the analysis uses
to reason about crosstalk-induced false cutoffs. They are only meaningful for the
`crosstalk` class; other classes leave the list empty.

---

## Manifest schema (`audio/manifest.json`)

`bench.py` loads `audio/manifest.json` and resolves it to a `ClipSpec` list. The
manifest is a thin index of sidecar files: the truth lives in the sidecars, so
there is exactly one place to edit a clip.

```jsonc
{
  "version": 1,
  "generated_by": "audio/generate_audio.py (real Aura clips)",
  "clips": [
    "clips/clean_short_1.json",   // paths are relative to audio/
    "clips/clean_short_2.json",
    "clips/clean_long_1.json",
    "clips/noisy_single_1.json",
    "clips/crosstalk_1.json"
    // ... one entry per committed sidecar
  ]
}
```

`bench.py` does, in effect:

```python
manifest = json.load(open("audio/manifest.json"))
clips = [ClipSpec.from_sidecar(Path("audio") / rel) for rel in manifest["clips"]]
```

To add a clip: drop a sidecar in `audio/clips/`, add its relative path to
`clips`, done.

---

## Ground truth: how `true_end_ms` is annotated

For **generated** clips (`audio/generate_audio.py`) the annotation is *exact by
construction*: each utterance is synthesized separately, concatenated with
silences of a known length, and `true_end_ms` is the running sample position at
the end of that turn's speech. There is no hand-labeling and no detector in the
loop: the ground truth is the clip's own edit list.

For **bring-your-own** clips, annotate `true_end_ms` by hand (e.g. in Audacity:
place a label at the last audible sample of each turn, read off the ms). Keep the
clip mono, 16 kHz, linear16.

A `true_end_ms` is the *true* end of speech. The whole point of the lab is to
measure detection latency: the gap from this instant to the first end-of-turn
event Flux emits (`true_end → EagerEndOfTurn/EndOfTurn`), the part the
`eot_threshold` knob actually controls. The downstream large-language-model and
text-to-speech pipeline is your own choice and is out of scope for this
measurement. If `true_end_ms` is wrong, every latency number derived from it is
wrong, so it is the one thing worth getting right.

---

## Mock vs real runs (the same sidecars drive both)

The sidecars in `audio/clips/` are committed; the WAVs they point at are **not**
(they are gitignored, regenerate with `audio/generate_audio.py`). That one set of
sidecars drives both modes:

| | `bench.py --mock` | `bench.py` (real run) |
| --- | --- | --- |
| Needs a key | no | yes (`DEEPGRAM_API_KEY`) |
| Needs the WAVs | no | yes (regenerate them) |
| Event timeline | synthesized deterministically by `flux_eot_lab/mock.py` | measured from real Flux |
| Ground truth | the sidecar's `true_end_ms` | the sidecar's `true_end_ms` |

Mock mode never reads `audio_file`: the mock Flux source synthesizes the event
timeline from each `true_end_ms` and the sweep point, so the committed sidecars
alone run the full detection pipeline with no audio and no key. A real run replays
the WAVs through Flux for measured numbers.

---

## Generating real audio

```bash
export DEEPGRAM_API_KEY=...           # see the README
python audio/generate_audio.py        # writes audio/<clip_id>.wav + audio/clips/<clip_id>.json
                                       # and rewrites audio/manifest.json to point at them
python audio/generate_audio.py --class crosstalk   # just one class
```

Each generated clip is **16 kHz mono linear16 WAV** with an exact sidecar. The
script uses Aura (`dg.speak.v1.audio.generate`) for the primary voice, a numpy
noise bed for `noisy_single`, and a second Aura voice overlaid on
`distractor_spans` for `crosstalk`. Run with no key and it prints
bring-your-own instructions instead of failing.
