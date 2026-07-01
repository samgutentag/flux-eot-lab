#!/usr/bin/env python3
"""Generate the four audio classes for the Flux end-of-turn (EOT) lab via Aura text-to-speech (TTS).

Each clip is synthesized utterance-by-utterance and concatenated with measured
silences, so the `true_end_ms` ground truth in the sidecar is *exact by
construction* (it's just the running sample position at the end of each turn's
speech). No hand-labeling, no detector in the loop.

Each audio class has SEVERAL clips (distinct scripts, same shape) so a sweep
point sees enough turns for a stable p50/p95.

Outputs, per clip:
  audio/<clip_id>.wav            16 kHz mono linear16
  audio/clips/<clip_id>.json     sidecar (ClipSpec schema; audio_file set)
and rewrites audio/manifest.json to point at every generated sidecar.

Requires DEEPGRAM_API_KEY (for Aura). With no key it prints bring-your-own
instructions and exits 0 - the committed sidecars already let
`bench.py --mock` run with zero audio and zero keys.

Usage:
  python audio/generate_audio.py                 # all classes, all clips
  python audio/generate_audio.py --class crosstalk
  python audio/generate_audio.py --out-dir audio
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Resolve the package whether run as `python audio/generate_audio.py` or `-m`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flux_eot_lab import config  # noqa: E402
from flux_eot_lab.records import ClipSpec, TurnSpec  # noqa: E402

# Zero-dependency .env load so keys in .env are seen even on system python
# outside the venv (config.load_env falls back to a manual parse + warns).
config.load_env(_REPO_ROOT / ".env")  # noqa: E402

SR = config.INPUT_SAMPLE_RATE  # SR = sample rate; 16000 Hz, the Flux input rate
PRIMARY_VOICE = config.DEFAULT_AURA_MODEL  # aura-2-andromeda-en
SECONDARY_VOICE = "aura-2-orion-en"  # a distinct voice for crosstalk distractors
FULL_SCALE = 32767


# --------------------------------------------------------------------------- #
# Clip definitions. Each class maps to a LIST of clips; each clip is a list of
# turns. A turn is a primary utterance plus the silences around it: `lead_ms` is
# silence before the speech, `gap_ms` is silence after (the inter-turn gap where
# an agent would reply). `distractors` (crosstalk) are secondary-voice phrases
# dropped onto the timeline at a clip-relative offset. Every clip stays
# single-utterance-per-turn with 2.5s gaps so Flux segments it 1:1.
# --------------------------------------------------------------------------- #
CLIP_DEFS: dict[str, list[dict]] = {
    "clean_short": [
        {
            "notes": "Short clean single-utterance turns, 2.5s gaps (1:1 with Flux).",
            "turns": [
                {"lead_ms": 300, "text": "Hey, what's the weather looking like today?", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Okay, and what about tomorrow?", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Got it. Set a reminder for nine a.m.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Actually, make it eight thirty.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Thanks, that's all.", "gap_ms": 2500},
            ],
        },
        {
            "notes": "Short clean single-utterance turns, 2.5s gaps (1:1 with Flux).",
            "turns": [
                {"lead_ms": 300, "text": "Can you start a timer for ten minutes?", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Add milk and eggs to my shopping list.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "What's on my calendar this afternoon?", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Move my three o'clock to four.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Great, that works for me.", "gap_ms": 2500},
            ],
        },
        {
            "notes": "Short clean single-utterance turns, 2.5s gaps (1:1 with Flux).",
            "turns": [
                {"lead_ms": 300, "text": "Turn the living room lights down a little.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Play something quiet in the kitchen.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "What time does the pharmacy close?", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Call mom when I get home.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Okay, thank you.", "gap_ms": 2500},
            ],
        },
    ],
    "clean_long": [
        {
            "notes": "Long single-utterance turns, no intra-turn pauses, 2.5s gaps (1:1 with Flux).",
            "turns": [
                {"lead_ms": 300, "text": "So the thing I'm trying to figure out is whether the migration needs to happen before the next release, or if we can just ship the smaller fix first and roll the rest out later.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "The other piece I keep going back and forth on is whether the on-call rotation can actually absorb another service this quarter without burning people out.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Thinking out loud, we have the staging environment but it's pointed at the old database, so we would need to repoint it first, then run the backfill, then verify.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Okay, I think the plan is clear enough, so let's write it up, get two reviewers on it, and target the end of the sprint for the first half of the rollout.", "gap_ms": 2500},
            ],
        },
        {
            "notes": "Long single-utterance turns, no intra-turn pauses, 2.5s gaps (1:1 with Flux).",
            "turns": [
                {"lead_ms": 300, "text": "What I want to understand before we commit to a vendor is how their pricing actually scales once we move past the free tier and start sending real production traffic through it.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "My worry is that the demo always looks clean, but the moment you hit rate limits or a noisy network the whole experience falls apart and the user just blames us.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "If we can get a small pilot in front of the support team first, they will find the rough edges faster than any test plan we could write ourselves.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "So let's scope a two week trial, pick one real workflow, and measure it honestly instead of arguing about hypotheticals in a meeting room.", "gap_ms": 2500},
            ],
        },
        {
            "notes": "Long single-utterance turns, no intra-turn pauses, 2.5s gaps (1:1 with Flux).",
            "turns": [
                {"lead_ms": 300, "text": "The reason I keep coming back to the latency question is that everything else we build sits on top of it, and if the foundation feels slow nothing we add later will fix that impression.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "I would rather ship a smaller feature set that feels instant than a long list of capabilities that each make the user wait a beat longer than they expect.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "We also need to be careful that the metrics we report actually match what a person feels, because an average can hide the handful of slow responses that ruin the whole session.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Let's agree to track the worst case alongside the median, and treat a bad tail as a real bug rather than rounding it away.", "gap_ms": 2500},
            ],
        },
    ],
    "noisy_single": [
        {
            "notes": "Single-speaker turns with noise on the speech only, clean gaps, 2.5s gaps.",
            "noise_snr_db": 15.0,
            "turns": [
                {"lead_ms": 300, "text": "Can you hear me okay over this background noise?", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Good. I'm at the airport, so it's a little loud here.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "I need to change my flight to the morning departure if there's anything available.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Window seat if you can, aisle is fine too.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Perfect, go ahead and book that one. Thank you.", "gap_ms": 2500},
            ],
        },
        {
            "notes": "Single-speaker turns with noise on the speech only, clean gaps, 2.5s gaps.",
            "noise_snr_db": 15.0,
            "turns": [
                {"lead_ms": 300, "text": "Sorry, there's a lot going on around me right now.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "I'm on the train and the next stop is mine.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Can you text me the address so I have it offline?", "gap_ms": 2500},
                {"lead_ms": 0, "text": "And roughly how long is the walk from the station?", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Got it, that helps a lot, thanks.", "gap_ms": 2500},
            ],
        },
        {
            "notes": "Single-speaker turns with noise on the speech only, clean gaps, 2.5s gaps.",
            "noise_snr_db": 15.0,
            "turns": [
                {"lead_ms": 300, "text": "It's pretty windy out here, let me know if I cut out.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "I'm walking the dog and heading back in a few minutes.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Can you remind me what we still need from the store?", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Add coffee to that too while you're at it.", "gap_ms": 2500},
                {"lead_ms": 0, "text": "Perfect, see you soon.", "gap_ms": 2500},
            ],
        },
    ],
    "crosstalk": [
        {
            "notes": "Primary turns with a secondary voice overlaid on distractor spans, 2.5s gaps.",
            "turns": [
                {"lead_ms": 300, "text": "I'd like to book a table for four this Friday evening.", "gap_ms": 2500,
                 "distractors": [{"offset_in_turn_ms": 1100, "text": "no, the other one"}]},
                {"lead_ms": 0, "text": "Around seven if you have it, otherwise seven thirty.", "gap_ms": 2500,
                 "distractors": [{"offset_in_turn_ms": 1300, "text": "ask about parking"}]},
                {"lead_ms": 0, "text": "It's for a birthday, so a quieter table would be great if that's possible.", "gap_ms": 2500,
                 "distractors": []},
                {"lead_ms": 0, "text": "One person in the group is vegetarian, just so the kitchen knows.", "gap_ms": 2500,
                 "distractors": [{"offset_in_turn_ms": 1400, "text": "and gluten free"}]},
                {"lead_ms": 0, "text": "That works, go ahead and confirm it under my name.", "gap_ms": 2500,
                 "distractors": [{"offset_in_turn_ms": 800, "text": "are you sure"}]},
            ],
        },
        {
            "notes": "Primary turns with a secondary voice overlaid on distractor spans, 2.5s gaps.",
            "turns": [
                {"lead_ms": 300, "text": "I'm calling to reschedule my appointment for next week.", "gap_ms": 2500,
                 "distractors": [{"offset_in_turn_ms": 1200, "text": "tell them Tuesday"}]},
                {"lead_ms": 0, "text": "Whatever you have in the late morning would be ideal.", "gap_ms": 2500,
                 "distractors": [{"offset_in_turn_ms": 900, "text": "not too early"}]},
                {"lead_ms": 0, "text": "I'll also need the paperwork sent over beforehand.", "gap_ms": 2500,
                 "distractors": []},
                {"lead_ms": 0, "text": "My insurance changed, so let me give you the new details.", "gap_ms": 2500,
                 "distractors": [{"offset_in_turn_ms": 1300, "text": "the blue card"}]},
                {"lead_ms": 0, "text": "Okay, thanks for fitting me in.", "gap_ms": 2500,
                 "distractors": [{"offset_in_turn_ms": 700, "text": "ask the cost"}]},
            ],
        },
        {
            "notes": "Primary turns with a secondary voice overlaid on distractor spans, 2.5s gaps.",
            "turns": [
                {"lead_ms": 300, "text": "Can you help me return an order that arrived damaged?", "gap_ms": 2500,
                 "distractors": [{"offset_in_turn_ms": 1200, "text": "keep the box"}]},
                {"lead_ms": 0, "text": "The screen was cracked right out of the packaging.", "gap_ms": 2500,
                 "distractors": [{"offset_in_turn_ms": 1000, "text": "take a photo"}]},
                {"lead_ms": 0, "text": "I'd prefer a replacement rather than a refund if I can.", "gap_ms": 2500,
                 "distractors": []},
                {"lead_ms": 0, "text": "How long does the exchange usually end up taking?", "gap_ms": 2500,
                 "distractors": [{"offset_in_turn_ms": 1400, "text": "ask for express"}]},
                {"lead_ms": 0, "text": "Alright, go ahead and start that for me.", "gap_ms": 2500,
                 "distractors": [{"offset_in_turn_ms": 800, "text": "are you sure"}]},
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# Aura synthesis
# --------------------------------------------------------------------------- #
def _aura_pcm(dg, text: str, voice: str) -> np.ndarray:
    """Synthesize `text` with Aura and return an int16 mono array at the sample rate.

    Mirrors the clients.py call shape: dg.speak.v1.audio.generate(text=...,
    model=..., encoding="linear16", container="none", sample_rate=SR), which
    yields raw PCM (pulse-code modulation) audio byte chunks.
    """
    chunks = dg.speak.v1.audio.generate(
        text=text,
        model=voice,
        encoding="linear16",
        container="none",
        sample_rate=SR,
    )
    raw = b"".join(chunk for chunk in chunks if chunk)
    return np.frombuffer(raw, dtype="<i2").astype(np.int16)


def _silence(ms: float) -> np.ndarray:
    return np.zeros(int(round(ms / 1000.0 * SR)), dtype=np.int16)


def _ms(samples: int) -> float:
    return samples / SR * 1000.0


def _overlay(base: np.ndarray, overlay: np.ndarray, at_sample: int, gain: float) -> None:
    """Mix `overlay` (scaled by gain) into `base` in place at `at_sample`."""
    end = min(at_sample + overlay.size, base.size)
    n = end - at_sample
    if n <= 0:
        return
    mixed = base[at_sample:end].astype(np.int32) + (overlay[:n].astype(np.float64) * gain).astype(np.int32)
    base[at_sample:end] = np.clip(mixed, -FULL_SCALE, FULL_SCALE).astype(np.int16)


def _add_noise_bed(
    pcm: np.ndarray,
    snr_db: float,
    seed: int,
    speech_spans: list[tuple[int, int]],
    fade_ms: float = 30.0,
) -> np.ndarray:
    """Add broadband noise at the given SNR (signal-to-noise ratio), but ONLY over the speech regions.

    The original version laid noise across the WHOLE clip, including the
    2.5s inter-turn gaps. Flux never saw clean silence between turns, so on the
    noisy class it spuriously split/merged turns (n != annotated, impossible
    negative detection). Gating the noise to the speech spans (with a short
    raised-cosine fade at each edge to avoid click onsets) keeps the gaps clean
    so Flux segments 1:1, while the speech itself is still a real SNR test.
    """
    speech = pcm.astype(np.float64)
    voiced = speech[speech != 0]
    # RMS = root-mean-square (signal energy). Set the noise level relative to the
    # speech RMS so the mix lands at the target signal-to-noise ratio.
    speech_rms = float(np.sqrt(np.mean(voiced ** 2))) if voiced.size else 1000.0
    noise_rms = speech_rms / (10 ** (snr_db / 20.0))

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_rms, size=pcm.size)

    # Gate: 1.0 inside speech spans, 0.0 in the gaps, raised-cosine fades at edges.
    mask = np.zeros(pcm.size, dtype=np.float64)
    fade = max(1, int(round(fade_ms / 1000.0 * SR)))
    for start, end in speech_spans:
        start = max(0, int(start))
        end = min(pcm.size, int(end))
        if end <= start:
            continue
        mask[start:end] = 1.0
        f = min(fade, (end - start) // 2)
        if f > 0:
            ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, f)))
            mask[start:start + f] = np.minimum(mask[start:start + f], ramp)
            mask[end - f:end] = np.minimum(mask[end - f:end], ramp[::-1])

    mixed = np.clip(speech + noise * mask, -FULL_SCALE, FULL_SCALE)
    return mixed.astype(np.int16)


# --------------------------------------------------------------------------- #
# Build one clip
# --------------------------------------------------------------------------- #
def build_clip(dg, audio_class: str, clip_def: dict, clip_id: str) -> tuple[np.ndarray, ClipSpec]:
    timeline: list[np.ndarray] = []
    cursor = 0  # running sample position
    turns: list[TurnSpec] = []
    distractor_jobs: list[tuple[int, np.ndarray]] = []  # (at_sample, pcm) overlays
    speech_spans: list[tuple[int, int]] = []  # (start, end) sample ranges of actual speech

    def emit(arr: np.ndarray) -> None:
        nonlocal cursor
        timeline.append(arr)
        cursor += arr.size

    for idx, t in enumerate(clip_def["turns"]):
        if t.get("lead_ms"):
            emit(_silence(t["lead_ms"]))

        speech_start = cursor
        turn_start = cursor
        emit(_aura_pcm(dg, t["text"], PRIMARY_VOICE))
        for d in t.get("distractors", []):
            at = turn_start + int(round(d["offset_in_turn_ms"] / 1000.0 * SR))
            distractor_jobs.append((at, _aura_pcm(dg, d["text"], SECONDARY_VOICE)))

        speech_spans.append((speech_start, cursor))
        true_end_ms = _ms(cursor)

        # Crosstalk: record spans (clip-relative) on the turn they overlap.
        dspans = []
        for d in t.get("distractors", []):
            start = _ms(speech_start + int(round(d["offset_in_turn_ms"] / 1000.0 * SR)))
            dspans.append({"_offset_in_turn_ms": d["offset_in_turn_ms"], "start_ms": round(start, 1)})

        turns.append(TurnSpec(
            turn_index=idx,
            true_end_ms=round(true_end_ms, 1),
            transcript=t.get("text", ""),
            speaker="primary",
            distractor_spans=dspans,
        ))

        if t.get("gap_ms"):
            emit(_silence(t["gap_ms"]))

    pcm = np.concatenate(timeline) if timeline else np.zeros(0, dtype=np.int16)

    # Mix distractor overlays and finalize their end_ms.
    span_iter = [s for tr in turns for s in tr.distractor_spans]
    for (at_sample, dpcm), span in zip(distractor_jobs, span_iter):
        _overlay(pcm, dpcm, at_sample, gain=0.7)
        span["end_ms"] = round(_ms(at_sample + dpcm.size), 1)
        span.pop("_offset_in_turn_ms", None)

    if "noise_snr_db" in clip_def:
        seed = int(hashlib.sha256(clip_id.encode()).hexdigest(), 16) & 0xFFFF
        pcm = _add_noise_bed(pcm, clip_def["noise_snr_db"], seed, speech_spans)

    clip = ClipSpec(
        clip_id=clip_id,
        audio_class=audio_class,
        audio_file=f"{clip_id}.wav",
        sample_rate=SR,
        encoding="linear16",
        duration_ms=round(_ms(pcm.size), 1),
        turns=turns,
        notes=clip_def["notes"],
    )
    return pcm, clip


# --------------------------------------------------------------------------- #
# WAV (waveform audio) + sidecar writers
# --------------------------------------------------------------------------- #
def write_wav(path: Path, pcm: np.ndarray) -> None:
    try:
        import soundfile as sf
        sf.write(str(path), pcm, SR, subtype="PCM_16")
    except Exception:  # soundfile/libsndfile missing - fall back to stdlib wave
        import wave
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.astype("<i2").tobytes())


def write_sidecar(path: Path, clip: ClipSpec) -> None:
    d = asdict(clip)
    path.write_text(json.dumps(d, indent=2) + "\n")


def rewrite_manifest(audio_dir: Path, clip_ids: list[str]) -> None:
    manifest = {
        "version": 1,
        "generated_by": "audio/generate_audio.py (real Aura clips)",
        "clips": [f"clips/{cid}.json" for cid in clip_ids],
    }
    (audio_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def append_manifest(audio_dir: Path, clip_ids: list[str]) -> int:
    """Add new sidecar refs to an existing manifest without dropping the base clips.

    Returns the count newly added. Used by the --defs path so generated condition
    clips join the sweep alongside the built-in classes instead of replacing them.
    """
    manifest_path = audio_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"version": 1, "generated_by": "audio/generate_audio.py", "clips": []}
    clips = manifest.get("clips", [])
    added = 0
    for cid in clip_ids:
        ref = f"clips/{cid}.json"
        if ref not in clips:
            clips.append(ref)
            added += 1
    manifest["clips"] = clips
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return added


# --------------------------------------------------------------------------- #
# CLI (command-line interface)
# --------------------------------------------------------------------------- #
# BYO = bring-your-own (audio).
BYO_INSTRUCTIONS = """\
DEEPGRAM_API_KEY is not set, so no audio was generated.

You have two options:

1) Generate the clips (recommended): export a Deepgram key and re-run.
     export DEEPGRAM_API_KEY=...        # see the README
     python audio/generate_audio.py

2) Bring your own audio. For each class in {clean_short, clean_long,
   noisy_single, crosstalk}:
     - Provide 16 kHz MONO linear16 WAV(s) at audio/<clip_id>.wav
     - Write a sidecar at audio/clips/<clip_id>.json following audio/spec.md,
       with audio_file set to "<clip_id>.wav" and an exact true_end_ms per turn.
     - Point audio/manifest.json at your sidecars.

Either way, the committed sidecars already let `bench.py --mock`
and the smoke test run with NO key and NO audio. You only need real clips for a
real (measured) run.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the Flux EOT audio classes via Aura.")
    ap.add_argument("--class", dest="only", choices=config.AUDIO_CLASSES, help="generate just one class")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent), help="audio/ directory")
    ap.add_argument(
        "--defs",
        default=None,
        help=(
            "path to a generated clip-defs JSON, shape {audio_class: [clip_def, ...]}, "
            "e.g. from the scaffold-test-clips skill. Renders these instead of the "
            "built-in CLIP_DEFS and APPENDS the new clips to manifest.json so they "
            "join the sweep alongside the base classes."
        ),
    )
    args = ap.parse_args(argv)

    audio_dir = Path(args.out_dir)
    clips_dir = audio_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        print(BYO_INSTRUCTIONS)
        return 0

    try:
        from deepgram import DeepgramClient
    except ImportError:
        print("deepgram-sdk is not installed. `pip install -r requirements.txt` first.", file=sys.stderr)
        return 1

    dg = DeepgramClient(api_key=key)

    if args.defs:
        defs_path = Path(args.defs)
        if not defs_path.exists():
            print(f"clip-defs file not found: {defs_path}", file=sys.stderr)
            return 1
        class_defs: dict[str, list[dict]] = json.loads(defs_path.read_text())
        print(
            f"[generate] loaded clip-defs from {defs_path}: "
            f"{sum(len(v) for v in class_defs.values())} clip(s) across "
            f"{len(class_defs)} class(es)"
        )
    else:
        selected = [args.only] if args.only else list(config.AUDIO_CLASSES)
        class_defs = {c: CLIP_DEFS[c] for c in selected}

    written_ids: list[str] = []
    for audio_class, clip_defs in class_defs.items():
        for i, clip_def in enumerate(clip_defs, start=1):
            clip_id = f"{audio_class}_{i}"
            print(f"[generate] {clip_id} ...", flush=True)
            pcm, clip = build_clip(dg, audio_class, clip_def, clip_id)
            write_wav(audio_dir / f"{clip_id}.wav", pcm)
            write_sidecar(clips_dir / f"{clip_id}.json", clip)
            written_ids.append(clip_id)
            print(f"  wrote {clip_id}.wav ({clip.duration_ms:.0f} ms, {len(clip.turns)} turns) + clips/{clip_id}.json")

    # --defs appends (join the sweep). A full built-in run rewrites. A single
    # built-in --class run leaves the manifest alone so it doesn't orphan clips.
    if args.defs:
        added = append_manifest(audio_dir, written_ids)
        print(
            f"[generate] appended {added} new clip(s) to manifest.json "
            f"({len(written_ids)} generated)"
        )
    elif not args.only:
        rewrite_manifest(audio_dir, written_ids)
        print(f"[generate] rewrote manifest.json -> {len(written_ids)} clips")
    else:
        print("[generate] single class generated; manifest.json left unchanged")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
