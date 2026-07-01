#!/usr/bin/env python3
"""Annotation assist for bring-your-own audio (SCAFFOLD, starting point).

The harness measures Flux turn detection against a ground-truth `true_end_ms` per
turn. For the synthetic clips in `generate_audio.py` that value is exact by
construction (the generator placed every sample, so it knows where speech stops).
A real recording has no such luxury, so this script drafts the sidecar for you
using Deepgram's OWN pre-recorded speech-to-text (NOT Flux): it transcribes the
file with word-level timestamps and utterance segmentation, then proposes one
turn per utterance with `true_end_ms` set to the end of that utterance's last
word. The transcript comes along for free.

IMPORTANT, read this:
  - This is a DRAFT, not ground truth. The human is the final authority on
    `true_end_ms`. A word-end timestamp is close to, but not exactly, the true
    end of speech (there is trailing breath and filler), so listen and correct.
  - Synthetic clips are exact; this is approximate. The +/-200ms tolerance window
    applied at analysis time (analysis.py) absorbs small errors, so you do not
    need sample-perfect labels, just honest ones.
  - It never auto-accepts. You review the emitted sidecar before running a sweep.

Scope note: this is a deliberately small scaffold. It does not include a playback
or click-to-edit user interface, does not resample audio (bring a 16 kHz mono
linear16 WAV; see audio/annotate-your-audio.md for the ffmpeg one-liner), and
does not try to be perfect on overlapping speech. It gets you 90% of the way to a
sidecar so the review is a few nudges, not hours in an audio editor.

Usage:
    python audio/annotate_real.py path/to/recording.wav --class noisy_single
    python audio/annotate_real.py call.wav --class crosstalk --primary-speaker 0

Needs DEEPGRAM_API_KEY (pre-recorded speech-to-text). See the README.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flux_eot_lab import config  # noqa: E402

# Zero-dependency .env load, mirroring the rest of the repo (config.load_env
# falls back to a manual parse when python-dotenv is absent).
config.load_env(_REPO_ROOT / ".env")  # noqa: E402

REQUIRED_RATE = config.INPUT_SAMPLE_RATE  # 16000
DEFAULT_STT_MODEL = "nova-2"  # widely available; nova-3 also works if enabled


def validate_wav(path: Path) -> float:
    """Confirm the WAV is 16 kHz mono linear16 (what the harness requires).

    Returns the duration in milliseconds. Raises ValueError with a clear, actionable
    message (pointing at the ffmpeg conversion) if the format is wrong. We do NOT
    resample here on purpose: a silent resample would hide a format mistake.
    """
    if not path.exists():
        raise ValueError(f"file not found: {path}")
    try:
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            rate = w.getframerate()
            width = w.getsampwidth()
            frames = w.getnframes()
    except wave.Error as exc:
        raise ValueError(
            f"{path} is not a readable WAV ({exc}). Convert it first, see "
            "audio/annotate-your-audio.md for the ffmpeg one-liner."
        ) from exc

    problems = []
    if channels != 1:
        problems.append(f"channels={channels} (need mono / 1)")
    if rate != REQUIRED_RATE:
        problems.append(f"sample_rate={rate} (need {REQUIRED_RATE})")
    if width != 2:
        problems.append(f"sample_width={width * 8}-bit (need 16-bit linear16)")
    if problems:
        raise ValueError(
            f"{path} is not 16 kHz mono linear16: " + "; ".join(problems) + ".\n"
            "Convert it first:\n"
            f"  ffmpeg -i {path.name} -ac 1 -ar {REQUIRED_RATE} -sample_fmt s16 converted.wav\n"
            "See audio/annotate-your-audio.md."
        )
    return frames / rate * 1000.0


def transcribe(path: Path, model: str) -> dict:
    """Run Deepgram pre-recorded speech-to-text with diarization + utterances.

    Returns the response normalized to a plain dict (the stable Deepgram JSON
    shape: results.utterances[] each with start, end, transcript, speaker).
    """
    import os

    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise SystemExit(
            "DEEPGRAM_API_KEY is not set. Put it in .env (see the README). "
            "This step uses pre-recorded speech-to-text, not Flux."
        )
    try:
        from deepgram import DeepgramClient
    except ImportError as exc:
        raise SystemExit(
            "deepgram-sdk is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    dg = DeepgramClient(api_key=key)
    audio_bytes = path.read_bytes()
    try:
        resp = dg.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model=model,
            diarize=True,        # speaker numbers, so we can separate primary vs crosstalk
            utterances=True,     # segment into utterances == our turn proposals
            punctuate=True,
            smart_format=True,
            filler_words=True,   # keep "um"/"uh" so a turn-end is not clipped early
        )
    except Exception as exc:  # surface the real failure, no silent fallback
        raise SystemExit(f"Deepgram transcription failed: {type(exc).__name__}: {exc}") from exc
    return _to_plain_dict(resp)


def _to_plain_dict(resp) -> dict:
    """Normalize the SDK response object to the documented JSON dict.

    NOTE: written against Deepgram's stable REST JSON shape (results.utterances).
    The exact SDK accessor is not exercised here (this scaffold makes no test call),
    so if a future SDK changes the response wrapper, this is the one spot to adjust.
    """
    for attr in ("model_dump", "dict"):
        fn = getattr(resp, attr, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                try:
                    return fn(mode="json")
                except Exception:
                    pass
    if isinstance(resp, dict):
        return resp
    to_json = getattr(resp, "json", None)
    if callable(to_json):
        try:
            return json.loads(to_json())
        except Exception:
            pass
    raise SystemExit(
        "Could not read the Deepgram response shape. Inspect it and adjust "
        "_to_plain_dict() in annotate_real.py."
    )


def utterances_of(data: dict) -> list[dict]:
    results = (data or {}).get("results") or {}
    utts = results.get("utterances") or []
    if not utts:
        raise SystemExit(
            "No utterances came back. Check the audio has speech, or pass a "
            "different --model. (utterances=True is required for turn proposals.)"
        )
    return sorted(utts, key=lambda u: float(u.get("start", 0.0)))


def pick_primary_speaker(utts: list[dict], override: int | None) -> int:
    """The primary speaker is the one who talks the most (by total duration),
    unless the user overrides it. Other speakers become crosstalk distractors."""
    if override is not None:
        return override
    talk: dict[int, float] = {}
    for u in utts:
        spk = int(u.get("speaker", 0) or 0)
        talk[spk] = talk.get(spk, 0.0) + (float(u.get("end", 0.0)) - float(u.get("start", 0.0)))
    return max(talk, key=talk.get) if talk else 0


def build_turns(utts: list[dict], audio_class: str, primary: int) -> list[dict]:
    """Draft the turn list. Every primary-speaker utterance is a turn; for the
    crosstalk class, other-speaker utterances become distractor_spans on the
    nearest preceding primary turn."""
    is_crosstalk = audio_class == "crosstalk"
    turns: list[dict] = []
    pending_distractors: list[dict] = []

    for u in utts:
        spk = int(u.get("speaker", 0) or 0)
        start_ms = round(float(u.get("start", 0.0)) * 1000.0, 1)
        end_ms = round(float(u.get("end", 0.0)) * 1000.0, 1)
        transcript = (u.get("transcript") or "").strip()

        if is_crosstalk and spk != primary:
            # A secondary voice: record it as a distractor span to attach to the
            # most recent primary turn once we have one.
            pending_distractors.append({"start_ms": start_ms, "end_ms": end_ms})
            continue

        turn = {
            "turn_index": len(turns),
            "true_end_ms": end_ms,           # PROPOSED: end of this utterance's last word
            "transcript": transcript,
            "speaker": "primary",
            "distractor_spans": [],
            "_draft_review": "confirm true_end_ms by ear; trim trailing breath/filler",
        }
        turns.append(turn)
        # Flush any secondary spans that occurred before this turn onto it.
        if pending_distractors:
            turn["distractor_spans"] = pending_distractors
            pending_distractors = []

    # Trailing secondary spans (after the last primary turn) attach to the last turn.
    if pending_distractors and turns:
        turns[-1]["distractor_spans"].extend(pending_distractors)
    return turns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Draft a ground-truth sidecar for a real recording using Deepgram "
        "pre-recorded speech-to-text. Output is a DRAFT to review, not final."
    )
    ap.add_argument("wav", type=Path, help="16 kHz mono linear16 WAV (see the guide to convert)")
    ap.add_argument(
        "--class", dest="audio_class", required=True, choices=config.AUDIO_CLASSES,
        help="which audio class this recording represents",
    )
    ap.add_argument("--out", type=Path, default=None, help="sidecar path (default: audio/clips/<stem>.json)")
    ap.add_argument("--model", default=DEFAULT_STT_MODEL, help=f"STT model (default {DEFAULT_STT_MODEL})")
    ap.add_argument(
        "--primary-speaker", type=int, default=None,
        help="speaker number to treat as primary (default: whoever talks most). "
        "Only matters for the crosstalk class.",
    )
    args = ap.parse_args(argv)

    try:
        duration_ms = validate_wav(args.wav)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    print(f"[annotate] transcribing {args.wav.name} with {args.model} (diarize + utterances) ...", flush=True)
    data = transcribe(args.wav, args.model)
    utts = utterances_of(data)
    primary = pick_primary_speaker(utts, args.primary_speaker)
    turns = build_turns(utts, args.audio_class, primary)
    if not turns:
        print(
            f"[error] no primary-speaker turns found (primary speaker = {primary}). "
            "Try --primary-speaker, or check the audio.",
            file=sys.stderr,
        )
        return 2

    stem = args.wav.stem
    sidecar = {
        "clip_id": stem,
        "audio_class": args.audio_class,
        "audio_file": args.wav.name,  # place the WAV in audio/ next to the sidecar's audio dir
        "sample_rate": REQUIRED_RATE,
        "encoding": "linear16",
        "duration_ms": round(duration_ms, 1),
        "notes": (
            "DRAFT from annotate_real.py (Deepgram STT-assisted, primary speaker "
            f"{primary}). REVIEW every true_end_ms by ear before trusting it: a word-end "
            "is approximate, the human is the authority. Synthetic clips are exact; this "
            "is not. The +/-200ms analysis tolerance absorbs small errors."
        ),
        "turns": turns,
    }

    out = args.out or (_REPO_ROOT / "audio" / "clips" / f"{stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(f"\n[annotate] wrote DRAFT sidecar: {out}")
    print(f"  {len(turns)} turn(s) proposed across {duration_ms / 1000:.1f}s "
          f"(class={args.audio_class}, primary speaker={primary}).")
    print("\n  NEXT: this is a DRAFT. Do this before any sweep:")
    print("   1. Put the WAV at audio/<name>.wav (or set audio_file to its path).")
    print("   2. Open the sidecar, play the clip, and correct each true_end_ms")
    print("      (the proposed value is the end of the last word; trim trailing breath/filler).")
    print("   3. Remove the _draft_review markers once each turn is confirmed.")
    print("   4. Register it in audio/manifest.json, then run bench.py.")
    print("  See audio/annotate-your-audio.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
