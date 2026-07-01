"""Real Deepgram Flux source for the detection-only harness.

`build_real_sources()` wires the single injected seam the agent depends on:
  - DeepgramFluxSource : Flux speech-to-text over the listen.v2 WebSocket,
                         replaying a clip WAV in ~100 ms chunks paced to wall
                         clock, and yielding the turn-detection event stream.

Scope: this harness measures Flux turn detection only. The downstream
large-language-model and text-to-speech pipeline is out of scope and is not
wired here.

Heavy third-party imports (deepgram, soundfile) are done lazily inside the
functions and methods so this module imports cleanly under `--mock` on a box
with only the standard library installed. A real run needs the dependencies in
requirements.txt.

The connect / EventType.MESSAGE / dict-event / send_media / start_listening
pattern is the one verified against a working Flux voice loop.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Iterator

from .config import (
    AURA_SAMPLE_RATE,
    DEFAULT_AURA_MODEL,
    DEFAULT_FLUX_MODEL,
    INPUT_SAMPLE_RATE,
)
from .records import EventStamp
from .agent import Sources

# Flux TurnInfo event names we care about (verified, R4).
_FLUX_EVENTS = frozenset(
    {"StartOfTurn", "Update", "EagerEndOfTurn", "TurnResumed", "EndOfTurn"}
)

# 16 kHz, 16-bit mono => 3200 bytes per 100 ms of audio.
_CHUNK_SEC = 0.1
_CHUNK_BYTES = int(INPUT_SAMPLE_RATE * 2 * _CHUNK_SEC)

_SENTINEL = object()


class DeepgramFluxSource:
    """Streams Flux turn events for a clip, replayed from its annotated WAV.

    Implements the FluxSource Protocol: `stream(clip, point)` yields EventStamp
    objects whose `t_rel_ms` is stamped relative to clip start.
    """

    def __init__(self, dg_client, flux_model: str = DEFAULT_FLUX_MODEL) -> None:
        self._dg = dg_client
        self._model = flux_model

    def stream(self, clip, point) -> Iterator[EventStamp]:
        import soundfile as sf  # lazy: needs libsndfile, real-run only
        from deepgram.core.events import EventType

        if not clip.audio_file:
            raise ValueError(
                f"clip {clip.clip_id!r} has no audio_file; a real run needs a WAV "
                f"(use --mock for a keyless wiring check)"
            )

        audio_bytes = self._read_pcm16(sf, clip.audio_file)

        connect_kwargs = {
            "model": self._model,
            "encoding": "linear16",
            "sample_rate": INPUT_SAMPLE_RATE,
            "eot_threshold": point.eot_threshold,
            "eot_timeout_ms": point.eot_timeout_ms,
        }
        # Pass eager_eot_threshold ONLY when set; unset disables eager EOT.
        if point.eager_eot_threshold is not None:
            connect_kwargs["eager_eot_threshold"] = point.eager_eot_threshold

        events: "queue.Queue" = queue.Queue()
        turn_idx = {"value": -1}

        with self._dg.listen.v2.connect(**connect_kwargs) as connection:
            clip_start = time.monotonic()

            def on_message(message) -> None:
                if not isinstance(message, dict):
                    return
                ev_name = message.get("event")
                if ev_name not in _FLUX_EVENTS:
                    return
                if ev_name == "StartOfTurn":
                    turn_idx["value"] += 1
                t_rel_ms = (time.monotonic() - clip_start) * 1000.0
                events.put(
                    EventStamp(
                        event=ev_name,
                        t_rel_ms=t_rel_ms,
                        transcript=message.get("transcript", "") or "",
                        end_of_turn_confidence=message.get("end_of_turn_confidence"),
                        turn_index=max(turn_idx["value"], 0),
                    )
                )

            connection.on(EventType.MESSAGE, on_message)

            # Wait long enough after the last audio for the trailing EndOfTurn to
            # fire (driven by eot_timeout_ms), capped so a bad clip can't hang.
            flush_wait = min(point.eot_timeout_ms / 1000.0 + 1.0, 8.0)

            def send_audio() -> None:
                try:
                    for i, start in enumerate(range(0, len(audio_bytes), _CHUNK_BYTES)):
                        connection.send_media(audio_bytes[start : start + _CHUNK_BYTES])
                        # Pace against an ABSOLUTE schedule tied to clip_start so the
                        # audio position tracks wall-clock. Sleeping a fixed _CHUNK_SEC
                        # after each send drifts by the per-send overhead, which
                        # accumulates down the clip and inflates later detection times
                        # (the event clock is wall-clock since clip_start).
                        target = clip_start + (i + 1) * _CHUNK_SEC
                        delay = target - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                    time.sleep(flush_wait)
                finally:
                    events.put(_SENTINEL)

            listener = threading.Thread(target=connection.start_listening, daemon=True)
            sender = threading.Thread(target=send_audio, daemon=True)
            listener.start()
            sender.start()

            while True:
                item = events.get()
                if item is _SENTINEL:
                    break
                yield item

    @staticmethod
    def _read_pcm16(sf, path: str) -> bytes:
        """Load a WAV as mono 16 kHz linear16 PCM bytes."""
        import numpy as np

        data, sr = sf.read(path, dtype="int16", always_2d=False)
        if getattr(data, "ndim", 1) > 1:  # downmix to mono
            data = data.mean(axis=1).astype("int16")
        if sr != INPUT_SAMPLE_RATE:
            raise ValueError(
                f"{path}: sample rate {sr} != required {INPUT_SAMPLE_RATE} Hz; "
                f"resample the clip before benching"
            )
        return np.ascontiguousarray(data).tobytes()


def build_real_sources(
    dg_key: str,
    flux_model: str = DEFAULT_FLUX_MODEL,
) -> Sources:
    """Wire the real Deepgram Flux source into the agent's seam.

    Detection-only: a real run needs DEEPGRAM_API_KEY and nothing else.
    """
    if not dg_key:
        raise ValueError("DEEPGRAM_API_KEY is required for a real run (see the README)")

    from deepgram import DeepgramClient

    dg = DeepgramClient(api_key=dg_key)
    return Sources(flux=DeepgramFluxSource(dg, flux_model))


def measure_rtt(dg_key: str) -> float:
    """Round-trip sanity check for bench.py --ping: seconds to the first byte of a
    tiny Aura request. Confirms the Deepgram key is live and reports the network
    round-trip that belongs in the README disclosure header."""
    if not dg_key:
        raise ValueError("DEEPGRAM_API_KEY is required for --ping")

    from deepgram import DeepgramClient

    dg = DeepgramClient(api_key=dg_key)
    t0 = time.monotonic()
    generator = dg.speak.v1.audio.generate(
        text="ping",
        model=DEFAULT_AURA_MODEL,
        encoding="linear16",
        container="none",
        sample_rate=AURA_SAMPLE_RATE,
    )
    for _chunk in generator:
        return time.monotonic() - t0
    return time.monotonic() - t0
