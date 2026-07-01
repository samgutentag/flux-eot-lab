"""Instrumented Flux turn-detection agent.

`InstrumentedFluxAgent.run_clip` consumes a `FluxSource` event stream for one
clip and sweep point and records every Flux event into a per-turn `TurnRecord`.
The headline measurement is DETECTION latency: the signed gap between the user's
annotated true end of speech and the first end-of-turn event (EagerEndOfTurn or
EndOfTurn) Flux fires for that turn.

Scope: the agent measures Flux turn detection only. The downstream
large-language-model and text-to-speech pipeline is out of scope (it is the
integrator's choice and varies independently of the threshold being tuned), so
the agent does not run it.

DETECTION timing comes from the source-stamped `EventStamp.t_rel_ms`, which is
already clip-relative off a single monotonic clock zeroed at clip start.

NO classification and NO tolerance are applied here. This module captures RAW
per-turn data only; analysis.py classifies and applies the tolerance window at
analysis time so both stay sweepable without re-running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .records import FluxSource, TurnRecord

_TRIGGER_EVENTS = ("EagerEndOfTurn", "EndOfTurn")


@dataclass
class Sources:
    """The single injected seam the agent depends on (the FluxSource Protocol)."""

    flux: FluxSource


class InstrumentedFluxAgent:
    def __init__(self, sources: Sources) -> None:
        self.sources = sources

    def run_clip(self, clip, point) -> list[TurnRecord]:
        """Replay one clip under one sweep point and return RAW per-turn records."""
        # Honesty guardrail: a mock FluxSource carries mock=True. Every record this
        # run emits inherits it, so analysis/README can never mistake wiring-fixture
        # timing for a measurement. Real sources have no such flag -> mock=False.
        is_mock = bool(getattr(self.sources.flux, "mock", False))

        true_end_by_idx = {t.turn_index: t.true_end_ms for t in clip.turns}

        turns: dict[int, dict] = {}
        order: list[int] = []

        def get_turn(idx: int) -> dict:
            if idx not in turns:
                turns[idx] = {
                    "events": [],
                    "eager": None,
                    "eot": None,
                    "resumed": None,
                    "triggered": False,
                    "first_eot_event_ms": None,
                    "final_transcript": "",
                    "eager_fired": False,
                    "turn_resumed_fired": False,
                    "error": None,
                }
                order.append(idx)
            return turns[idx]

        stream_error: str | None = None
        try:
            for ev in self.sources.flux.stream(clip, point):
                st = get_turn(ev.turn_index)
                st["events"].append(ev)

                if ev.event == "EagerEndOfTurn":
                    st["eager_fired"] = True
                    if st["eager"] is None:
                        st["eager"] = ev.t_rel_ms
                elif ev.event == "EndOfTurn":
                    if st["eot"] is None:
                        st["eot"] = ev.t_rel_ms
                elif ev.event == "TurnResumed":
                    st["turn_resumed_fired"] = True
                    if st["resumed"] is None:
                        st["resumed"] = ev.t_rel_ms

                # Freeze the transcript at the first end-of-turn event; later
                # trailing Updates (often the next turn's first word) must not
                # clobber it.
                if ev.transcript and not st["triggered"]:
                    st["final_transcript"] = ev.transcript

                # Mark the detection point once per turn, on the earliest
                # end-of-turn-type event.
                if not st["triggered"] and ev.event in _TRIGGER_EVENTS:
                    st["triggered"] = True
                    st["first_eot_event_ms"] = ev.t_rel_ms
        except Exception as exc:  # never crash the sweep
            stream_error = f"{type(exc).__name__}: {exc}"

        if stream_error is not None:
            if order:
                last = turns[order[-1]]
                if last["error"] is None:
                    last["error"] = stream_error
            else:
                # Stream blew up before any event arrived: emit one error record
                # so the failure is visible in the raw data, not swallowed.
                st = get_turn(0)
                st["error"] = stream_error

        return [self._build_record(clip, point, idx, turns[idx], true_end_by_idx, is_mock) for idx in order]

    @staticmethod
    def _build_record(clip, point, idx: int, st: dict, true_end_by_idx: dict, is_mock: bool = False) -> TurnRecord:
        true_end = true_end_by_idx.get(idx)
        first_eot = st["first_eot_event_ms"]
        if first_eot is not None and true_end is not None:
            detection_delta = first_eot - true_end
        else:
            detection_delta = None

        return TurnRecord(
            clip_id=clip.clip_id,
            audio_class=clip.audio_class,
            turn_index=idx,
            eot_threshold=point.eot_threshold,
            eager_eot_threshold=point.eager_eot_threshold,
            eot_timeout_ms=point.eot_timeout_ms,
            true_end_ms=true_end if true_end is not None else float("nan"),
            events=st["events"],
            eager_eot_ms=st["eager"],
            end_of_turn_ms=st["eot"],
            turn_resumed_ms=st["resumed"],
            first_eot_event_ms=first_eot,
            detection_delta_ms=detection_delta,
            final_transcript=st["final_transcript"],
            eager_fired=st["eager_fired"],
            turn_resumed_fired=st["turn_resumed_fired"],
            mock=is_mock,
            error=st["error"],
        )
