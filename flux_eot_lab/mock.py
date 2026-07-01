"""Deterministic mock FluxSource for the Flux end-of-turn lab.

A single fake that satisfies the FluxSource Protocol from
``flux_eot_lab.records`` so the whole harness runs with NO application
programming interface key and NO WAV files. ``build_mock_sources()`` returns a
``Sources`` the agent drives exactly like the real one; swapping mock for real
means passing a key, nothing else.

HONESTY GUARDRAIL
-----------------
Everything here is a WIRING FIXTURE, never a measurement. The synthetic event
timing is plausibly *shaped* (so curves bend and every classification category
populates) but it is not data. ``MockFluxSource`` carries a ``mock = True`` flag
so the agent stamps ``mock=True`` on every ``TurnRecord`` it emits on this path;
analysis may compute over it to exercise the code, but the README's measured
cells stay PENDING_REAL_RUN. No reported number ever comes from this file.

DETERMINISM
-----------
Output is identical across runs (and processes). The spec says "seeded by
hash(clip_id)" - we use ``hashlib.sha256`` rather than the builtin ``hash`` so
the seed is stable across interpreter runs (builtin str hashing is randomized
by PYTHONHASHSEED). Per-clip "personality" and per-turn jitter are derived from
that stable seed and are *independent of the sweep point*, so the only thing the
SweepPoint moves is the systematic detection offset -> clean, monotonic curves.

Pure stdlib. No SDK imports.
"""

from __future__ import annotations

import hashlib
from random import Random
from typing import Iterator

from flux_eot_lab.records import ClipSpec, EventStamp, TurnSpec
from flux_eot_lab.agent import Sources

# --------------------------------------------------------------------------
# Tunable shape constants. These move WHERE synthetic events land relative to
# the annotated true end; they are not measurements. The goal is only that
# across sweep_grid() x the audio classes, all four classify() categories
# (clean / hard_cutoff / near_miss / late) reliably appear.
# --------------------------------------------------------------------------

# EndOfTurn offset from true_end, as a function of eot_threshold.
# Higher threshold -> Flux waits for more confidence -> EndOfTurn fires LATER
# (nearer/after the true end). Lower threshold -> fires EARLIER -> hard cutoffs.
_EOT_SLOPE_MS = 1600.0          # ms of EndOfTurn shift per 1.0 of eot_threshold
_EOT_INTERCEPT_MS = 40.0        # offset at the default threshold (0.7)

# Per-class difficulty bias (ms). Negative = detection skews earlier = more
# hard cutoffs. Crosstalk (a second voice) and noise trip the endpointer early;
# clean/short is the easy case. This is the "crosstalk robustness" axis (D-spec),
# not diarization.
_CLASS_BIAS_MS = {
    "clean_short": 60.0,
    "clean_long": -40.0,
    "noisy_single": -140.0,
    "crosstalk": -220.0,
}
_CLASS_BIAS_DEFAULT = 0.0

# Eager lead: how far before EndOfTurn the (speculative) EagerEndOfTurn fires.
# Grows with the gap between eot_threshold and eager_eot_threshold - a more
# aggressive (lower) eager threshold fires sooner and is likelier premature.
_EAGER_BASE_LEAD_MS = 200.0
_EAGER_GAP_GAIN_MS = 900.0      # extra lead per 1.0 of (eot - eager) gap

# If the eager fire lands more than this far before the true end, the speaker
# was still talking -> Flux retracts with TurnResumed (a near_miss / wasted
# eager). Chosen to sit just under the default tolerance so the raw geometry
# and the analysis-time classification line up sensibly.
_PREMATURE_MARGIN_MS = 150.0

# Spread.
_PERSONALITY_MS = 80.0          # per-clip constant offset, +/- this
_TURN_JITTER_MS = 60.0          # per-turn noise, +/- this
_EAGER_JITTER_MS = 40.0

def _stable_seed(*parts: object) -> int:
    """A process-stable integer seed derived from the given parts.

    Uses sha256 instead of builtin hash() so seeding is reproducible across
    interpreter runs (str hashing is salted per-process by default).
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class MockFluxSource:
    """Synthesizes a Flux turn-event timeline per TurnSpec - it reads NO audio.

    For each turn it emits StartOfTurn -> a few Updates -> optional
    EagerEndOfTurn (+ optional TurnResumed when the eager was premature) ->
    EndOfTurn, with ``t_rel_ms`` a deterministic function of ``true_end_ms`` and
    the ``SweepPoint``. The earliest of EagerEndOfTurn / EndOfTurn is the
    detection mark the agent records; classification and tolerance are applied
    later, in analysis, off the raw events captured here.
    """

    # The agent reads this flag and stamps mock=True on every TurnRecord it
    # produces from this source (honesty guardrail).
    mock = True

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    # -- internal: where does EndOfTurn land for this turn at this point? ----
    def _end_of_turn_ms(self, clip: ClipSpec, turn: TurnSpec, eot_threshold: float) -> float:
        base = (eot_threshold - 0.7) * _EOT_SLOPE_MS + _EOT_INTERCEPT_MS
        class_bias = _CLASS_BIAS_MS.get(turn_class(clip, turn), _CLASS_BIAS_DEFAULT)

        clip_rng = Random(_stable_seed(self.seed, clip.clip_id))
        personality = clip_rng.uniform(-_PERSONALITY_MS, _PERSONALITY_MS)

        turn_rng = Random(_stable_seed(self.seed, clip.clip_id, turn.turn_index, "jitter"))
        jitter = turn_rng.uniform(-_TURN_JITTER_MS, _TURN_JITTER_MS)

        return turn.true_end_ms + base + class_bias + personality + jitter

    def stream(self, clip: ClipSpec, point) -> Iterator[EventStamp]:
        prev_end_ms = 0.0
        for turn in clip.turns:
            words = (turn.transcript or "").split()
            speech_dur = max(600.0, len(words) * 280.0)

            end_of_turn_ms = self._end_of_turn_ms(clip, turn, point.eot_threshold)

            # Place the start of speech before the true end; never overlap the
            # previous turn.
            turn_start_ms = max(prev_end_ms + 80.0, turn.true_end_ms - speech_dur)
            # EndOfTurn can be before the true end (a hard cutoff) but never
            # before the speaker started.
            end_of_turn_ms = max(end_of_turn_ms, turn_start_ms + 200.0)

            eager_jitter_rng = Random(
                _stable_seed(self.seed, clip.clip_id, turn.turn_index, "eager")
            )
            eager_jitter = eager_jitter_rng.uniform(-_EAGER_JITTER_MS, _EAGER_JITTER_MS)

            eager_eot_ms = None
            turn_resumed_ms = None
            if point.eager_eot_threshold is not None:
                gap = max(0.0, point.eot_threshold - point.eager_eot_threshold)
                lead = _EAGER_BASE_LEAD_MS + gap * _EAGER_GAP_GAIN_MS + eager_jitter
                eager_eot_ms = max(turn_start_ms + 120.0, end_of_turn_ms - lead)
                if eager_eot_ms < turn.true_end_ms - _PREMATURE_MARGIN_MS:
                    # Premature speculative end: the speaker kept going, so Flux
                    # retracts via TurnResumed. The committed EndOfTurn still
                    # lands at the real end. -> near_miss + wasted eager.
                    turn_resumed_ms = min(
                        end_of_turn_ms - 50.0,
                        max(eager_eot_ms + 60.0, turn.true_end_ms - 30.0),
                    )

            yield from self._emit_turn(
                turn=turn,
                turn_start_ms=turn_start_ms,
                end_of_turn_ms=end_of_turn_ms,
                eager_eot_ms=eager_eot_ms,
                turn_resumed_ms=turn_resumed_ms,
                point=point,
                words=words,
            )

            prev_end_ms = end_of_turn_ms

    # -- internal: build + time-order the EventStamps for one turn -----------
    def _emit_turn(
        self,
        turn: TurnSpec,
        turn_start_ms: float,
        end_of_turn_ms: float,
        eager_eot_ms,
        turn_resumed_ms,
        point,
        words,
    ) -> Iterator[EventStamp]:
        idx = turn.turn_index
        full = turn.transcript or ""
        eot_conf = point.eot_threshold
        eager_conf = (
            min(point.eot_threshold - 0.01, point.eager_eot_threshold + 0.05)
            if point.eager_eot_threshold is not None
            else None
        )

        stamps: list[EventStamp] = []

        # StartOfTurn
        stamps.append(
            EventStamp(
                event="StartOfTurn",
                t_rel_ms=turn_start_ms,
                transcript="",
                end_of_turn_confidence=None,
                turn_index=idx,
            )
        )

        # A few Updates, partial transcripts with rising (sub-threshold) eot conf.
        n_updates = max(1, min(3, len(words)))
        span = max(1.0, turn.true_end_ms - turn_start_ms)
        for u in range(1, n_updates + 1):
            frac = u / (n_updates + 1)
            k = max(1, round(len(words) * frac)) if words else 0
            partial = " ".join(words[:k]) if words else ""
            rising = 0.1 + (point.eot_threshold - 0.2) * frac
            stamps.append(
                EventStamp(
                    event="Update",
                    t_rel_ms=turn_start_ms + span * frac,
                    transcript=partial,
                    end_of_turn_confidence=round(max(0.0, rising), 3),
                    turn_index=idx,
                )
            )

        # Optional EagerEndOfTurn (+ optional TurnResumed when premature).
        if eager_eot_ms is not None:
            stamps.append(
                EventStamp(
                    event="EagerEndOfTurn",
                    t_rel_ms=eager_eot_ms,
                    transcript=full,
                    end_of_turn_confidence=round(eager_conf, 3) if eager_conf else None,
                    turn_index=idx,
                )
            )
            if turn_resumed_ms is not None:
                stamps.append(
                    EventStamp(
                        event="TurnResumed",
                        t_rel_ms=turn_resumed_ms,
                        transcript=full,
                        end_of_turn_confidence=None,
                        turn_index=idx,
                    )
                )
                # one more partial after the speaker resumes, before the real end
                stamps.append(
                    EventStamp(
                        event="Update",
                        t_rel_ms=min(end_of_turn_ms - 30.0, turn_resumed_ms + 80.0),
                        transcript=full,
                        end_of_turn_confidence=round(point.eot_threshold - 0.15, 3),
                        turn_index=idx,
                    )
                )

        # Committed EndOfTurn.
        stamps.append(
            EventStamp(
                event="EndOfTurn",
                t_rel_ms=end_of_turn_ms,
                transcript=full,
                end_of_turn_confidence=round(eot_conf, 3),
                turn_index=idx,
            )
        )

        stamps.sort(key=lambda s: s.t_rel_ms)
        yield from stamps


def turn_class(clip: ClipSpec, turn: TurnSpec) -> str:
    """Resolve the audio class that biases this turn's detection difficulty.

    The class lives on the clip; this indirection keeps the bias lookup in one
    place in case a future spec moves it per-turn.
    """
    return clip.audio_class


def build_mock_sources(seed: int = 0) -> Sources:
    """Assemble the deterministic mock FluxSource into a Sources for the agent.

    No keys, no audio, no software development kits. ``bench.py --mock`` calls
    this; swapping in ``build_real_sources(...)`` is the only change for a real
    run.
    """
    return Sources(flux=MockFluxSource(seed=seed))
