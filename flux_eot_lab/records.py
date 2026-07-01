"""The data contract for the Flux end-of-turn lab.

Every other module imports its types from here, so these signatures are frozen.
Pure standard library only (json, dataclasses, typing), with NO software
development kit imports, so this module loads with zero external dependencies and
the mock path needs no application programming interface keys.

Scope: this harness measures Flux turn DETECTION only, from the true end of
speech to the first end-of-turn event. The downstream large-language-model and
text-to-speech pipeline is deliberately out of scope. That pipeline is the
integrator's own choice and its latency varies independently of the
eot_threshold knob being tuned here, so measuring it would only add variance the
threshold cannot move.

Flux event vocabulary (the only valid values for EventStamp.event and the
strings the agent and mock emit):

    StartOfTurn     - speech detected, a new turn begins
    Update          - interim transcript update within the turn
    EagerEndOfTurn  - speculative early end-of-turn (only when eager is enabled);
                      lets downstream work start before the turn is committed
    TurnResumed     - the speaker kept going, so a prior EagerEndOfTurn is walked
                      back (the speculative start was premature)
    EndOfTurn       - the committed end of the turn

Design rule: TurnRecord is RAW. It carries the full Flux event timeline but bakes
in NO classification and NO tolerance. Those are applied at analysis time
(analysis.py) so the tolerance window and operating point stay sweepable without
re-running the harness.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Iterable, Iterator, Protocol, runtime_checkable

# Canonical event vocabulary. Modules should reference this rather than
# hard-coding strings where practical.
EVENT_TYPES = (
    "StartOfTurn",
    "Update",
    "EagerEndOfTurn",
    "TurnResumed",
    "EndOfTurn",
)


# --- Sweep point -------------------------------------------------------------
@dataclass
class SweepPoint:
    """One point in the threshold sweep.

    eager_eot_threshold is None for the eager-OFF pass (eager disabled). When
    set it must be <= eot_threshold (enforced by config.sweep_grid).
    """

    eot_threshold: float
    eager_eot_threshold: float | None
    eot_timeout_ms: int = 5000

    @property
    def eager_on(self) -> bool:
        return self.eager_eot_threshold is not None

    def label(self) -> str:
        eager = "off" if self.eager_eot_threshold is None else f"{self.eager_eot_threshold:g}"
        return f"eot={self.eot_threshold:g}/eager={eager}"


# --- Ground-truth clip specs (audio/clips/*.json sidecars) -------------------
@dataclass
class TurnSpec:
    """One annotated turn inside a clip.

    true_end_ms is the ground-truth end of the speaker's speech (the headline
    detection metric measures from here). distractor_spans marks secondary-voice
    regions for the crosstalk class: [{"start_ms": ..., "end_ms": ...}].
    """

    turn_index: int
    true_end_ms: float
    transcript: str
    speaker: str = "primary"
    distractor_spans: list[dict] = field(default_factory=list)


@dataclass
class ClipSpec:
    """A single audio clip and its ground-truth annotations.

    audio_file may be None in --mock mode (the mock source synthesizes events
    from the TurnSpecs and never reads audio).
    """

    clip_id: str
    audio_class: str
    audio_file: str | None
    sample_rate: int
    encoding: str
    duration_ms: float
    turns: list[TurnSpec]
    notes: str = ""

    @classmethod
    def from_sidecar(cls, path: str) -> "ClipSpec":
        """Parse an audio/clips/*.json sidecar into a ClipSpec.

        Sidecar schema (one JSON object):
            {
              "clip_id": "clean_short_001",
              "audio_class": "clean_short",
              "audio_file": "clean_short_001.wav",   // null/omit for mock-only
              "sample_rate": 16000,
              "encoding": "linear16",
              "duration_ms": 4200.0,
              "notes": "...",
              "turns": [
                {"turn_index": 0, "true_end_ms": 1850.0,
                 "transcript": "what's the weather", "speaker": "primary",
                 "distractor_spans": [{"start_ms": 900, "end_ms": 1400}]}
              ]
            }
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        af = data.get("audio_file")
        if af and not os.path.isabs(af):
            # audio_file is relative to the audio/ dir (per audio/spec.md);
            # sidecars live in audio/clips/, so the audio/ base is two levels up.
            audio_base = os.path.dirname(os.path.dirname(os.path.abspath(path)))
            data["audio_file"] = os.path.join(audio_base, af)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "ClipSpec":
        turns = [
            TurnSpec(
                turn_index=int(t["turn_index"]),
                true_end_ms=float(t["true_end_ms"]),
                transcript=t.get("transcript", ""),
                speaker=t.get("speaker", "primary"),
                distractor_spans=list(t.get("distractor_spans", []) or []),
            )
            for t in data.get("turns", [])
        ]
        return cls(
            clip_id=data["clip_id"],
            audio_class=data["audio_class"],
            audio_file=data.get("audio_file"),
            sample_rate=int(data.get("sample_rate", 16000)),
            encoding=data.get("encoding", "linear16"),
            duration_ms=float(data.get("duration_ms", 0.0)),
            turns=turns,
            notes=data.get("notes", ""),
        )


# --- Raw event + turn records ------------------------------------------------
@dataclass
class EventStamp:
    """One Flux event, time-stamped relative to clip start.

    event is one of EVENT_TYPES. t_rel_ms is ms since clip start (a single
    monotonic clock zeroed when the clip begins streaming). end_of_turn_confidence
    is the model's reported confidence where applicable, else None.
    """

    event: str
    t_rel_ms: float
    transcript: str
    end_of_turn_confidence: float | None
    turn_index: int


@dataclass
class TurnRecord:
    """RAW per-turn record. No classification, no tolerance baked in.

    All *_ms fields are ms since clip start off one monotonic clock. The signed
    detection_delta_ms (first_eot_event_ms - true_end_ms) is the raw detection
    error; classification and tolerance live in analysis.py.
    """

    clip_id: str
    audio_class: str
    turn_index: int

    # the sweep point this turn was run under
    eot_threshold: float
    eager_eot_threshold: float | None
    eot_timeout_ms: int

    # ground truth
    true_end_ms: float

    # full raw timeline for this turn
    events: list[EventStamp]

    # key event timestamps (ms since clip start), None if the event never fired
    eager_eot_ms: float | None
    end_of_turn_ms: float | None
    turn_resumed_ms: float | None
    first_eot_event_ms: float | None    # earliest of eager/eot: the detection mark

    # raw signed detection error: first_eot_event_ms - true_end_ms
    detection_delta_ms: float | None

    final_transcript: str
    eager_fired: bool
    turn_resumed_fired: bool

    mock: bool = False
    error: str | None = None

    # --- JSON round-trip helpers ---
    def to_dict(self) -> dict:
        d = asdict(self)  # nested EventStamp dataclasses become dicts
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TurnRecord":
        # Tolerate older rows that still carry the now-removed downstream pipeline
        # fields (llm_request_ms / llm_first_token_ms / tts_request_ms /
        # first_audio_byte_ms): keep only current dataclass fields, so a
        # detection-only build can still read a pre-scope-cut JSONL.
        valid = {f.name for f in fields(cls)}
        events = [EventStamp(**e) for e in d.get("events", [])]
        d = {k: v for k, v in d.items() if k in valid}
        d["events"] = events
        return cls(**d)


# --- JSONL IO (one JSON object per line, round-trips cleanly) -----------------
def write_jsonl(path: str, records: Iterable[TurnRecord]) -> int:
    """Write records as JSONL (one object per line). Returns the count written."""
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[TurnRecord]:
    """Read a JSONL file written by write_jsonl back into TurnRecords."""
    out: list[TurnRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(TurnRecord.from_dict(json.loads(line)))
    return out


# --- Injection seam (agent depends ONLY on this Protocol) ---------------------
@runtime_checkable
class FluxSource(Protocol):
    """Yields Flux events for a clip at a sweep point, t_rel_ms already stamped
    relative to clip start. The real implementation streams audio over the Flux
    WebSocket; the mock implementation synthesizes the timeline from ground
    truth. This is the only seam the agent depends on."""

    def stream(self, clip: "ClipSpec", point: "SweepPoint") -> Iterator[EventStamp]: ...
