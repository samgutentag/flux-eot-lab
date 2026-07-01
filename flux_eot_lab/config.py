"""Locked constants and the sweep grid for the Flux EOT lab.

All values here are frozen so every other module can import them. The Flux
param ranges/defaults are the VERIFIED values from Deepgram's Flux documentation
(developers.deepgram.com); do not edit them to match a guess.

Verified Flux turn params (developers.deepgram.com):
  - eot_threshold       0.5-0.9, default 0.7. Confidence required to COMMIT an
                        EndOfTurn. Higher = wait longer / closer to true end of
                        speech (fewer hard cutoffs, more latency).
  - eager_eot_threshold 0.3-0.9, must be <= eot_threshold. UNSET disables eager
                        EOT entirely. When set, Flux speculatively emits an
                        EagerEndOfTurn earlier so downstream work can start
                        before the turn is fully committed; a TurnResumed walks
                        it back if the speaker continues.
  - eot_timeout_ms      500-10000, default 5000. Silence fallback that forces an
                        EndOfTurn.

These same three params pass through the managed Voice Agent API under
agent.listen.provider (version v2, model flux-general-en).
"""

from __future__ import annotations

from flux_eot_lab.records import SweepPoint


def load_env(path) -> None:
    """Load KEY=VALUE lines from a .env into os.environ without overriding vars
    already set in the shell.

    Prefers python-dotenv, but falls back to a tiny manual parse so `.env` works
    with zero dependencies (for example on system python outside the venv). If
    the file exists but can't be read, it warns on stderr instead of failing
    silently, which is the trap that hides a missing key.
    """
    import os
    import sys
    from pathlib import Path

    p = Path(path)
    try:
        from dotenv import load_dotenv

        load_dotenv(p)
        return
    except ImportError:
        pass  # no python-dotenv; fall back to the manual parse below

    if not p.exists():
        return
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except OSError as exc:
        print(f"[warn] could not read {p}: {exc}", file=sys.stderr)

# --- Audio classes -----------------------------------------------------------
# Speaker axis is crosstalk robustness, NOT diarization.
AUDIO_CLASSES: list[str] = [
    "clean_short",    # (1) clean, short turns
    "clean_long",     # (2) clean, long-form single-utterance turns
    "noisy_single",   # (3) noisy, single speaker
    "crosstalk",      # (4) secondary voice / crosstalk
]

# --- Model ids ---------------------------------------------------------------
DEFAULT_FLUX_MODEL = "flux-general-en"
# Aura is used only to synthesize the test clips (audio/generate_audio.py) and to
# measure round-trip time for --ping; it is NOT in the measured detection path.
DEFAULT_AURA_MODEL = "aura-2-andromeda-en"

# --- Audio formats -----------------------------------------------------------
INPUT_SAMPLE_RATE = 16000   # Flux input: linear16, 16 kHz
INPUT_ENCODING = "linear16"
AURA_SAMPLE_RATE = 24000    # Aura TTS output

# --- Verified param ranges / defaults -----------------------------------
EOT_THRESHOLD_RANGE = (0.5, 0.9)
EOT_THRESHOLD_DEFAULT = 0.7

EAGER_EOT_THRESHOLD_RANGE = (0.3, 0.9)   # must be <= eot_threshold; None = eager disabled
EOT_TIMEOUT_MS_RANGE = (500, 10000)
EOT_TIMEOUT_MS_DEFAULT = 5000

# --- Sweep -------------------------------------------------------------------
EOT_SWEEP = [0.5, 0.6, 0.7, 0.8, 0.9]
# Eager is set a couple of steps below each eot value when eager is "on".
EAGER_STEP_BELOW = 0.2
# --refine adds intermediate eot values through the 0.6-0.8 knee.
REFINE_EOT_EXTRA = [0.55, 0.65, 0.75]

# --- Analysis defaults -------------------------------------------------------
DEFAULT_TOLERANCE_MS = 200          # +/- window applied at ANALYSIS time, sweepable
DEFAULT_OPERATING_POINT_MS = 600    # the stated operating point for the budget band

# false-cutoff fractions. green <= 2%, yellow <= 5%, >yellow = red. (R3)
# No published industry standard exists for this band; see README footnote.
BUDGET_BANDS = {"green": 0.02, "yellow": 0.05}

# --- Honest-placeholder token ------------------------------------------------
# Every measured number stays this until a real (non-mock) run fills it in.
PENDING = "PENDING_REAL_RUN"


def _clamp_eager(eot: float, step: float = EAGER_STEP_BELOW) -> float:
    """Eager threshold a step below `eot`, clamped to its verified range and
    guaranteed <= eot."""
    lo, hi = EAGER_EOT_THRESHOLD_RANGE
    eager = round(eot - step, 2)
    eager = max(lo, min(hi, eager))
    eager = min(eager, eot)   # invariant: eager_eot_threshold <= eot_threshold
    return eager


def sweep_grid(refine: bool = False) -> list[SweepPoint]:
    """Build the sweep: one eager-OFF and one eager-ON point per eot value.

    eager-off  -> eager_eot_threshold = None (eager disabled)
    eager-on   -> eager_eot_threshold a couple of steps below the eot value

    The default pass is fast (5 eot values x 2 = 10 points). `refine=True`
    inserts finer eot values near the 0.6-0.8 knee for a tighter curve.
    """
    eots = list(EOT_SWEEP)
    if refine:
        eots = sorted(set(eots) | set(REFINE_EOT_EXTRA))

    grid: list[SweepPoint] = []
    for eot in eots:
        # eager OFF
        grid.append(
            SweepPoint(
                eot_threshold=eot,
                eager_eot_threshold=None,
                eot_timeout_ms=EOT_TIMEOUT_MS_DEFAULT,
            )
        )
        # eager ON (a couple of steps below eot, range-clamped, <= eot)
        grid.append(
            SweepPoint(
                eot_threshold=eot,
                eager_eot_threshold=_clamp_eager(eot),
                eot_timeout_ms=EOT_TIMEOUT_MS_DEFAULT,
            )
        )
    return grid
