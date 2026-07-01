"""Flux End-of-Turn Detection Tuning Lab.

A reproducible harness that measures Deepgram Flux turn-taking latency vs.
false-cutoff across four audio classes, sweeps the real turn knobs
(eot_threshold / eager_eot_threshold / eot_timeout_ms), and emits per-class
threshold recommendations + tradeoff curves.

Package layout (one owner per module):
    config.py    - constants, verified Flux param ranges, sweep grid
    records.py   - the data contract: dataclasses, JSONL IO, client Protocols
    agent.py     - InstrumentedFluxAgent: loop + timing instrumentation
    clients.py   - real Deepgram Flux source builder (detection-only)
    mock.py      - deterministic fake sources for --mock / smoke test
    analysis.py  - classification + tolerance at analysis time, recommend

The whole pipeline runs WITHOUT API keys via --mock. Mock numbers are wiring
fixtures only; every mock TurnRecord sets mock=True and measured result cells
stay PENDING_REAL_RUN until a real run fills them.
"""
