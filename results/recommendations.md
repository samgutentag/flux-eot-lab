# Flux end-of-turn detection recommendations

> Generated from a real harness run against real Deepgram sources.

Detection latency = true end of speech to the first end-of-turn event. Tolerance window &plusmn;200 ms. Operating point ~600 ms. Budget band: green &le; 2% / yellow &le; 5% / red &gt; 5% false-cutoff. 597 raw turns, 1 errored, 105 excluded as suspect (mis-segmented).

## Per-class threshold recommendations

| Audio class | Recommended `eot_threshold` | Eager (`eager_eot_threshold`) | Detection p50 | Detection p95 | False-cutoff rate | Band |
|---|---|---|---|---|---|---|
| Clean / short turns | `0.6` | `0.40` | -81 ms | 48 ms | 0.0% | green |
| Clean / long-form turns | `0.6` | `0.40` | 41 ms | 614 ms | 0.0% | green |
| Noisy / single speaker | `0.9` | `0.70` | 349 ms | 5054 ms | 0.0% | green |
| Crosstalk (secondary voice) | `0.8` | `0.60` | 220 ms | 671 ms | 0.0% | green |

## The eager-EOT head-start and cost

At the best eager-on point per class: the head-start (how much earlier `EagerEndOfTurn` fires than the committed `EndOfTurn`, the lead time eager hands downstream work) and the wasted-speculation rate (eager fires walked back by `TurnResumed`).

| Audio class | Head-start (p50) | Wasted-speculation rate, eager on |
|---|---|---|
| Clean / short turns | 0 ms | 7.7% |
| Clean / long-form turns | 0 ms | 0.0% |
| Noisy / single speaker | 387 ms | 7.7% |
| Crosstalk (secondary voice) | 209 ms | 6.7% |

## Full curve data

Every swept point. The hero plot is detection p50 against false-cutoff rate, one curve per class.

| Class | `eot` | Eager | n | Detection p50 | Detection p95 | Head-start p50 | False-cutoff | Wasted-spec |
|---|---|---|---|---|---|---|---|---|
| Clean / short turns | `0.5` | `off` | 11 | -55 ms | 164 ms | n/a | 9.1% | 0.0% |
| Clean / short turns | `0.5` | `0.30` | 10 | -101 ms | 49 ms | 0 ms | 10.0% | 0.0% |
| Clean / short turns | `0.6` | `off` | 14 | -7 ms | 227 ms | n/a | 7.1% | 0.0% |
| Clean / short turns | `0.6` | `0.40` | 13 | -81 ms | 48 ms | 0 ms | 0.0% | 7.7% |
| Clean / short turns | `0.7` | `off` | 15 | 115 ms | 704 ms | n/a | 0.0% | 0.0% |
| Clean / short turns | `0.7` | `0.50` | 13 | -28 ms | 159 ms | 0 ms | 0.0% | 7.7% |
| Clean / short turns | `0.8` | `off` | 14 | 105 ms | 1062 ms | n/a | 7.1% | 0.0% |
| Clean / short turns | `0.8` | `0.60` | 15 | -16 ms | 230 ms | 89 ms | 0.0% | 6.7% |
| Clean / short turns | `0.9` | `off` | 15 | 510 ms | 1378 ms | n/a | 0.0% | 0.0% |
| Clean / short turns | `0.9` | `0.70` | 14 | 90 ms | 812 ms | 297 ms | 0.0% | 7.1% |
| Clean / long-form turns | `0.5` | `off` | 12 | 75 ms | 964 ms | n/a | 8.3% | 0.0% |
| Clean / long-form turns | `0.5` | `0.30` | 11 | 57 ms | 549 ms | 0 ms | 0.0% | 0.0% |
| Clean / long-form turns | `0.6` | `off` | 12 | 65 ms | 1100 ms | n/a | 0.0% | 0.0% |
| Clean / long-form turns | `0.6` | `0.40` | 12 | 41 ms | 614 ms | 0 ms | 0.0% | 0.0% |
| Clean / long-form turns | `0.7` | `off` | 12 | 154 ms | 1271 ms | n/a | 0.0% | 0.0% |
| Clean / long-form turns | `0.7` | `0.50` | 11 | 98 ms | 1037 ms | 181 ms | 0.0% | 0.0% |
| Clean / long-form turns | `0.8` | `off` | 12 | 655 ms | 1493 ms | n/a | 0.0% | 0.0% |
| Clean / long-form turns | `0.8` | `0.60` | 12 | 75 ms | 1122 ms | 247 ms | 0.0% | 0.0% |
| Clean / long-form turns | `0.9` | `off` | 12 | 874 ms | 1741 ms | n/a | 0.0% | 0.0% |
| Clean / long-form turns | `0.9` | `0.70` | 12 | 104 ms | 1315 ms | 451 ms | 0.0% | 0.0% |
| Noisy / single speaker | `0.5` | `off` | 10 | 323 ms | 5112 ms | n/a | 10.0% | 0.0% |
| Noisy / single speaker | `0.5` | `0.30` | 9 | 330 ms | 5013 ms | 0 ms | 11.1% | 0.0% |
| Noisy / single speaker | `0.6` | `off` | 10 | 367 ms | 5068 ms | n/a | 10.0% | 0.0% |
| Noisy / single speaker | `0.6` | `0.40` | 9 | 274 ms | 5062 ms | 0 ms | 11.1% | 0.0% |
| Noisy / single speaker | `0.7` | `off` | 10 | 435 ms | 5121 ms | n/a | 10.0% | 0.0% |
| Noisy / single speaker | `0.7` | `0.50` | 9 | 456 ms | 5068 ms | 0 ms | 11.1% | 0.0% |
| Noisy / single speaker | `0.8` | `off` | 9 | 361 ms | 5216 ms | n/a | 11.1% | 0.0% |
| Noisy / single speaker | `0.8` | `0.60` | 9 | 356 ms | 5074 ms | 1 ms | 11.1% | 0.0% |
| Noisy / single speaker | `0.9` | `off` | 14 | 587 ms | 5256 ms | n/a | 0.0% | 0.0% |
| Noisy / single speaker | `0.9` | `0.70` | 13 | 349 ms | 5054 ms | 387 ms | 0.0% | 7.7% |
| Crosstalk (secondary voice) | `0.5` | `off` | 13 | 12 ms | 517 ms | n/a | 7.7% | 0.0% |
| Crosstalk (secondary voice) | `0.5` | `0.30` | 10 | -146 ms | 63 ms | 204 ms | 10.0% | 20.0% |
| Crosstalk (secondary voice) | `0.6` | `off` | 13 | 138 ms | 568 ms | n/a | 7.7% | 0.0% |
| Crosstalk (secondary voice) | `0.6` | `0.40` | 13 | -15 ms | 402 ms | 199 ms | 7.7% | 7.7% |
| Crosstalk (secondary voice) | `0.7` | `off` | 15 | 412 ms | 1135 ms | n/a | 0.0% | 0.0% |
| Crosstalk (secondary voice) | `0.7` | `0.50` | 13 | -4 ms | 436 ms | 191 ms | 7.7% | 0.0% |
| Crosstalk (secondary voice) | `0.8` | `off` | 15 | 521 ms | 1382 ms | n/a | 0.0% | 0.0% |
| Crosstalk (secondary voice) | `0.8` | `0.60` | 15 | 220 ms | 671 ms | 209 ms | 0.0% | 6.7% |
| Crosstalk (secondary voice) | `0.9` | `off` | 15 | 1015 ms | 2012 ms | n/a | 0.0% | 0.0% |
| Crosstalk (secondary voice) | `0.9` | `0.70` | 15 | 365 ms | 1134 ms | 647 ms | 0.0% | 6.7% |
