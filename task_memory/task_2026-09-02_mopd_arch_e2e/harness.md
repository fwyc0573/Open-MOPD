# Task Harness and Gates

## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-03 | Captured the executable gates and evidence rules for the final E2E run. |

## Required Gates

* Keep all models and datasets local; fail if a model path is not present.
* Export `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`.
* Require H800 allocation with the verified `codesign` quota group.
* Run the Hydra compose probe before GPU stages; require `unexpected outcomes: 0`.
* Run the Open-MOPD unit suite and record the exact pass count.
* Require SFT checkpoint tracker `6`, merged weights, two RL steps, two corrected OPD
  steps, two corrected MT-OPD steps per mode, and 12 eval rows.
* Record shipped OPD/MT-OPD Hydra failures as release defects and keep corrected runs
  visibly separate.
* Report every non-core test failure with test name, observed value/error, and root cause.

## Evidence Locations

```text
/data/ycfeng/tmp/openmopd-e2e-r5-20260903/logs
/data/ycfeng/tmp/openmopd-e2e-r5-20260903/runs
```

The wrapper continues after a stage failure to collect downstream evidence, while its
summary and exit code retain the failure signal.
