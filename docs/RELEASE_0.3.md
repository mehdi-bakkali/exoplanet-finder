# Version 0.3 — multi-quarter experiments

## Added

- A separate Q9/Q10 dataset acquisition command requesting 4,000 stars, excluding every pilot star and requiring both quarters.
- A fixed train/validation/calibration/test partition shared by three seeds and four capacity-matched input ablations.
- Twelve neural fits with raw and independently calibrated metrics, plus a predeclared three-seed dual-view ensemble.
- A BLS period-search CLI and API, an interactive CSV period-search control, and real-background box injection/recovery trials with uninjected controls.
- A Research Study page with ablation, calibration and recovery tables, report export, and a static publication figure.
- Tests for split invariants, ablation behavior, calibration, test-label isolation, FITS cache validation and blind detection recovery.

## Inspect the science

Read [study design](RESEARCH_STUDY.md) and [measured results](RESEARCH_RESULTS.md). Individual predictions, checkpoints, fit plans and injection trials are retained under `runs/`.

The pilot explorer remains tied to the original Q9 checkpoint and scores. The research study uses a distinct dataset and partitions. Calibration estimates a catalog-label frequency for an eligible KOI cohort; it does not confirm planets. BLS recovery and CNN vetting are evaluated separately.

## Reproduce the figure

```sh
uv pip install --python .venv/bin/python -e '.[research]'
.venv/bin/python -m exofinder.study_report
```

Research plotting uses Matplotlib; it is optional for serving the app or running the standard tests. The figure generator requires completed training and injection/recovery artifacts.
