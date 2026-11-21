# Multi-quarter research study

## Research questions

1. Can the dual-view CNN generalize to a larger, naturally sampled cohort of Kepler KOI stars observed in two quarters?
2. How sensitive are results to initialization, and what information is carried by global flux, local flux and missing-data masks?
3. Does an independently fitted sigmoid improve held-out score reliability and proper scoring rules?
4. Can a blind BLS search recover known injected box transits in real observation backgrounds?

## Dataset

The acquisition command requests 4,000 stars from the bundled frozen DR25 catalog. It excludes every star used in the pilot, requires finite positive ephemerides and duration less than one quarter of the period, and limits periods to 0.5–30 days. One randomly chosen KOI per star is retained. The deterministic selection seed is 2026. No class balancing, eclipse-flag selection or S/N ranking is applied.

Both Q9 and Q10 long-cadence PDCSAP FITS files are required. SAP_QUALITY must equal zero; nonfinite or nonpositive flux is removed. Each quarter is normalized by its own median, then concatenated. The existing transit-masked, segment-wise detrending and 512/128 global/local phase representation are applied. Each source file's URL, SHA-256, quarter identity and cadence count are recorded.

`data/research-q9-q10/manifest.json` contains selection counts, usable counts, excluded objects and hashes. Download failures and phase-coverage failures can alter class prevalence and the accessible population.

**Population:** the eligible KOI-star cohort with both quarters available. This is not a representative sample of all Kepler stars, all periods or confirmed planets. “Candidate” remains a catalog disposition.

## Four independent partitions

Each stellar ID is deterministically assigned using SHA-256 of `kepler-research-2026-v1:KIC`. The hash interval allocates approximately 55% training, 15% validation, 15% calibration and 15% test. Hash allocation preserves natural cohort prevalence in expectation, without forcing class ratios. All modes and initialization seeds use the identical partition. Both classes are required in every partition.

- **Training:** weight updates and class weights.
- **Validation:** checkpoint selection and operating threshold.
- **Calibration:** monotonic sigmoid fit only; no neural checkpoint selection.
- **Test:** final comparisons and calibration assessment.

The saved `plan.json` fixes dataset and metadata hashes, split rule, seeds, modes, epoch budget, code hashes and ensemble policy before training. Completed fits are resumed only when this plan agrees. The app continues to serve the original pilot in its explorer; the new study appears separately in Research Study.

## Twelve predeclared neural fits

Seeds: **42, 43, 44**. Modes:

| Mode | Global input | Local input |
|---|---|---|
| Dual | Flux + occupancy | Flux + occupancy |
| Global | Flux + occupancy | All zeros |
| Local | All zeros | Flux + occupancy |
| Mask | Occupancy only; flux zeroed | Occupancy only; flux zeroed |

All modes use the same 48,865-parameter network. Zeroed branches can learn constants; they receive no per-object information. This is a capacity-matched input ablation, not a smaller architecture comparison. The mask experiment tests observation-coverage shortcuts.

AdamW uses learning rate 0.001, weight decay 0.001, batch size 64, class-weighted binary cross entropy, gradient clipping at 5, phase reversal augmentation, dropout 0.25, at most 30 epochs and six-epoch validation-AP patience. The checkpoint with best validation AP is retained. Threshold F1 is selected on validation only.

A seven-feature logistic baseline is also fitted and scaled on the same training partition, with a validation-selected threshold and separately fitted calibration.

The study reports every seed, then mean and sample standard deviation across seeds. Standard deviation across three seeds is not a confidence interval for population performance. Test metrics are not used to choose a mode, seed or epoch budget.

## Independent calibration

For each completed model, a monotonic Platt-style sigmoid is fitted on **calibration stars only**. Inputs are clipped score logits; slope is constrained positive so ranking is preserved. The three dual-view scores are also averaged under a predeclared ensemble rule, then that mean is calibrated independently on the same reserved partition. No best seed is selected.

The test report contains raw/calibrated Brier score, log loss, AP and reliability bins. Calibration may improve or worsen held-out metrics; both are retained. Brier score reflects discrimination as well as reliability, so it is interpreted alongside the bin plot. The calibrator estimates the probability of a catalog label for this selected cohort, not a physical probability that a planet exists.

## Blind BLS and injection/recovery

`exofinder.bls` searches periods without a supplied ephemeris using Astropy BoxLeastSquares. Before searching, a two-day segment-wise median trend is removed without masking the injected transit. Duration trials are 0.04, 0.08, 0.12 and 0.20 days. The fixed search spans 0.5–15 days, frequency_factor=5 and oversample=5. No injected period is inserted into the search grid.

The experiment selects the first eight KICs in the fresh test partition without a quietness filter. It removes windows around all known short-period KOIs for each background star from the bundled catalog; unknown signals and variability remain. Each real Q9/Q10 background receives box injections at periods **2, 5, 10 days**, depths **200, 500, 1500 ppm**, and duration **0.12 days**, with seeded random epochs: **72 trials**. Each background also gets one uninjected control search.

Strict recovery requires a period within 1%, epoch within one injected duration, and depth SNR ≥7.1. Half/double-period harmonics are reported separately. SNR 7.1 is an exploratory operating criterion here, not a calibrated global false-alarm probability. A large peak on an uninjected background illustrates this distinction.

The trials table records every injected/recovered ephemeris and match decision. Recovery is detection-only and is not multiplied by CNN performance or described as mission-wide completeness. Box transits omit limb darkening and physical parameter distributions; eight selected KOI backgrounds provide only a small initial experiment.

## Commands

```sh
.venv/bin/python -m exofinder.multiquarter --limit 4000 --workers 12
.venv/bin/python -m exofinder.research --epochs 30 --device auto
.venv/bin/python -m exofinder.bls --experiment --stars 8

# Blind search on a new CSV:
.venv/bin/python -m exofinder.bls --csv examples/kepler-479-b.csv --output bls-result.json
```

Raw observations are cached under `data/raw/` and excluded from Git. Completed datasets and studies are protected from accidental overwrite. The original pilot artifacts remain reproducible. Use a new study version and untouched test stars for subsequent tuning informed by this study's test results.

## Primary references

- [Astropy BoxLeastSquares guide](https://docs.astropy.org/en/stable/timeseries/bls.html)
- [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
- [NASA KOI column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html)

## Reproduce from a fresh clone

The repository includes completed experiment records and processed views. Commands protect those records from overwrite. To run a new hardware reproduction, archive the reference run folders first, then execute the same fixed study:

```sh
mv runs/research-q9-q10 runs/research-q9-q10-reference
.venv/bin/python -m exofinder.research --epochs 30 --device auto

# Restore only the 16 FITS files needed for the eight experiment backgrounds.
.venv/bin/python -m exofinder.cache --backgrounds 8
mv runs/injection-recovery runs/injection-recovery-reference
.venv/bin/python -m exofinder.bls --experiment --stars 8
```

`cache` verifies the downloaded FITS SHA-256 against the published metadata. Use `--backgrounds 0` to restore all usable raw sources. To rebuild the processed dataset too, archive `data/research-q9-q10` under a reference name before running the acquisition command; its original selection and manifest remain available for comparison. This is a reproducibility check, not a new independent scientific test.
