# Model card — pilot

## Intended use

Local educational and portfolio research into vetting known Kepler transit signals. Positive means the DR25 catalog label CANDIDATE, not astrophysical confirmation. Not for claims of new planet discovery.

## Dataset and provenance

Retrieved: 2026-11-20T22:31:50.569829+00:00. Source: frozen `q1_q17_dr25_koi` table and Q9 DR25 PDCSAP FITS files from MAST’s public S3 archive.

400 targets selected, 356 usable: 188 candidates and 168 false positives. One randomly chosen KOI per KIC; class balance applied before download. Periods 0.5–30 days; duration positive and shorter than a quarter-period. Strict SAP_QUALITY == 0 filtering.

44 exclusions: 35 files unavailable at the Q9 URL (HTTP 404), nine duration/period validity failures. Of 200 selected candidates, 188 remain; of 200 false positives, 168 remain. Exclusion is not label-independent and changes the population. No synthetic data enters training or evaluation.

The catalog snapshot, request query, exact selection, every exclusion and FITS SHA-256 are retained in `data/`. Processed dataset hash:

`aa3f5c0e540d43d7dd786fe4e9b1f4b4bdab8dfc24cdf431cb2235b547f792e0`

## Architecture and experiment

48,865-parameter dual-view 1D CNN. Inputs: 512 global and 128 local phase bins, each with a flux and occupancy channel. PyTorch 2.15.0; Apple M4 MPS. Seed 42; AdamW, learning rate 0.001, weight decay 0.001; batch size 32; weighted binary cross entropy; dropout 0.25; phase reversal augmentation. Early stopping patience 12; best checkpoint at epoch 5. Training plus final inference/baseline fit took approximately 5.2 seconds (excludes acquisition and bootstrap).

Splits: 213 training stars (113 candidates/100 false positives), 71 validation (37/34), 72 test (38/34). No KIC overlap. Validation AP selects epoch; validation F1 selects threshold. CNN threshold: 0.5697445869445801. Baseline threshold: 0.4942812919616699.

## Held-out results

| Metric | CNN | Logistic shape baseline |
|---|---:|---:|
| Average precision | 0.8746 | 0.8036 |
| ROC AUC | 0.8506 | 0.8135 |
| Precision | 0.8158 | 0.6250 |
| Recall | 0.8158 | 0.9211 |
| F1 | 0.8158 | 0.7447 |
| Brier score | 0.2115 | 0.2179 |

CNN AP bootstrap 95% interval: 0.7624–0.9550; 1,000 resamples of independent test objects. No-skill AP reference: test prevalence 0.5278. This interval excludes training-seed and domain-shift uncertainty. The observed AP difference is not a claim of statistically established superiority.

CNN confusion matrix, rows actual [false positive, candidate], columns predicted in the same order: `[[27, 7], [7, 31]]`. The baseline trades higher recall for lower precision at its own validation-selected threshold.

## Error examples

These are catalog disagreements on the held-out split, not newly adjudicated astronomical classifications.

| KOI | Catalog label | CNN score | Eclipse flag |
|---|---|---:|---:|
| K02112.01 | False positive | 0.625 | 0 |
| K02929.01 | False positive | 0.598 | 1 |
| K02284.01 | False positive | 0.593 | 1 |
| K02752.01 | False positive | 0.579 | 0 |
| K02761.01 | False positive | 0.576 | 1 |
| K08006.01 | False positive | 0.573 | 0 |
| K05307.01 | False positive | 0.573 | 0 |
| K00161.01 | Candidate | 0.568 | 0 |

## Limits and next steps

- Pilot subset, not mission-wide performance.
- DR25 candidates are not confirmed planets.
- Ephemeris supplied by catalog: this vets known signals, not a blind transit search.
- One quarter; short periods; balanced sampling changes prevalence.
- Uncalibrated scores are not astrophysical probabilities.
- MPS seeds improve repeatability but do not guarantee bitwise deterministic kernels.
- Candidate and false-positive labels may contain catalog errors.
- Detrending masks the primary event only; secondary events and other planets can affect the trend.
- Strict filtering, random subset selection and unavailable quarters introduce selection effects.
- Mask-channel shortcuts need ablation; score calibration needs a representative population.
- Current CLI preserves completed run directories; it does not enforce a registry-level ban on future test reuse.

## Verification

Six automated tests passed: preservation/phase invariance, input validity/order, grouped leakage prevention, gradient flow in both views, threshold semantics and trained API integration. Real CSV inference for K00166.01 produced 0.648, agreeing with the saved prediction at displayed precision. Browser verification covers target filtering, view switching, evaluation navigation and real CSV submission.

## Defensible résumé bullet

> Built a PyTorch dual-view 1D CNN to vet NASA Kepler transit candidates, achieving 0.875 average precision on 72 held-out stars in a 356-star pilot, with grouped splits, baseline comparison, reproducible data provenance, and an interactive local analysis app.

Keep the pilot scope and candidate-vetting wording when presenting this result.

## Attribution

[NASA Exoplanet Archive column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html), DOI 10.26133/NEA4; [MAST Kepler archive](https://archive.stsci.edu/kepler/); [Google AstroNet reference](https://github.com/google-research/exoplanet-ml). This project implements its own compact architecture and is not a reproduction of the full published AstroNet study.
