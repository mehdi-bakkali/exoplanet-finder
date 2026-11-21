# Exoplanet Finder — implementation plan

## Scientific question

Can a compact dual-view 1D CNN distinguish Kepler DR25 planet-candidate signals from catalog false positives using Q9 photometry alone, with catalog ephemerides supplied? This is a **signal-vetting task**, not an unsupervised discovery pipeline or physical planet confirmation.

## Delivered architecture

1. NASA Exoplanet Archive TAP → versioned local DR25 catalog snapshot.
2. Deterministic, class-balanced selection, one KOI per KIC → MAST public S3 FITS files.
3. Clean PDCSAP cadences → transit-masked median trend → phase fold → robust noise scaling → global/local median bins and occupancy channels.
4. Star-grouped 60/20/20 approximate partitions → two convolutional encoders → binary classifier.
5. Validation AP selects checkpoint; validation F1 selects threshold. A logistic baseline sees seven morphology features derived from the same views.
6. One final held-out test → metrics, uncertainty, every prediction, exact split IDs, checkpoint and dataset fingerprints.
7. FastAPI → responsive browser interface → catalog explorer, phase charts, evaluation, exports and CSV inference. Everything runs locally.

## Design decisions

- Python 3.12 for broad scientific package compatibility. The dependency snapshot is in `requirements-lock.txt`.
- PyTorch MPS for training on Apple silicon; CPU for small inference workloads.
- A small network and 512/128 bins keep the pilot inexpensive. This is inspired by AstroNet's two-view idea, not a reproduction of its architecture, full dataset or published metrics.
- No target IDs, catalog dispositions, false-positive flags or disposition scores are model features. Catalog ephemerides define the fold and window; that assumption is disclosed.
- Missing bins are neutral with an explicit mask. Charts omit them.
- No invented benchmark numbers, simulated observations in the explorer, or claims of planet confirmation.

## Acceptance checks

- Real FITS inputs and catalog provenance are available and hashed.
- Every split has both classes and no overlapping KIC IDs.
- Transit remains visible after detrending; arbitrary input ordering is handled.
- Both CNN branches receive gradients.
- Baseline fit and scaling use training data only.
- Threshold and checkpoint selection exclude test data.
- API rejects malformed curves and returns actual checkpoint predictions.
- Explorer and evaluation read the run artifacts, not hard-coded figures.
- Desktop/mobile navigation, filtering, phase switching, keyboard focus and upload flow work.

## Next research milestones

1. **Dataset scale:** at least several thousand KOIs, multiple quarters, a predeclared selection function and a failure audit by label. Preserve the pilot as a separate experiment.
2. **Robustness:** three or more predeclared seeds; global-only/local-only/mask-only ablations; stratify errors by period, depth/SNR and stellar-eclipse flags. Do not tune repeatedly against this pilot test set.
3. **Calibration:** fit a calibrator on a separate representative calibration set, assess reliability and Brier score, and choose a recall-oriented operating point appropriate for follow-up capacity.
4. **Blind detection:** introduce BLS period search and compare recovery of injected transits. Report detection completeness separately from vetting quality.
5. **Scientific validation:** pixel-centroid evidence, odd/even transit comparisons, secondary eclipses, contamination checks, external catalogs and expert review. A CNN score alone cannot validate a planet.

## Portfolio narrative

Lead with the measured question, show the raw data provenance and leakage controls, explain one success and one failure, and discuss whether the CNN actually beats the baseline. Demonstrating sound evaluation is more valuable than a large unsupported accuracy number.
