# Advanced learning and experiment guide

## 1. Reconstruct the observation

Read `exofinder/data.py` and inspect a FITS header. Identify BKJD, PDCSAP_FLUX, SAP_QUALITY, quarter and data release. Compare a raw curve with its binned model input. Explain why a single quarter restricts long-period coverage.

**Exercise:** count exclusions by reason and compare usable candidate/false-positive proportions to the pre-download selection. Is missingness independent of label? Record the answer rather than assuming it.

## 2. Audit preprocessing

Read `preprocess.py`. The trend uses a masked primary transit and treats time gaps over half a day separately. Noise normalization is object-local. Occupancy masks distinguish missing measurements from zero relative flux.

**Exercise:** inject transits of known depth/duration into quiet real curves and measure depth attenuation. Vary the detrending window. Avoid judging preprocessing by downstream test AP.

**Limitation:** secondary eclipses and other planets are not masked. The local representation uses catalog duration. This is prior information, not blind signal discovery.

## 3. Trace the network

Each branch: Conv(2→16, k7) → ReLU → Pool → Conv(16→32, k5) → ReLU → Pool → Conv(32→32, k5) → ReLU → adaptive pool to 8 positions. Concatenate 256+256 values, then Dense(512→64→1) with dropout.

**Exercise:** compute the receptive field before adaptive pooling. Explain what information the global branch can capture that the close-up cannot. Run a mask-only ablation to detect observation-coverage shortcuts.

## 4. Defend evaluation

Read `train.py` and the saved `splits.json`. Grouped folds prevent related events from sharing stars across partitions. Validation selects the best AP checkpoint and F1 threshold. The baseline scaler sees training rows only.

**Exercise:** explain why average precision depends on class prevalence and why this deliberately balanced subset cannot establish mission-wide precision. Compare the no-skill AP reference (test prevalence) with the measured AP.

The bootstrap CI measures resampling uncertainty on this test sample. It does not include seed, dataset selection or domain-shift uncertainty. Repeated test inspection converts a test set into development data; reserve a new holdout for later iterations.

## 5. Study errors

Filter `runs/pilot/predictions.csv` to test rows and sort false positives by score. Use the explorer to compare global and local views. Consult eclipse/centroid/ephemeris flags only for analysis, never as hidden model inputs.

**Exercise:** write a failure taxonomy: low SNR, sparse transit coverage, secondary eclipse, V-shaped eclipse, incorrect ephemeris, instrumental trend. Manually reviewed classifications remain hypotheses.

## 6. Ship honestly

Use the API docs at `/docs`. Upload a real CSV exported from the same mission using known ephemerides. Validate expected behavior under malformed input, missing data and periods outside the training scope.

The web app is a scientific interface, not a discovery certificate. Its candidate score is uncalibrated. The displayed DR25 label and model prediction are intentionally separate.

## Primary references

- [Google AstroNet source and walkthrough](https://github.com/google-research/exoplanet-ml)
- [NASA KOI column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html)
- [NASA KOI table purpose and delivery history](https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html)
- [MAST Kepler archive](https://archive.stsci.edu/kepler/)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)

NASA Exoplanet Archive acknowledgment DOI: 10.26133/NEA4. Kepler data are NASA observations hosted by MAST/STScI; this independent project is not affiliated with NASA or Google.
