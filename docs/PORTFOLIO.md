# Portfolio walkthrough

## The problem

A shallow periodic dip in a stellar light curve can come from a planet, an eclipsing stellar companion, contamination, or instrumental effects. A useful classifier should rank signals for inspection and expose enough evidence to challenge its prediction.

## What this project delivers

An end-to-end research prototype: real public data acquisition, transit-preserving preprocessing, dual-view convolutional classification, a simple baseline, leakage-aware evaluation, and a local browser interface for inspecting results.

The app includes the processed pilot data and checkpoint. Reviewers can try inference with the bundled real example without retraining or accessing an external account.

## Five-minute walkthrough

1. Start the app with the README instructions and search for **K00166**. Inspect the transit close-up, then switch to the full orbit.
2. Open **Evaluation**. Explain AP, test prevalence and the CNN/baseline trade-off. The CNN has higher precision at its selected threshold; the baseline has higher recall.
3. Inspect the reliability bins. Explain why an uncalibrated score should not be described as a probability of a planet existing.
4. Move the validation threshold slider. Show that saved test metrics stay fixed.
5. Open a false alarm or missed candidate in the error table. Explain the scientific uncertainty instead of rationalizing the model's answer.
6. Load the real Kepler example in the inference dialog and export the result JSON.

## Engineering decisions to discuss

- Grouping by stellar ID prevents related events from leaking between train and test.
- Local and global views represent transit morphology and orbital context separately.
- Explicit occupancy channels distinguish missing observations from neutral flux.
- The baseline tests whether network complexity adds measurable value.
- Recorded hashes, exact splits and per-object predictions make the experiment inspectable.
- The build-free frontend and local API reduce setup and inference overhead.

## Measured claim

> Built a PyTorch dual-view 1D CNN to vet NASA Kepler transit candidates, achieving 0.875 average precision on 72 held-out stars in a 356-star pilot, with grouped splits, baseline comparison, data provenance, and an interactive analysis app.

Scope matters: this is a single-quarter pilot using supplied ephemerides and catalog labels. A stronger next experiment needs more stars, multiple quarters, ablations, several seeds, and a fresh holdout.
