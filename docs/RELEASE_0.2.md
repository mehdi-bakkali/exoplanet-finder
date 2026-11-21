# Version 0.2 — diagnostic research workspace

## New workflow

1. Open **Evaluation** to compare the frozen test precision–recall curves of the CNN and baseline. The no-skill reference is the test candidate prevalence.
2. Inspect **Score reliability**. Bins show mean model score versus observed catalog candidate fraction, with counts and 95% Wilson intervals. Empty bins are omitted. This is a diagnostic; no probability calibration has been fitted.
3. Use the **validation threshold** slider to examine precision/recall and follow-up load. This control reads validation predictions only. Reset restores the original validation-selected threshold. Nothing is saved or applied to test metrics or inference.
4. Review the subgroup table and the 14 test disagreements. Clicking a KOI opens that object in the explorer. Subgroups are descriptive, small, and partially overlapping; the eclipse flag is for analysis, not a model feature.
5. In the explorer, filter false alarms, missed candidates, or all catalog disagreements. Sort by high/low score or distance from the saved threshold. Empty searches hide stale target details.
6. In **Analyze a light curve**, load the included real Kepler example or select a CSV. The result now includes CNN and baseline scores, the processed transit view, bin coverage, discarded observations and a downloadable JSON report.

## Scientific boundaries

The original checkpoint, split manifest, predictions, processed dataset and headline metrics remain unchanged. These new displays inspect that existing experiment. The model is not improved or recalibrated by this release. Observing test errors is analysis; any future model tuning informed by them requires a new untouched holdout.

The reliability plot concerns DR25 catalog labels within this selected population. It does not turn a model score into an astrophysical probability. The subgroup recall is undefined when a group has no catalog candidates; the UI shows a dash rather than a fabricated zero.

## API and offline export

- `GET /api/diagnostics`: test PR/ROC arrays, reliability bins, error rows and subgroup summaries.
- `GET /api/validation-threshold?threshold=0.5`: validation-only operating point, including counts and confusion matrix.
- `GET /api/example`: bundled real curve and its ephemeris.
- `GET /api/export/diagnostics`: downloadable diagnostic JSON.
- `POST /api/predict`: adds baseline score, ephemeris, run identity, bin coverage and period-domain indication.

Run `python -m exofinder.diagnostics --run pilot` from the project directory to regenerate `runs/pilot/diagnostics.json` offline.

## CSV and precision

The parser accepts BOM, CRLF, quoted headers, quoted numeric values and scientific notation. Duplicate required columns, malformed rows, missing/nonfinite values and inputs outside 200–100,000 rows are rejected. Multiline fields are unsupported. Quality filtering still belongs upstream of upload; positive linear flux and BKJD are required.

The example CSV was re-exported from the original FITS using float64 and 17 significant digits. The previous float32-to-text export caused a small score discrepancy; this release matches the saved CNN and baseline predictions within 1e-5 absolute tolerance.

## Validation

- 11 Python tests pass, including threshold isolation, untouched artifact hashes, reliability edge cases and real-example prediction parity.
- 4 Node parser tests pass.
- 1 jsdom integration test passes: real local API, diagnostic charts, filtering/sorting, empty states, error navigation, threshold isolation and bundled CSV inference.
- JavaScript syntax checks pass.
- A new rendered-browser visual check was unavailable in this session. `docs/preview.jpg` remains the **version 0.1** screenshot, not a picture of the new panels. DOM tests do not verify visual layout or screen-reader experience.

The pnpm/jsdom dependency is development-only. The application still needs no Node runtime or frontend build to run; Python serves plain HTML/CSS/JavaScript.

## References

- [scikit-learn precision–recall curve API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
- [scikit-learn probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
