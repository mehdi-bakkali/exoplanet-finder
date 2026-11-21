# Contributing

Start with the README, model card and implementation plan. For a local change, create a branch, keep the work focused, and run the Python and relevant frontend tests before opening a pull request.

Please include the problem, resulting behavior, validation, and any scientific assumptions in the pull request. Screenshots help when changing the interface.

## Scientific changes

- Preserve the pilot artifacts so changes can be compared with a fixed reference.
- Keep stars grouped across splits. Fit preprocessing statistics and model parameters on training data only.
- Select models and operating thresholds using validation data. If pilot errors guide a new model, evaluate the final design on a new untouched holdout.
- Record source data, exclusions, seeds, package versions and dataset hashes.
- Report baselines, uncertainty and failure cases alongside headline metrics.

The project labels known catalog signals; a model score alone does not confirm a planet.

## Development environment

Use Python 3.12 and the setup commands in the README. Node 24 and pnpm 11.25.0 run the optional DOM/CSV tests. The application itself requires no frontend build.
