# Verification record — original v0.1

- Six automated tests passed against the trained pilot checkpoint.
- Python acquisition and training completed using actual Kepler Q9 files.
- MPS training completed; 356 usable stars, 72 held-out test objects.
- Real CSV upload for Kepler-479 b returned 0.648, matching the stored score at displayed precision.
- Browser checked at 1440×1000 and 390×844; neither produced horizontal document overflow.
- Verified target search, global/local chart switch, evaluation navigation and upload submission.
- Browser console had no captured errors at the end of the check.
- JavaScript syntax and launcher shell syntax checked successfully.
- Reduced-motion and reduced-transparency CSS are included; a human assistive-technology audit remains future work.
- Dependency warning: Starlette currently deprecates its httpx test-client adapter; this did not affect the passing tests.

Preview: `preview.jpg`. Start locally with `./start.command`; stop with Control-C.

For the current release, see [v0.2 verification](RELEASE_0.2.md#validation). The preview image above predates the new diagnostic panels.
