# Portfolio publication checks

- 11 scientific/API tests and five frontend tests passed locally (four parser tests, one full DOM/API workflow).
- Current explorer and evaluation screenshots were captured from the running application and are used in the README.
- The evaluation dashboard was inspected at desktop width and at 390 px mobile width. No horizontal document overflow was observed; wide data tables scroll inside their containers.
- The original pilot checkpoint, splits, dataset and measured metrics are retained.
- Git excludes raw FITS downloads, node_modules, Python environments, build output, archive bundles and environment files.
- The GitHub Actions workflow runs Python and frontend integration checks from the repository contents on Ubuntu.

The historical `VERIFICATION.md` and `RELEASE_0.2.md` records describe checks available at those earlier milestones. The screenshots `explorer.jpg` and `evaluation.jpg` supersede the old `preview.jpg` for the current interface.
