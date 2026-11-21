# Data provenance and attribution

This repository bundles a small, processed research dataset and a trained checkpoint so the demo can run without downloading raw observations.

| Artifact | Origin / purpose |
|---|---|
| `data/catalog.csv` | NASA Exoplanet Archive `q1_q17_dr25_koi` TAP response |
| `data/selection.csv` | Deterministic 400-object selection from the snapshot |
| `data/objects.csv` | 356 usable objects, metadata and source FITS URLs/hashes |
| `data/views.npz` | Global/local flux bins and occupancy channels produced by this project |
| `data/manifest.json` | Retrieval time, query, hashes, label mapping and exclusions |
| `examples/kepler-479-b.csv` | Quality-filtered Q9 PDCSAP photometry, exported with float64 precision |
| `runs/pilot/` | Project-generated checkpoint, model scores, metrics and split IDs |

Raw FITS observations are retrieved from MAST's public Kepler archive and cached in `data/raw/`, which is excluded from Git. Use the acquisition command in the README to retrieve them. The query and individual URLs are recorded, so no API key is required.

## Acknowledgments

NASA Exoplanet Archive DOI: **10.26133/NEA4**. Kepler observations were collected by NASA and hosted by MAST/STScI. Consult the archive's acknowledgment and data-use guidance when reusing the data in research:

- [NASA Exoplanet Archive acknowledgments](https://exoplanetarchive.ipac.caltech.edu/docs/acknowledge.html)
- [MAST data-use acknowledgment](https://archive.stsci.edu/publishing/data-use)
- [Kepler archive](https://archive.stsci.edu/kepler/)

The MIT license covers this project's original code and documentation; it does not replace terms or attribution attached to source observations or third-party dependencies.
