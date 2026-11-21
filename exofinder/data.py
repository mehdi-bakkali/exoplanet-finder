"""Fetch a deterministic DR25 subset and actual Q9 PDCSAP FITS light curves."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from .preprocess import prepare

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAMP = "2011177032512"  # Kepler Q9 long cadence, DR25
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
COLUMNS = "kepid,kepoi_name,kepler_name,koi_pdisposition,koi_period,koi_time0bk,koi_duration,koi_fpflag_ss,koi_fpflag_nt,koi_fpflag_co,koi_fpflag_ec"
QUERY = f"select {COLUMNS} from q1_q17_dr25_koi where koi_period > 0.5 and koi_period < 30 and koi_duration > 0"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acquire(row):
    row = dict(row)
    kic = f"{int(row['kepid']):09d}"
    url = f"https://stpubdata.s3.amazonaws.com/kepler/public/lightcurves/{kic[:4]}/{kic}/kplr{kic}-{STAMP}_llc.fits"
    file = DATA / "raw" / f"{kic}-q9.fits"
    try:
        if not file.exists():
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            temp = file.with_suffix(".tmp")
            temp.write_bytes(response.content)
            temp.replace(file)
        with fits.open(file, memmap=False) as hdul:
            if hdul[0].header.get("QUARTER") != 9:
                raise ValueError("Expected Q9 data")
            curve = hdul[1].data
            valid = curve['SAP_QUALITY'] == 0
            t, f = np.array(curve['TIME'][valid], float), np.array(curve['PDCSAP_FLUX'][valid], float)
        g, l, info = prepare(t, f, row['koi_period'], row['koi_time0bk'], row['koi_duration'])
        row.update(info, url=url, fits_sha256=sha(file), label=int(row['koi_pdisposition'] == "CANDIDATE"))
        return row, g, l, None
    except Exception as exc:
        return row, None, None, str(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=400, help="Requested objects, balanced before download failures")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.limit < 40:
        parser.error("Use at least 40 objects for grouped train/validation/test splits.")
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    catalog = DATA / "catalog.csv"
    if not catalog.exists():
        response = requests.get(TAP, params={"query": QUERY, "format": "csv"}, timeout=90)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        if 'kepid' not in frame:
            raise ValueError("Unexpected archive response")
        catalog.write_text(response.text)
    df = pd.read_csv(catalog).dropna(subset=['koi_period','koi_time0bk','koi_duration'])
    df = df[df.koi_pdisposition.isin(['CANDIDATE','FALSE POSITIVE'])]
    # One deterministic, randomly chosen KOI per star; no label-dependent ranking.
    df = df.sample(frac=1, random_state=42).drop_duplicates('kepid')
    selected = pd.concat([g.sample(n=min(len(g), args.limit // 2), random_state=42)
                          for _, g in df.groupby('koi_pdisposition')]).sort_values('kepoi_name')
    selected.to_csv(DATA / "selection.csv", index=False)
    records, globals_, locals_, failures = [], [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (row, g, l, error) in enumerate(pool.map(acquire, selected.to_dict('records')), 1):
            if error:
                failures.append({"id": row['kepoi_name'], "reason": error})
            else:
                records.append(row); globals_.append(g); locals_.append(l)
            if i % 20 == 0:
                print(f"Downloaded/processed {i}/{len(selected)}; usable={len(records)}; skipped={len(failures)}", flush=True)
    if not records:
        raise RuntimeError("No usable real data. Check archive availability.")
    metadata = pd.DataFrame(records)
    metadata.to_csv(DATA / "objects.csv", index=False)
    np.savez_compressed(DATA / "views.npz", global_view=np.array(globals_), local_view=np.array(locals_), y=metadata.label.values)
    manifest = {"source": "NASA Kepler DR25 KOI; MAST public Q9 PDCSAP light curves", "query": QUERY,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
                "requested": args.limit, "usable": len(records), "failures": failures,
                "catalog_sha256": sha(catalog), "views_sha256": sha(DATA / "views.npz"),
                "scope": "Q9 only; 0.5 < period < 30 days; balanced subsample; one KOI per star; quality == 0",
                "labels": {"0": "DR25 false positive", "1": "DR25 planet candidate (not necessarily confirmed)"}}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ['usable','scope']}, indent=2))


if __name__ == "__main__":
    main()
