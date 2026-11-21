"""Versioned multi-quarter acquisition; never modifies the bundled pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import shutil
import time
import threading
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from .data import DATA, ROOT, sha
from .preprocess import prepare

STAMPS = {9: '2011177032512', 10: '2011271113734'}
DEFAULT_DATASET = DATA / 'research-q9-q10'
_SESSIONS = threading.local()

def session():
    if not hasattr(_SESSIONS, 'http'):
        _SESSIONS.http = requests.Session()
    return _SESSIONS.http


def load_quarter(kepid, quarter, download=True):
    kic = f'{int(kepid):09d}'
    path = DATA/'raw'/f'{kic}-q{quarter}.fits'
    url = f'https://stpubdata.s3.amazonaws.com/kepler/public/lightcurves/{kic[:4]}/{kic}/kplr{kic}-{STAMPS[quarter]}_llc.fits'
    if not path.exists():
        if not download:
            raise FileNotFoundError(f'Raw quarter missing: {path.name}; reacquire the dataset first.')
        for attempt in range(3):
            try:
                r = session().get(url, timeout=(10, 30))
                r.raise_for_status()
                tmp = path.with_suffix('.tmp')
                tmp.write_bytes(r.content); tmp.replace(path)
                break
            except requests.RequestException as exc:
                if getattr(exc.response, 'status_code', None) == 404 or attempt == 2:
                    raise
                time.sleep(attempt + 1)
    with fits.open(path, memmap=False) as h:
        if h[0].header.get('QUARTER') != quarter or h[0].header.get('KEPLERID') != int(kepid):
            raise ValueError('FITS identity/quarter mismatch')
        c = h[1].data
        valid = (c['SAP_QUALITY'] == 0) & np.isfinite(c['TIME']) & np.isfinite(c['PDCSAP_FLUX']) & (c['PDCSAP_FLUX'] > 0)
        t, f = np.asarray(c['TIME'][valid], float), np.asarray(c['PDCSAP_FLUX'][valid], float)
    if len(t) < 200:
        raise ValueError('Fewer than 200 clean observations in quarter')
    return t, f / np.median(f), {'quarter':quarter, 'url':url, 'sha256':sha(path), 'cadences':len(t)}


def acquire(row):
    try:
        parts = [load_quarter(row['kepid'], q) for q in STAMPS]
        t = np.concatenate([p[0] for p in parts]); f = np.concatenate([p[1] for p in parts])
        g, l, info = prepare(t, f, row['koi_period'], row['koi_time0bk'], row['koi_duration'])
        record = {**row, **info, 'label':int(row['koi_pdisposition']=='CANDIDATE'),
                  'quarters':'9,10', 'sources':json.dumps([p[2] for p in parts]),
                  'global_coverage':float(g[1].mean()), 'local_coverage':float(l[1].mean())}
        return record, g, l, None
    except Exception as exc:
        return row, None, None, str(exc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit', type=int, default=4000)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.limit < 100 or not 1 <= args.workers <= 128:
        p.error('Use >=100 objects and 1–128 workers.')
    out = DEFAULT_DATASET
    if (out/'manifest.json').exists():
        p.error('Completed dataset exists; preserve it and use its processed views.')
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        p.error('Need at least 5 GB free for raw observations.')
    out.mkdir(parents=True, exist_ok=True); (DATA/'raw').mkdir(exist_ok=True)
    df = pd.read_csv(DATA/'catalog.csv').dropna(subset=['koi_period','koi_time0bk','koi_duration'])
    pilot = set(pd.read_csv(DATA/'objects.csv').kepid)
    df = df[~df.kepid.isin(pilot)]
    df = df[df.koi_pdisposition.isin(['CANDIDATE','FALSE POSITIVE'])]
    df = df[(df.koi_duration > 0) & (df.koi_duration / 24 < df.koi_period / 4)]
    eligible = df.sample(frac=1, random_state=2026).drop_duplicates('kepid')
    selected = eligible.iloc[:args.limit].copy()  # No balancing, flags or S/N ranking.
    selected.to_csv(out/'selection.csv', index=False)
    records, gs, ls, failures = [], [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (row,g,l,error) in enumerate(pool.map(acquire, selected.to_dict('records')),1):
            if error: failures.append({'id':row['kepoi_name'],'kepid':int(row['kepid']),'reason':error})
            else: records.append(row); gs.append(g); ls.append(l)
            if i%50==0:
                print(f'{i}/{len(selected)} processed; usable={len(records)} excluded={len(failures)}',flush=True)
    if not records: raise RuntimeError('No usable observations')
    meta=pd.DataFrame(records); meta.to_csv(out/'objects.csv',index=False)
    np.savez_compressed(out/'views.npz',global_view=np.array(gs),local_view=np.array(ls),y=meta.label.to_numpy())
    manifest={'source':'NASA Kepler DR25 KOIs and Q9/Q10 PDCSAP FITS','created_utc':datetime.now(timezone.utc).isoformat(),
              'selection_seed':2026,'requested':args.limit,'eligible_stars':len(eligible),'selected':len(selected),
              'usable':len(meta),'quarters':list(STAMPS),'required_quarters':'both', 'pilot_stars_excluded':len(pilot),
              'sampling':'Random eligible star cohort without class balancing; one KOI per star.',
              'population':'DR25 KOI stars, 0.5<period<30d, valid duration, Q9 and Q10 available; not all Kepler stars.',
              'selection_counts':selected.koi_pdisposition.value_counts().to_dict(),'usable_counts':meta.koi_pdisposition.value_counts().to_dict(),
              'catalog_sha256':sha(DATA/'catalog.csv'),'views_sha256':sha(out/'views.npz'),'objects_sha256':sha(out/'objects.csv'),
              'failures':failures}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='failures'},indent=2))


if __name__=='__main__': main()
