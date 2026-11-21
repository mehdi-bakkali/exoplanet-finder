"""Restore and verify raw FITS caches without overwriting published processed data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import pandas as pd
from .multiquarter import DEFAULT_DATASET, load_quarter
from .research import research_splits


def restore(row):
    expected={s['quarter']:s['sha256'] for s in json.loads(row['sources'])}
    for q in [9,10]:
        _,_,source=load_quarter(row['kepid'],q,download=True)
        if source['sha256']!=expected[q]:raise ValueError(f"Source hash changed for KIC {row['kepid']} Q{q}")
    return int(row['kepid'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backgrounds',type=int,default=8,help='First test KICs used by injection experiment; 0 restores all usable stars')
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if a.backgrounds<0 or not 1<=a.workers<=32:p.error('Nonnegative backgrounds and 1–32 workers required')
    meta=pd.read_csv(DEFAULT_DATASET/'objects.csv')
    if a.backgrounds:
        meta=meta.loc[research_splits(meta.kepid)=='test'].sort_values('kepid').head(a.backgrounds)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,_ in enumerate(pool.map(restore,meta.to_dict('records')),1):
            if i%20==0 or i==len(meta):print(f'Verified raw FITS for {i}/{len(meta)} stars',flush=True)


if __name__=='__main__':main()
