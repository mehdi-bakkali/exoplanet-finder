"""Blind BLS detection and box-transit injection/recovery on real backgrounds."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from scipy.ndimage import median_filter
from .data import ROOT, DATA, sha
from .preprocess import phase_days
from .multiquarter import load_quarter, DEFAULT_DATASET
from .research import research_splits


def blind_flatten(time,flux):
    t,f=np.asarray(time,float),np.asarray(flux,float)
    if t.ndim!=1 or f.ndim!=1 or len(t)!=len(f):raise ValueError('Equal one-dimensional time/flux arrays required')
    valid=np.isfinite(t)&np.isfinite(f)&(f>0);t,f=t[valid],f[valid]
    order=np.argsort(t);t,f=t[order],f[order];t,idx=np.unique(t,return_index=True);f=f[idx]
    if len(t)<200 or np.ptp(t)<3:raise ValueError('Need 200 clean observations spanning three days')
    cadence=np.median(np.diff(t));window=max(11,int(2/cadence)|1)
    result=np.empty_like(f)
    for segment in np.split(np.arange(len(t)),np.flatnonzero(np.diff(t)>.5)+1):
        normalized=f[segment]/np.median(f[segment])
        trend=median_filter(normalized,size=min(window,(len(segment)//2)*2+1),mode='nearest')
        result[segment]=normalized/np.maximum(trend,1e-8)-1
    noise=max(1.4826*np.median(np.abs(result-np.median(result))),1e-6)
    return t,result,noise


def search(time,flux,minimum_period=.5,maximum_period=15):
    if not .25 <= minimum_period < maximum_period <= 30:raise ValueError('Require 0.25 ≤ minimum < maximum ≤ 30 days')
    t,y,noise=blind_flatten(time,flux)
    maximum_period=min(maximum_period,float(np.ptp(t)/2))
    if maximum_period<=minimum_period:raise ValueError('Insufficient baseline for requested period range')
    bls=BoxLeastSquares(t,y,dy=np.full(len(t),noise))
    durations=np.array([.04,.08,.12,.2]);durations=durations[durations<minimum_period]
    result=bls.autopower(durations,minimum_period=minimum_period,maximum_period=maximum_period,
                         frequency_factor=5,oversample=5,objective='snr')
    i=int(np.argmax(result.power));sample=np.unique(np.r_[np.linspace(0,len(result.period)-1,min(500,len(result.period))).astype(int),i])
    return {'period':float(result.period[i]),'epoch':float(result.transit_time[i]),'duration_hours':float(result.duration[i]*24),
            'depth_ppm':float(result.depth[i]*1e6),'depth_snr':float(result.depth_snr[i]),'cadences':len(t),'noise_ppm':noise*1e6,
            'search_minimum_period':minimum_period,'search_maximum_period':maximum_period,'frequency_factor':5,
            'periodogram':{'period':result.period[sample].tolist(),'power':result.power[sample].tolist()},
            'warning':'Strongest BLS box match; depth SNR is not a global false-alarm probability. Requires vetting.'}


def inject(time,flux,period,epoch,duration_days,depth_ppm):
    if period<=0 or not 0<duration_days<period or not 0<depth_ppm<1e6:raise ValueError('Invalid injected transit')
    mask=np.abs(phase_days(time,period,epoch))<duration_days/2
    return np.asarray(flux,float)*(1-depth_ppm*1e-6*mask)


def recovery(result,period,epoch,duration_days):
    ratio=result['period']/period
    period_ok=abs(ratio-1)<.01
    epoch_error=abs(float(phase_days([result['epoch']],period,epoch)[0]))
    return {'strict_recovered':bool(period_ok and epoch_error<duration_days and result['depth_snr']>=7.1),
            'period_match':bool(period_ok),'epoch_error_days':epoch_error,
            'harmonic_match':bool(any(abs(ratio/h-1)<.01 for h in [.5,2]))}


def experiment(stars=8):
    out=ROOT/'runs'/'injection-recovery'
    if (out/'summary.json').exists():raise ValueError('Completed experiment exists; preserve its results')
    out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(DEFAULT_DATASET/'objects.csv')
    split=research_splits(meta.kepid)
    selected=meta.loc[split=='test'].sort_values('kepid').head(stars)
    if len(selected)<stars:raise ValueError('Insufficient test stars')
    plan={'stars':stars,'periods_days':[2.,5.,10.],'depths_ppm':[200.,500.,1500.],
          'duration_days':.12,'seed':2026,'search_period_range':[.5,15.], 'quarters':[9,10],
          'recovery':'Period within 1%, epoch within injected duration, depth SNR >=7.1; harmonics reported separately.',
          'background_selection':'First KICs in fresh test partition, no quiet-star/SNR ranking.',
          'nulls':'One uninjected search per background; SNR threshold is exploratory, not calibrated global FAP.',
          'dataset_sha256':sha(DEFAULT_DATASET/'views.npz'),'code_sha256':sha(Path(__file__))}
    (out/'plan.json').write_text(json.dumps(plan,indent=2));selected[['kepid','kepoi_name']].to_csv(out/'backgrounds.csv',index=False)
    catalog=pd.read_csv(DATA/'catalog.csv');rng=np.random.default_rng(2026);rows=[];nulls=[]
    for _,star in selected.iterrows():
        parts=[load_quarter(star.kepid,q,download=False) for q in [9,10]]
        t=np.concatenate([p[0] for p in parts]);f=np.concatenate([p[1] for p in parts])
        keep=np.ones(len(t),bool)
        # Remove all known short-period KOI windows on the background star, not injected windows.
        for _,signal in catalog.loc[catalog.kepid==star.kepid].dropna(subset=['koi_period','koi_time0bk','koi_duration']).iterrows():
            keep &= np.abs(phase_days(t,signal.koi_period,signal.koi_time0bk))>1.5*signal.koi_duration/24
        t,f=t[keep],f[keep]
        null=search(t,f);nulls.append({'kepid':int(star.kepid),'depth_snr':null['depth_snr'],'period':null['period']})
        for period in plan['periods_days']:
            epoch=float(t.min()+rng.uniform(0,period))
            for depth in plan['depths_ppm']:
                result=search(t,inject(t,f,period,epoch,.12,depth))
                rows.append({'kepid':int(star.kepid),'injected_period':period,'injected_epoch':epoch,'injected_depth_ppm':depth,
                             'recovered_period':result['period'],'recovered_epoch':result['epoch'],'depth_snr':result['depth_snr'],
                             **recovery(result,period,epoch,.12)})
        pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
        print(f"BLS backgrounds {len(nulls)}/{stars}; trials={len(rows)}",flush=True)
    trials=pd.DataFrame(rows);pd.DataFrame(nulls).to_csv(out/'nulls.csv',index=False)
    grouped=trials.groupby(['injected_period','injected_depth_ppm']).strict_recovered.agg(['sum','count','mean']).reset_index()
    summary={'plan':plan,'trials':len(rows),'strict_recovered':int(trials.strict_recovered.sum()),
             'strict_recovery_fraction':float(trials.strict_recovered.mean()),'harmonic_matches':int(trials.harmonic_match.sum()),
             'cells':grouped.to_dict('records'),'null_backgrounds_above_snr_7_1':sum(n['depth_snr']>=7.1 for n in nulls),
             'limitations':['Box injections are simplified, without limb darkening.','Known catalog windows were removed; unknown signals and stellar variability remain.',
                            'Small selected KOI-background experiment, not mission-wide detection completeness.',
                            'BLS detection recovery is separate from CNN vetting; no joint discovery-performance claim.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['plan','cells']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path);p.add_argument('--experiment',action='store_true')
    p.add_argument('--stars',type=int,default=8);p.add_argument('--output',type=Path,default=Path('bls-result.json'))
    p.add_argument('--min-period',type=float,default=.5);p.add_argument('--max-period',type=float,default=15)
    a=p.parse_args()
    if a.experiment:
        if not 1<=a.stars<=100:p.error('Use 1–100 backgrounds')
        experiment(a.stars)
    elif a.csv:
        if a.output.exists():p.error('Output exists; choose a new filename')
        c=pd.read_csv(a.csv);result=search(c.time,c.flux,a.min_period,a.max_period)
        a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='periodogram'},indent=2))
    else:p.error('Supply --csv or --experiment')


if __name__=='__main__':main()
