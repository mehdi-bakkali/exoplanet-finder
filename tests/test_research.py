import numpy as np
import pandas as pd
import pytest
from scipy.special import expit,logit
from sklearn.metrics import brier_score_loss
from fastapi.testclient import TestClient
from exofinder.research import research_splits,ablation_views,fit_platt,calibrate
from exofinder.bls import inject,search,recovery
from exofinder.server import app


def test_fixed_stellar_partitions_are_order_invariant():
    groups=np.repeat(np.arange(1000),2)
    split=research_splits(groups)
    assert np.array_equal(split[::2],split[1::2])
    assert np.array_equal(research_splits(groups[::-1])[::-1],split)
    assert set(split)=={'train','validation','calibration','test'}
    sets=[set(groups[split==name]) for name in np.unique(split)]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[:i])


def test_ablation_removes_only_designated_information():
    g=np.ones((4,2,512),dtype='float32');l=np.ones((4,2,128),dtype='float32')
    gm,lm=ablation_views(g,l,'mask')
    assert not gm[:,0].any() and not lm[:,0].any()
    assert gm[:,1].all() and lm[:,1].all()
    gg,lg=ablation_views(g,l,'global');gl,ll=ablation_views(g,l,'local')
    assert gg.all() and not lg.any() and not gl.any() and ll.all()
    assert g.all() and l.all()  # No in-place destruction of other ablations.
    with pytest.raises(ValueError):ablation_views(g,l,'unknown')


def test_separate_calibration_improves_distorted_scores_and_preserves_rank():
    rng=np.random.default_rng(2);p=rng.uniform(.02,.98,8000);y=rng.binomial(1,p)
    distorted=expit(.3*logit(p)+1)
    params=fit_platt(distorted[:5000],y[:5000])
    held=calibrate(distorted[5000:],params)
    assert params['fit_count']==5000
    assert brier_score_loss(y[5000:],held)<brier_score_loss(y[5000:],distorted[5000:])
    assert np.all(np.diff(calibrate(np.linspace(0,1,100),params))>0)
    with pytest.raises(ValueError):fit_platt([.2,.3],[1,1])


def test_blind_search_recovers_strong_injection_without_supplied_ephemeris():
    rng=np.random.default_rng(7);t=np.arange(0,30,.0204);base=1+rng.normal(0,.0001,len(t))
    result=search(t,inject(t,base,3.,1.,.12,1500),.5,10)
    assert recovery(result,3.,1.,.12)['strict_recovered']
    assert abs(result['period']-3)<.03
    alias=recovery({'period':6.,'epoch':1.,'depth_snr':20},3.,1.,.12)
    assert alias['harmonic_match'] and not alias['strict_recovered']
    with pytest.raises(ValueError):search(t,base,3,2)


def test_bls_api_validates_lengths_and_bounds():
    client=TestClient(app);t=np.arange(0,10,.02);f=np.ones(len(t))
    bad=client.post('/api/bls',json={'time':t.tolist(),'flux':f[:-1].tolist()})
    assert bad.status_code==422
    bad=client.post('/api/bls',json={'time':t.tolist(),'flux':f.tolist(),'minimum_period':5,'maximum_period':2})
    assert bad.status_code==422
    report=client.get('/api/research')
    assert report.status_code==200 and set(report.json())=={'dataset','study','injection'}
    assert report.headers['cache-control']=='no-store'


def test_test_labels_cannot_change_weights_threshold_or_calibrator(tmp_path):
    import torch
    from exofinder.research import train_one
    torch.set_num_threads(1)
    rng=np.random.default_rng(3)
    g=rng.normal(size=(80,2,512)).astype('float32');g[:,1]=1
    l=rng.normal(size=(80,2,128)).astype('float32');l[:,1]=1
    split=np.repeat(['train','validation','calibration','test'],20)
    meta=pd.DataFrame({'kepid':np.arange(80),'kepoi_name':['K'+str(i) for i in range(80)],'label':np.arange(80)%2})
    first=tmp_path/'first';second=tmp_path/'second';first.mkdir();second.mkdir()
    a=train_one(meta,g,l,split,'dual',42,1,torch.device('cpu'),first)
    changed=meta.copy();changed.loc[split=='test','label']=1-changed.loc[split=='test','label']
    b=train_one(changed,g,l,split,'dual',42,1,torch.device('cpu'),second)
    assert a['threshold']==b['threshold'] and a['calibrator']==b['calibrator']
    wa=torch.load(first/'model.pt',weights_only=True)['state_dict'];wb=torch.load(second/'model.pt',weights_only=True)['state_dict']
    assert all(torch.equal(wa[k],wb[k]) for k in wa)


def test_quarter_cache_checks_identity_quality_and_normalization(tmp_path,monkeypatch):
    from astropy.io import fits
    import exofinder.multiquarter as mq
    monkeypatch.setattr(mq,'DATA',tmp_path)
    (tmp_path/'raw').mkdir()
    n=250
    columns=[fits.Column(name='TIME',format='D',array=np.arange(n)*.02),
             fits.Column(name='PDCSAP_FLUX',format='D',array=np.full(n,12000.)),
             fits.Column(name='SAP_QUALITY',format='J',array=np.r_[1,np.zeros(n-1,dtype=int)])]
    primary=fits.PrimaryHDU();primary.header['QUARTER']=9;primary.header['KEPLERID']=123
    path=tmp_path/'raw/000000123-q9.fits'
    fits.HDUList([primary,fits.BinTableHDU.from_columns(columns)]).writeto(path)
    t,f,record=mq.load_quarter(123,9,download=False)
    assert len(t)==249 and np.median(f)==1
    assert record['quarter']==9 and len(record['sha256'])==64
    primary.header['KEPLERID']=999
    fits.HDUList([primary,fits.BinTableHDU.from_columns(columns)]).writeto(path,overwrite=True)
    with pytest.raises(ValueError,match='identity'):mq.load_quarter(123,9,download=False)
    with pytest.raises(FileNotFoundError):mq.load_quarter(123,10,download=False)
