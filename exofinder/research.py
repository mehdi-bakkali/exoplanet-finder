"""Predeclared multi-seed ablations with independent calibration and test stars."""
import argparse
import hashlib
import json
import random
import platform
import sklearn
import scipy
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .preprocess import summary_features
from sklearn.metrics import average_precision_score, log_loss, brier_score_loss
from .data import ROOT, DATA, sha
from .model import TransitCNN, device_for
from .train import metrics, threshold_for
from .diagnostics import reliability
from .multiquarter import DEFAULT_DATASET

MODES = ('dual','global','local','mask')
SEEDS = (42,43,44)
RESEARCH = ROOT/'runs'/'research-q9-q10'


def research_splits(groups):
    """Fixed hash allocation preserves natural prevalence; all seeds share the split."""
    assigned=[]
    for group in groups:
        value=int(hashlib.sha256(f'kepler-research-2026-v1:{int(group)}'.encode()).hexdigest()[:12],16)/16**12
        assigned.append('train' if value<.55 else 'validation' if value<.70 else 'calibration' if value<.85 else 'test')
    return np.asarray(assigned)


def ablation_views(g,l,mode):
    if mode not in MODES: raise ValueError('Unknown ablation')
    g,l=np.array(g,copy=True),np.array(l,copy=True)
    if mode=='global': l[:]=0
    elif mode=='local': g[:]=0
    elif mode=='mask': g[:,0]=0; l[:,0]=0
    return g,l


def fit_platt(scores,labels):
    """Monotonic sigmoid on logits; fitted ONLY on the calibration partition."""
    x=logit(np.clip(np.asarray(scores,float),1e-7,1-1e-7)); y=np.asarray(labels,float)
    if len(np.unique(y))!=2: raise ValueError('Calibration needs both classes')
    def objective(params):
        z=params[0]*x+params[1]
        return np.mean(np.logaddexp(0,z)-y*z)
    result=minimize(objective,[1.,0.],method='L-BFGS-B',bounds=[(.0001,100),(-100,100)])
    if not result.success: raise RuntimeError('Calibration optimizer failed: '+str(result.message))
    return {'slope':float(result.x[0]),'intercept':float(result.x[1]),'method':'monotonic Platt sigmoid on clipped logits',
            'fit_count':len(y),'fit_prevalence':float(y.mean())}


def calibrate(scores,parameters):
    return expit(parameters['slope']*logit(np.clip(scores,1e-7,1-1e-7))+parameters['intercept'])


def score_report(y,p):
    return {'average_precision':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)), 'log_loss':float(log_loss(y,p,labels=[0,1])),
            'reliability':reliability(y,p)}


def predict_batches(model,g,l,device):
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(g[i:i+128],device=device),torch.tensor(l[i:i+128],device=device)).sigmoid().cpu().numpy()
                               for i in range(0,len(g),128)])


def train_one(meta,g,l,split,mode,seed,epochs,device,out):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    g,l=ablation_views(g,l,mode)
    ids={name:np.flatnonzero(split==name) for name in ['train','validation','calibration','test']}
    y=meta.label.to_numpy(dtype=int);tr=ids['train'];va=ids['validation'];ca=ids['calibration'];te=ids['test']
    model=TransitCNN().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y[tr]==0).sum()/(y[tr]==1).sum()],dtype=torch.float32,device=device))
    history=[];best_ap=-1;stale=0;start=time.time()
    for epoch in range(epochs):
        model.train();losses=[]
        for batch in np.array_split(np.random.permutation(tr),max(1,int(np.ceil(len(tr)/64)))):
            bg=torch.tensor(g[batch],device=device);bl=torch.tensor(l[batch],device=device)
            if random.random()<.5: bg,bl=bg.flip(-1),bl.flip(-1)
            opt.zero_grad();loss=criterion(model(bg,bl),torch.tensor(y[batch],dtype=torch.float32,device=device))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.item())
        model.eval();vp=predict_batches(model,g[va],l[va],device)
        ap=float(average_precision_score(y[va],vp))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_ap':ap})
        if ap>best_ap+1e-5:
            best_ap,stale,best_epoch=ap,0,epoch+1
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>=6: break
    model.load_state_dict(best);model.eval()
    scores=predict_batches(model,g,l,device)
    threshold=threshold_for(y[va],scores[va])
    params=fit_platt(scores[ca],y[ca]);calibrated=calibrate(scores,params)
    calibrated_threshold=float(calibrate(np.array([threshold]),params)[0])
    result={'mode':mode,'device':str(device),'seed':seed,'best_epoch':best_epoch,'epochs_run':len(history),'seconds':time.time()-start,
            'validation_ap':best_ap,'threshold':threshold,'calibrated_threshold':calibrated_threshold,
            'raw_test':metrics(y[te],scores[te],threshold), 'raw_probability_metrics':score_report(y[te],scores[te]),
            'calibrated_test':score_report(y[te],calibrated[te]),'calibrator':params,
            'counts':{k:len(v) for k,v in ids.items()}}
    pd.DataFrame({'kepoi_name':meta.kepoi_name,'kepid':meta.kepid,'label':y,'split':split,'score':scores,'calibrated_score':calibrated}).to_csv(out/'predictions.csv',index=False)
    torch.save({'state_dict':best,'ablation':mode,'seed':seed,'threshold':threshold},out/'model.pt')
    (out/'history.json').write_text(json.dumps(history,indent=2))
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--device',choices=['auto','mps','cpu'],default='auto')
    args=p.parse_args()
    if args.epochs<1:p.error('Epochs must be positive')
    RESEARCH.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(DEFAULT_DATASET/'objects.csv');data=np.load(DEFAULT_DATASET/'views.npz')
    if not np.array_equal(data['y'],meta.label.to_numpy()):raise ValueError('Label/view alignment mismatch')
    if set(meta.kepid)&set(pd.read_csv(DATA/'objects.csv').kepid):raise ValueError('Pilot star leakage')
    split=research_splits(meta.kepid)
    for name in ['train','validation','calibration','test']:
        if len(np.unique(meta.loc[split==name,'label']))!=2:raise ValueError('Both labels needed per partition')
    device=device_for(args.device)
    plan={'device':str(device),'dataset_sha256':sha(DEFAULT_DATASET/'views.npz'),'metadata_sha256':sha(DEFAULT_DATASET/'objects.csv'),
          'modes':list(MODES),'seeds':list(SEEDS),'epochs':args.epochs,'patience':6,'split_rule':'SHA256 kepler-research-2026-v1:KIC; 55/15/15/15',
          'calibration':'Fit monotonic Platt on calibration only; report on test; no model selection on calibration.',
          'ensemble':'Mean of three predeclared dual-view seed scores; calibrate that mean on calibration only.',
          'code_sha256':sha(Path(__file__)),'model_code_sha256':sha(ROOT/'exofinder/model.py'),
          'preprocessing_sha256':sha(ROOT/'exofinder/preprocess.py'),
          'acquisition_sha256':sha(ROOT/'exofinder/multiquarter.py'),
          'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}}
    path=RESEARCH/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Existing experiment plan differs; preserve the completed study.')
    path.write_text(json.dumps(plan,indent=2))
    pd.DataFrame({'kepid':meta.kepid,'kepoi_name':meta.kepoi_name,'split':split}).to_csv(RESEARCH/'splits.csv',index=False)
    torch.set_num_threads(4);results=[]
    for mode in MODES:
        for seed in SEEDS:
            out=RESEARCH/f'{mode}-seed-{seed}';out.mkdir(exist_ok=True)
            if (out/'metrics.json').exists():result=json.loads((out/'metrics.json').read_text())
            else:result=train_one(meta,data['global_view'],data['local_view'],split,mode,seed,args.epochs,device,out)
            results.append(result)
            print(f"{mode} seed={seed} test_AP={result['raw_test']['average_precision']:.4f} calibrated_Brier={result['calibrated_test']['brier']:.4f}",flush=True)
    summary={'plan':plan,'dataset_size':len(meta),'device':str(device),'counts':results[0]['counts'],'modes':{}}
    for mode in MODES:
        rows=[r for r in results if r['mode']==mode]
        summary['modes'][mode]={}
        for metric in ['average_precision','roc_auc','f1','brier']:
            values=[r['raw_test'][metric] for r in rows]
            summary['modes'][mode][metric]={'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)),'values':values}
    scores=np.mean([pd.read_csv(RESEARCH/f'dual-seed-{seed}'/'predictions.csv').score.to_numpy() for seed in SEEDS],axis=0)
    ca=split=='calibration';te=split=='test';y=meta.label.to_numpy()
    params=fit_platt(scores[ca],y[ca]);cp=calibrate(scores,params)
    summary['ensemble']={'raw_test':score_report(y[te],scores[te]),'calibrated_test':score_report(y[te],cp[te]),'calibrator':params}
    features=summary_features(data['global_view'],data['local_view'])
    baseline=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42))
    baseline.fit(features[split=='train'],y[split=='train'])
    bp=baseline.predict_proba(features)[:,1];bt=threshold_for(y[split=='validation'],bp[split=='validation'])
    bc=fit_platt(bp[ca],y[ca])
    summary['baseline']={'raw_test':metrics(y[te],bp[te],bt),'calibrated_test':score_report(y[te],calibrate(bp[te],bc)),'threshold':bt,'calibrator':bc}
    pd.DataFrame({'kepoi_name':meta.kepoi_name,'split':split,'score':bp,'label':y,'calibrated_score':calibrate(bp,bc)}).to_csv(RESEARCH/'baseline-predictions.csv',index=False)
    scaler,lr=baseline.steps[0][1],baseline.steps[1][1]
    (RESEARCH/'baseline.json').write_text(json.dumps({'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coef':lr.coef_.tolist(),'intercept':lr.intercept_.tolist()}))
    pd.DataFrame({'kepoi_name':meta.kepoi_name,'kepid':meta.kepid,'label':y,'split':split,'score':scores,'calibrated_score':cp}).to_csv(RESEARCH/'ensemble-predictions.csv',index=False)
    (RESEARCH/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({'dataset_size':len(meta),'counts':summary['counts'],'ensemble':summary['ensemble']},indent=2))


if __name__=='__main__':main()
