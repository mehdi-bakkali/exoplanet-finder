"""Local-only API and static research interface. Start from the project directory."""
from functools import lru_cache
import json
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from .data import ROOT, DATA
from .model import TransitCNN
from .preprocess import prepare, summary_features
from .diagnostics import build_diagnostics, operating_point
from scipy.special import expit

app = FastAPI(title='Exoplanet Finder', version='0.2.0')


@lru_cache(maxsize=1)
def state():
    active = ROOT / 'runs' / 'active.txt'
    if not active.exists():
        raise HTTPException(503, 'No trained run. Run python -m exofinder.data, then python -m exofinder.train.')
    run_name = active.read_text().strip()
    if not run_name or '/' in run_name or run_name in ('.','..'):
        raise HTTPException(503, 'Invalid active run')
    run = ROOT / 'runs' / run_name
    report = json.loads((run/'metrics.json').read_text())
    from .data import sha
    if report['dataset_sha256'] != sha(DATA/'views.npz'):
        raise HTTPException(503, 'Dataset changed since training. Train a new run before serving.')
    meta = pd.read_csv(run/'predictions.csv').fillna('')
    views = np.load(DATA/'views.npz')
    model = TransitCNN()
    model.load_state_dict(torch.load(run/'model.pt', map_location='cpu', weights_only=True)['state_dict'])
    model.eval()
    return run, report, meta, views, model


@app.get('/api/health')
def health():
    _, report, meta, _, _ = state()
    return {'status':'ready','objects':len(meta),'run':report['run']}


@app.get('/api/overview')
def overview():
    run, report, meta, _, _ = state()
    return {'metrics':report,'objects':meta.to_dict('records'),
            'history':json.loads((run/'history.json').read_text()),
            'manifest':json.loads((DATA/'manifest.json').read_text())}


@app.get('/api/objects/{object_id}')
def object_view(object_id: str):
    _, report, meta, views, _ = state()
    matches = np.flatnonzero(meta.kepoi_name.values == object_id)
    if not len(matches):
        raise HTTPException(404, 'Unknown KOI')
    idx = int(matches[0])
    return {'object':meta.iloc[idx].to_dict(), 'global_view':views['global_view'][idx].tolist(),
            'local_view':views['local_view'][idx].tolist(), 'threshold':report['threshold']}


@app.get('/api/diagnostics')
def diagnostics():
    _, report, meta, _, _ = state()
    return build_diagnostics(meta, report)


@app.get('/api/example')
def example():
    _, _, meta, _, _ = state()
    rows = meta.loc[meta.kepoi_name == 'K00166.01']
    file = ROOT/'examples'/'kepler-479-b.csv'
    if rows.empty or not file.exists():
        raise HTTPException(404, 'The bundled example is unavailable for this dataset.')
    row = rows.iloc[0]
    return {'name': 'Kepler-479 b / K00166.01', 'csv': file.read_text(),
            'period':float(row.koi_period), 'epoch':float(row.koi_time0bk), 'duration':float(row.koi_duration)}


@app.get('/api/validation-threshold')
def validation_threshold(threshold: float = Query(ge=0, le=1, allow_inf_nan=False)):
    _, report, meta, _, _ = state()
    validation = meta.loc[meta.split == 'validation']
    return {'split': 'validation', 'saved_threshold': report['threshold'],
            **operating_point(validation.label, validation.score, threshold)}


class CurveInput(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    time: list[float] = Field(min_length=200, max_length=100000)
    flux: list[float] = Field(min_length=200, max_length=100000)
    period: float = Field(gt=0, le=10000)
    epoch: float
    duration: float = Field(gt=0)


@app.post('/api/predict')
def predict(body: CurveInput):
    run, report, _, _, model = state()
    if len(body.time) != len(body.flux):
        raise HTTPException(422, 'Time and flux arrays must have equal length.')
    try:
        g, l, info = prepare(body.time, body.flux, body.period, body.epoch, body.duration)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with torch.no_grad():
        score = float(model(torch.tensor(g[None]), torch.tensor(l[None])).sigmoid().item())
    baseline = json.loads((run/'baseline.json').read_text())
    features = summary_features(g[None], l[None])
    standardized = (features - np.array(baseline['mean'])) / np.array(baseline['scale'])
    baseline_score = float(expit(standardized @ np.array(baseline['coef']).T + np.array(baseline['intercept'])).item())
    info.update(global_coverage=float(g[1].mean()), local_coverage=float(l[1].mean()),
                supplied_observations=len(body.time), discarded_observations=len(body.time)-info['cadences'])
    return {'score':score,'threshold':report['threshold'],'global_view':g.tolist(),'local_view':l.tolist(),
            'baseline_score':baseline_score, 'baseline_threshold':report['baseline_threshold'],
            'run':report['run'], 'in_training_period_range':0.5 < body.period < 30,
            'ephemeris':{'period':body.period,'epoch':body.epoch,'duration':body.duration},
            'info':info,'warning':'Uncalibrated candidate score. Training domain: Kepler Q9, periods 0.5–30 days. Not a confirmation.'}


@app.get('/api/export/{kind}')
def export(kind: str):
    run, *_ = state()
    if kind == 'diagnostics':
        return JSONResponse(diagnostics(), headers={'Content-Disposition':'attachment; filename="diagnostics.json"'})
    files = {'predictions':'predictions.csv','metrics':'metrics.json','splits':'splits.json'}
    if kind not in files:
        raise HTTPException(404,'Unknown export')
    return FileResponse(run/files[kind], filename=files[kind])


@app.get('/')
def index():
    return FileResponse(ROOT/'web'/'index.html')


app.mount('/static', StaticFiles(directory=ROOT/'web'), name='static')
