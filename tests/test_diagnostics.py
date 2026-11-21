import hashlib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from exofinder.diagnostics import operating_point, reliability, build_diagnostics
from exofinder.server import app, state
from exofinder.data import ROOT


def test_reliability_includes_endpoints_and_omits_empty_bins():
    bins = reliability([0, 0, 1, 1], [0., .1, .9, 1.], bins=4)
    assert len(bins) == 2
    assert sum(b['count'] for b in bins) == 4
    assert bins[0]['candidate_fraction'] == 0
    assert bins[1]['candidate_fraction'] == 1
    assert bins[0]['fraction_95ci'][1] > 0
    assert bins[1]['fraction_95ci'][0] < 1


def test_empty_denominators_are_undefined_not_misleading_zeros():
    none = operating_point([0, 0], [.1, .2], .5)
    assert none['precision'] is None and none['recall'] is None
    assert none['confusion_matrix'] == [[2, 0], [0, 0]]
    all_selected = operating_point([0, 1], [.2, .8], 0)
    assert all_selected['precision'] == .5 and all_selected['recall'] == 1


@pytest.fixture
def trained():
    if not (ROOT/'runs'/'active.txt').exists():
        pytest.skip('Requires included trained artifacts')
    return state()


def test_diagnostics_never_include_training_or_validation(trained):
    _, report, meta, _, _ = trained
    d = build_diagnostics(meta, report)
    assert d['count'] == report['counts']['test']
    assert len(d['errors']) == 14
    assert {e['split'] for e in d['errors']} == {'test'}
    assert sum(b['count'] for b in d['models']['cnn']['reliability']) == d['count']
    changed = meta.copy()
    changed.loc[changed.split != 'test', ['score','baseline_score','label']] = 0
    assert build_diagnostics(changed, report) == d


def test_threshold_sandbox_is_validation_only_and_preserves_artifacts(trained):
    run, report, meta, _, _ = trained
    files = [run/'metrics.json', run/'model.pt', run/'predictions.csv']
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    client = TestClient(app)
    for threshold in [0, report['threshold'], 1]:
        response = client.get('/api/validation-threshold', params={'threshold':threshold})
        assert response.status_code == 200
        d = response.json()
        assert d['split'] == 'validation' and d['count'] == report['counts']['validation']
        val = meta[meta.split == 'validation']
        assert d['confusion_matrix'] == operating_point(val.label, val.score, threshold)['confusion_matrix']
    assert client.get('/api/validation-threshold?threshold=1.1').status_code == 422
    assert client.get('/api/validation-threshold?threshold=nan').status_code == 422
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]


def test_real_example_matches_saved_cnn_and_baseline(trained):
    _, _, meta, _, _ = trained
    client = TestClient(app)
    sample = client.get('/api/example').json()
    import io
    curve = pd.read_csv(io.StringIO(sample['csv']))
    response = client.post('/api/predict',json={
        'time':curve.time.tolist(),'flux':curve.flux.tolist(),
        **{k:sample[k] for k in ['period','epoch','duration']}})
    assert response.status_code == 200
    p = response.json()
    saved = meta.loc[meta.kepoi_name == 'K00166.01'].iloc[0]
    assert p['score'] == pytest.approx(saved.score, abs=1e-5)
    assert p['baseline_score'] == pytest.approx(saved.baseline_score, abs=1e-5)
    assert 0 < p['info']['local_coverage'] <= 1
    assert p['info']['discarded_observations'] == 0
    assert p['in_training_period_range'] is True
    download = client.get('/api/export/diagnostics')
    assert download.status_code == 200
    assert 'attachment' in download.headers['content-disposition']
