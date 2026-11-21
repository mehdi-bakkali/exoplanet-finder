import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from exofinder.preprocess import prepare, phase_days
from exofinder.train import split_indices, threshold_for, metrics
from exofinder.model import TransitCNN
from exofinder.server import app


def curve():
    rng = np.random.default_rng(2)
    t = np.arange(0, 80, .0204)
    p = phase_days(t, 3, 1)
    f = 1 + .0001 * rng.normal(size=len(t)) - .004 * np.exp(-.5 * (p / .06)**2)
    return t, f


def test_transit_survives_and_epoch_is_periodic():
    t, f = curve()
    g, l, _ = prepare(t, f, 3, 1, 4)
    g2, l2, _ = prepare(t, f, 3, 4, 4)
    assert g.shape == (2, 512) and l.shape == (2, 128)
    assert np.isfinite(g).all() and np.isfinite(l).all()
    assert np.mean(l[0, 58:70]) < np.mean(l[0, :20]) - 5
    np.testing.assert_allclose(g, g2, atol=.01)
    np.testing.assert_allclose(l, l2, atol=.01)


def test_quality_validation_and_input_order():
    t, f = curve()
    expected = prepare(t, f, 3, 1, 4)
    reverse = prepare(t[::-1], f[::-1], 3, 1, 4)
    np.testing.assert_allclose(expected[0], reverse[0])
    with pytest.raises(ValueError):
        prepare(t[:100], f[:100], 3, 1, 4)
    with pytest.raises(ValueError):
        prepare(t, f, 3, 1, -1)
    with pytest.raises(ValueError):
        phase_days(t, 0, 1)


def test_stellar_siblings_never_cross_splits():
    groups = np.repeat(np.arange(100), 2)
    y = np.repeat(np.arange(100) % 2, 2)
    splits = split_indices(y, groups)
    sets = [set(groups[v]) for v in splits.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    assert sorted(np.concatenate(list(splits.values()))) == list(range(200))
    assert all(np.array_equal(v, split_indices(y, groups)[k]) for k, v in splits.items())


def test_model_learns_gradients_from_both_views():
    model = TransitCNN()
    g = torch.randn(3,2,512,requires_grad=True)
    l = torch.randn(3,2,128,requires_grad=True)
    logits = model(g,l)
    assert logits.shape == (3,)
    logits.sum().backward()
    assert g.grad.abs().sum() > 0 and l.grad.abs().sum() > 0


def test_threshold_and_metric_semantics():
    y = np.array([0,0,1,1])
    p = np.array([.1,.2,.7,.9])
    t = threshold_for(y,p)
    result = metrics(y,p,t)
    assert result['confusion_matrix'] == [[2,0],[0,2]]
    assert result['average_precision'] == 1


def test_api_with_trained_artifacts():
    from exofinder.data import ROOT
    if not (ROOT/'runs'/'active.txt').exists():
        pytest.skip('Train a run before integration tests')
    client = TestClient(app)
    assert client.get('/').status_code == 200
    overview = client.get('/api/overview').json()
    obj = overview['objects'][0]
    response = client.get('/api/objects/'+obj['kepoi_name'])
    assert response.status_code == 200
    assert len(response.json()['global_view'][0]) == 512
    assert client.get('/api/objects/nonexistent').status_code == 404
    assert client.get('/api/export/metrics').status_code == 200
    assert client.get('/api/export/nope').status_code == 404
    t,f=curve()
    prediction = client.post('/api/predict', json={'time':t.tolist(),'flux':f.tolist(),'period':3,'epoch':1,'duration':4})
    assert prediction.status_code == 200
    assert 0 <= prediction.json()['score'] <= 1
    bad = client.post('/api/predict', json={'time':t.tolist(),'flux':f[:-1].tolist(),'period':3,'epoch':1,'duration':4})
    assert bad.status_code == 422
