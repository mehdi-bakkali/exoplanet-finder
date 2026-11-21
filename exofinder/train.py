"""Train on train only; select epoch/threshold on validation; report held-out test."""
import argparse
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import DATA, ROOT, sha
from .model import TransitCNN, device_for
from .preprocess import summary_features


def split_indices(y, groups, seed=42):
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    dev, test = next(outer.split(np.zeros(len(y)), y, groups))
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed + 1)
    tr, va = next(inner.split(np.zeros(len(dev)), y[dev], groups[dev]))
    parts = {'train': dev[tr], 'validation': dev[va], 'test': test}
    sets = [set(groups[v]) for v in parts.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    if any(len(np.unique(y[v])) < 2 for v in parts.values()):
        raise ValueError("Both labels are required in every split; acquire more objects.")
    return parts


def threshold_for(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    return float(thresholds[int(np.argmax(f1))])


def metrics(y, p, threshold):
    pred = p >= threshold
    return {'average_precision': float(average_precision_score(y, p)), 'roc_auc': float(roc_auc_score(y, p)),
            'precision': float(precision_score(y, pred, zero_division=0)), 'recall': float(recall_score(y, pred)),
            'f1': float(f1_score(y, pred)), 'brier': float(brier_score_loss(y, p)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def bootstrap(y, p, seed):
    rng, values = np.random.default_rng(seed), []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            values.append(average_precision_score(y[idx], p[idx]))
    return np.percentile(values, [2.5, 97.5]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto','cpu','mps'], default='auto')
    parser.add_argument('--run', default='pilot')
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        parser.error('Run name must be a simple directory name')
    out = ROOT / 'runs' / args.run
    if (out / 'metrics.json').exists():
        parser.error('Completed run exists; choose a new --run name to preserve results.')
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = device_for(args.device)
    data = np.load(DATA / 'views.npz')
    meta = pd.read_csv(DATA / 'objects.csv')
    y = data['y'].astype(int)
    split = split_indices(y, meta.kepid.values, args.seed)
    (out / 'splits.json').write_text(json.dumps({k: meta.iloc[v].kepoi_name.tolist() for k,v in split.items()}, indent=2))
    g = torch.tensor(data['global_view'], device=device)
    l = torch.tensor(data['local_view'], device=device)
    labels = torch.tensor(y, dtype=torch.float32, device=device)
    model = TransitCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    tr, va, te = [split[k] for k in ['train','validation','test']]
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y[tr] == 0).sum() / (y[tr] == 1).sum()], dtype=torch.float32, device=device))
    history, best_ap, stale = [], -1, 0
    start = time.time()
    for epoch in range(args.epochs):
        model.train(); losses = []
        for batch in np.array_split(np.random.permutation(tr), max(1, int(np.ceil(len(tr)/32)))):
            idx = torch.tensor(batch, device=device)
            bg, bl = g[idx], l[idx]
            # Phase reversal preserves transit morphology. Occupancy channels are reversed together.
            if random.random() < .5:
                bg, bl = bg.flip(-1), bl.flip(-1)
            optimizer.zero_grad()
            loss = loss_fn(model(bg, bl), labels[idx]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vp = model(g[torch.tensor(va, device=device)], l[torch.tensor(va, device=device)]).sigmoid().cpu().numpy()
        ap = float(average_precision_score(y[va], vp))
        history.append({'epoch': epoch+1, 'train_loss': float(np.mean(losses)), 'validation_ap': ap})
        if ap > best_ap + 1e-5:
            best_ap, stale, best_epoch = ap, 0, epoch + 1
            best = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        print(f"epoch={epoch+1:02d} loss={np.mean(losses):.4f} validation_AP={ap:.3f} device={device}", flush=True)
        if stale >= 12:
            break
    model.load_state_dict(best); model.eval()
    with torch.no_grad():
        probabilities = model(g,l).sigmoid().cpu().numpy()
    threshold = threshold_for(y[va], probabilities[va])
    features = summary_features(data['global_view'], data['local_view'])
    baseline = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', random_state=args.seed, max_iter=2000))
    baseline.fit(features[tr], y[tr])
    bp = baseline.predict_proba(features)[:,1]
    bt = threshold_for(y[va], bp[va])
    report = {'run': args.run, 'seed': args.seed, 'device': str(device), 'seconds': time.time()-start,
              'parameters': sum(p.numel() for p in model.parameters()), 'best_epoch': best_epoch,
              'threshold': threshold, 'baseline_threshold': bt,
              'dataset_sha256': sha(DATA/'views.npz'), 'counts': {k:len(v) for k,v in split.items()},
              'test_prevalence': float(y[te].mean()), 'cnn': metrics(y[te],probabilities[te],threshold),
              'baseline': metrics(y[te],bp[te],bt),
              'test_ap_95ci': bootstrap(y[te],probabilities[te],args.seed),
              'validation': metrics(y[va],probabilities[va],threshold),
              'limitations': ['Pilot subset, not mission-wide performance.', 'DR25 candidates are not confirmed planets.',
                              'Ephemeris supplied by catalog: this vets known signals, not a blind transit search.',
                              'One quarter; short periods; balanced sampling changes prevalence.',
                              'Uncalibrated scores are not astrophysical probabilities.',
                              'MPS seeds improve repeatability but do not guarantee bitwise deterministic kernels.']}
    # Persist baseline coefficients so inference can be reproduced without pickle.
    scaler, lr = baseline.steps[0][1], baseline.steps[1][1]
    (out/'baseline.json').write_text(json.dumps({'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coef':lr.coef_.tolist(),'intercept':lr.intercept_.tolist()}))
    torch.save({'state_dict':best,'threshold':threshold,'seed':args.seed}, out/'model.pt')
    meta['score'] = probabilities; meta['baseline_score'] = bp
    meta['split'] = ''
    for name, ids in split.items():
        meta.loc[ids, 'split'] = name
    meta.to_csv(out/'predictions.csv', index=False)
    (out/'history.json').write_text(json.dumps(history, indent=2))
    (out/'metrics.json').write_text(json.dumps(report, indent=2))
    (ROOT/'runs'/'active.txt').write_text(args.run)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
