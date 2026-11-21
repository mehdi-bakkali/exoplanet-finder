"""Read-only diagnostics; never fit a model or choose a test-set threshold."""
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def operating_point(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(scores) >= threshold
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    return {"count": len(labels), "threshold": float(threshold),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "accuracy": (tp + tn) / len(labels) if len(labels) else None,
            "selected": tp + fp, "confusion_matrix": [[tn, fp], [fn, tp]]}


def reliability(labels, scores, bins=8):
    """Equal-width bins, including score 1. Empty bins are omitted, not zero-filled."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    assignment = np.minimum((scores * bins).astype(int), bins - 1)
    result = []
    for i in range(bins):
        selected = assignment == i
        n = int(selected.sum())
        if not n:
            continue
        rate = float(labels[selected].mean())
        # Wilson interval for the observed candidate fraction, not score uncertainty.
        z = 1.959963984540054
        center = (rate + z*z / (2*n)) / (1 + z*z/n)
        half = z * np.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1 + z*z/n)
        result.append({"lower": i/bins, "upper": (i+1)/bins, "count": n,
                       "mean_score": float(scores[selected].mean()), "candidate_fraction": rate,
                       "fraction_95ci": [max(0., center-half), min(1., center+half)]})
    return result


def build_diagnostics(metadata, report):
    test = metadata.loc[metadata.split == 'test'].copy()
    labels = test.label.to_numpy(dtype=int)
    result = {"run": report['run'], "split": "test", "count": len(test),
              "threshold": report['threshold'], "prevalence": float(labels.mean()), "models": {}}
    for name, column in [('cnn','score'), ('baseline','baseline_score')]:
        scores = test[column].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        result['models'][name] = {'pr': {'recall': recall.tolist(), 'precision': precision.tolist()},
                                  'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
                                  'reliability': reliability(labels, scores)}
    groups = [('0.5–3 days', (test.koi_period < 3)),
              ('3–10 days', (test.koi_period >= 3) & (test.koi_period < 10)),
              ('10–30 days', test.koi_period >= 10),
              ('Stellar-eclipse flag', test.koi_fpflag_ss == 1),
              ('No stellar-eclipse flag', test.koi_fpflag_ss != 1)]
    result['subgroups'] = [dict(name=name, **operating_point(test.loc[mask,'label'], test.loc[mask,'score'], report['threshold']))
                           for name, mask in groups]
    wrong = (test.score >= report['threshold']).astype(int) != test.label
    errors = test.loc[wrong].copy()
    errors['error_type'] = np.where(errors.label == 0, 'false_alarm', 'missed_candidate')
    errors['distance_from_threshold'] = np.abs(errors.score - report['threshold'])
    result['errors'] = errors.sort_values('distance_from_threshold', ascending=False).to_dict('records')
    result['notes'] = ['Read-only diagnostics of the original test predictions; no retraining or calibration applied.',
                       'Subgroup sizes are small; eclipse groups overlap period groups.',
                       'Reliability intervals describe catalog candidate fractions, not astrophysical planet probabilities.']
    return result


def main():
    import argparse
    import json
    from pathlib import Path
    import pandas as pd
    from .data import ROOT
    parser = argparse.ArgumentParser(description='Export diagnostics without retraining.')
    parser.add_argument('--run', default='pilot')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        parser.error('Run name must be a simple directory name.')
    run = ROOT/'runs'/args.run
    report = json.loads((run/'metrics.json').read_text())
    meta = pd.read_csv(run/'predictions.csv').fillna('')
    output = run/'diagnostics.json'
    output.write_text(json.dumps(build_diagnostics(meta, report), indent=2, allow_nan=False))
    print(output)


if __name__ == '__main__':
    main()
