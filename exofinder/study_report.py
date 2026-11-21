"""Generate a static scientific figure and Markdown report from saved study results."""
import json
from pathlib import Path
import numpy as np
from .data import ROOT
from .multiquarter import DEFAULT_DATASET
from .research import RESEARCH, MODES


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    study=json.loads((RESEARCH/'summary.json').read_text())
    injection=json.loads((ROOT/'runs/injection-recovery/summary.json').read_text())
    dataset=json.loads((DEFAULT_DATASET/'manifest.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    labels=['Dual','Global','Local','Mask only']
    means=[study['modes'][mode]['average_precision']['mean'] for mode in MODES]
    std=[study['modes'][mode]['average_precision']['std'] for mode in MODES]
    axes[0].bar(labels,means,yerr=std,capsize=4,color=['#357b78','#779a99','#a2b9b8','#c6d6d5'])
    axes[0].set(ylim=(0,1.05),ylabel='Held-out average precision',title='Input ablations: mean ± seed SD')
    ensemble=study['ensemble']
    for key,label,color in [('raw_test','Raw','#c88e40'),('calibrated_test','Calibrated','#357b78')]:
        bins=ensemble[key]['reliability']
        axes[1].plot([b['mean_score'] for b in bins],[b['candidate_fraction'] for b in bins],'o-',label=label,color=color)
    axes[1].plot([0,1],[0,1],'--',color='.6',linewidth=1)
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel='Mean score',ylabel='Observed candidate fraction',title='Independent ensemble calibration')
    axes[1].legend(frameon=False)
    periods=injection['plan']['periods_days'];depths=injection['plan']['depths_ppm']
    grid=np.array([[next(c['mean'] for c in injection['cells'] if c['injected_period']==p and c['injected_depth_ppm']==d) for d in depths] for p in periods])
    im=axes[2].imshow(grid,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    axes[2].set(xticks=np.arange(len(depths)),xticklabels=[int(d) for d in depths],yticks=np.arange(len(periods)),yticklabels=periods,
                xlabel='Injected depth (ppm)',ylabel='Injected period (days)',title='Strict BLS recovery fraction')
    for i in range(len(periods)):
        for j in range(len(depths)):axes[2].text(j,i,f'{grid[i,j]:.0%}',ha='center',va='center',color='white' if grid[i,j]>.5 else 'black')
    fig.colorbar(im,ax=axes[2],shrink=.8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    dest=ROOT/'docs/figures';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'research-study.png',dpi=180);fig.savefig(dest/'research-study.pdf');plt.close(fig)
    lines=['# Multi-quarter study results','',f"{dataset['usable']:,} usable stars with Q9 and Q10, from {dataset['selected']:,} requested. Every pilot star excluded.",'',
           '![Ablation, calibration and BLS recovery figure](figures/research-study.png)','',
           '## Cohort and partitions','',f"Usable catalog counts: {dataset['usable_counts']}. Selected counts: {dataset['selection_counts']}.",'',
           f"Partitions: {study['counts']}. Device: {study['device']}. All seeds share these star-disjoint partitions.",'',
           '## Three-seed input ablations','', '| Mode | AP, mean ± seed SD | ROC AUC, mean ± seed SD | F1, mean ± seed SD |','|---|---:|---:|---:|']
    for mode in MODES:
        row=study['modes'][mode]
        lines.append('| '+mode+' | '+' | '.join(f"{row[k]['mean']:.4f} ± {row[k]['std']:.4f}" for k in ['average_precision','roc_auc','f1'])+' |')
    lines += ['', f"Same-cohort logistic baseline: test AP {study['baseline']['raw_test']['average_precision']:.4f}, ROC AUC {study['baseline']['raw_test']['roc_auc']:.4f}. Its scaler and classifier fit training stars only.", '', 'Every seed result and model checkpoint is stored under `runs/research-q9-q10/`. Sample SD measures initialization variation, not population uncertainty. All modes retain identical network capacity.','',
              '## Ensemble calibration','', '| Score version | Test Brier | Test log loss | Test AP |','|---|---:|---:|---:|']
    for key,label in [('raw_test','Raw'),('calibrated_test','Calibrated')]:
        r=ensemble[key];lines.append(f"| {label} | {r['brier']:.4f} | {r['log_loss']:.4f} | {r['average_precision']:.4f} |")
    lines += ['',f"Calibrator fitted on {ensemble['calibrator']['fit_count']} separate calibration stars, candidate fraction {ensemble['calibrator']['fit_prevalence']:.3f}. The ensemble averages all three predeclared dual-view seeds; no best seed is chosen.",'',
              'Reliability refers to DR25 candidate labels in the eligible, available-quarter KOI cohort. Availability filtering and the period range restrict representativeness. These are not probabilities of physical planet confirmation.','',
              '## BLS injection/recovery','', f"Strict recoveries: **{injection['strict_recovered']}/{injection['trials']}** ({injection['strict_recovery_fraction']:.1%}); harmonic matches: {injection['harmonic_matches']}.",'',
              f"Uninjected backgrounds above depth SNR 7.1: **{injection['null_backgrounds_above_snr_7_1']}/{injection['plan']['stars']}**. This threshold is exploratory and does not establish a global false-alarm probability.",'',
              '| Period (days) | Depth (ppm) | Strict recoveries | Fraction |','|---:|---:|---:|---:|']
    for cell in injection['cells']:lines.append(f"| {cell['injected_period']:g} | {cell['injected_depth_ppm']:g} | {cell['sum']}/{cell['count']} | {cell['mean']:.1%} |")
    lines += ['', 'Box injections on eight actual KOI backgrounds are a small detection experiment, not a population completeness measurement. No combined BLS+CNN discovery-performance claim is made.','',
              '## Reproduce and inspect','', '- [Study design, isolation rules and commands](RESEARCH_STUDY.md)',
              '- [Data manifest](../data/research-q9-q10/manifest.json)', '- [Predeclared training plan](../runs/research-q9-q10/plan.json)',
              '- [Study summary](../runs/research-q9-q10/summary.json)', '- [Ensemble predictions](../runs/research-q9-q10/ensemble-predictions.csv)',
              '- [Every BLS trial](../runs/injection-recovery/trials.csv)', '- [Uninjected controls](../runs/injection-recovery/nulls.csv)',
              '- [Exportable figure (PDF)](figures/research-study.pdf)', '',
              'After reviewing these held-out results, model changes guided by these errors require a new untouched test set. The original pilot remains separately available in the app.']
    (ROOT/'docs/RESEARCH_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(ROOT/'docs/RESEARCH_RESULTS.md')


if __name__=='__main__':main()
