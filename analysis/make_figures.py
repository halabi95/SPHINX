"""
make_figures.py — Generate manuscript figures from results/.

Produces:
  - Figure 4: 4-panel sensitivity sweep (u, v, p errors vs interface fraction f)
  - Figures 5/7/9 + per-case N-scaling: per-case N-scaling at f = 0.1 (u, v, p panels with baseline band)
  - Figure 12: SPHINX vs RBF u-velocity curves vs interface fraction
  - Figure 13: Cross-case summary bar chart (baseline vs SPHINX u-velocity)

Outputs SVG (vector) and PNG (preview) into figures/ alongside this script.

Run from the repository root:
    python analysis/make_figures.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, 'results')
OUT = os.path.join(REPO_ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)

CASES = ['cylinder', 'naca_re1k', 'naca_re10k', 'fsi1']
CASE_TITLES = {
    'cylinder':   r'Cylinder Re = 40',
    'naca_re1k':  r'NACA 0012 Re = 1,000, $\alpha=5^\circ$',
    'naca_re10k': r'NACA 0012 Re = 10,000, $\alpha=5^\circ$',
    'fsi1':       r'Turek-Hron FSI1 Re = 20',
}
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = [42, 123, 456]

# Baseline (XPINN N = 0) values, from the manuscript
BASELINES = {
    'cylinder':   {'u': (9.95, 0.30),   'v': (29.75, 0.22),  'p': (58.04, 11.20)},
    'naca_re1k':  {'u': (24.40, 0.96),  'v': (59.66, 9.78),  'p': (97.89, 4.66)},
    'naca_re10k': {'u': (24.12, 0.28),  'v': (58.36, 2.12),  'p': (196.48, 74.41)},
    'fsi1':       {'u': (49.86, 7.72),  'v': (44.75, 3.45),  'p': (94.22, 19.99)},
}

# Plot style
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.4,
    'lines.markersize': 5,
})

COLOR_U = '#1f4e79'
COLOR_V = '#d18a2e'
COLOR_P = '#2e7d32'
COLOR_BASELINE_LINE = '#666666'
COLOR_BASELINE_FILL = '#cccccc'


# -------------------------------------------------------------------- #
# Data loaders
# -------------------------------------------------------------------- #
def load_sweep_case(case):
    """Returns {frac: (u_mean, u_std, v_mean, v_std, p_mean, p_std)}."""
    root = os.path.join(RESULTS, 'sensitivity_sweep', case)
    out = {}
    for f in FRACS:
        trials = []
        for s in SEEDS:
            p = os.path.join(root, f'frac_{f}', f'seed_{s}', 'metrics.npz')
            if not os.path.exists(p):
                continue
            try:
                d = dict(np.load(p, allow_pickle=True))
                trials.append((float(d['u_err']), float(d['v_err']), float(d['p_err'])))
            except Exception:
                continue
        if trials:
            arr = np.array(trials)
            mu, sd = arr.mean(axis=0), arr.std(axis=0)
            out[f] = (mu[0], sd[0], mu[1], sd[1], mu[2], sd[2])
    return out


def load_nscaling_case(case):
    """Returns {N: (u_m, u_s, v_m, v_s, p_m, p_s)}."""
    root = os.path.join(RESULTS, 'n_scaling_f01', case)
    out = {}
    if not os.path.isdir(root):
        # Try to add N=200 from main sweep
        sweep = load_sweep_case(case)
        if 0.1 in sweep:
            out[200] = sweep[0.1]
        return out
    for n_sub in sorted(os.listdir(root)):
        if not n_sub.startswith('N'):
            continue
        try:
            N = int(n_sub[1:])
        except ValueError:
            continue
        trials = []
        for s in SEEDS:
            p = os.path.join(root, n_sub, f'seed_{s}', 'metrics.npz')
            if not os.path.exists(p):
                continue
            try:
                d = dict(np.load(p, allow_pickle=True))
                trials.append((float(d['u_err']), float(d['v_err']), float(d['p_err'])))
            except Exception:
                continue
        if trials:
            arr = np.array(trials)
            mu, sd = arr.mean(axis=0), arr.std(axis=0)
            out[N] = (mu[0], sd[0], mu[1], sd[1], mu[2], sd[2])

    # Always include N=200 from the main sweep at f=0.1
    sweep = load_sweep_case(case)
    if 0.1 in sweep and 200 not in out:
        out[200] = sweep[0.1]
    return out


def load_rbf_case(case):
    """Returns {frac: u_err}. RBF is deterministic across seeds, use any seed."""
    root = os.path.join(RESULTS, 'rbf_baseline', case)
    out = {}
    if not os.path.isdir(root):
        return out
    for f in FRACS:
        for s in SEEDS:
            p = os.path.join(root, f'frac_{f}', f'seed_{s}', 'metrics.npz')
            if os.path.exists(p):
                try:
                    d = dict(np.load(p, allow_pickle=True))
                    out[f] = float(d['u_err'])
                    break
                except Exception:
                    continue
    return out


# -------------------------------------------------------------------- #
# Figure 4: sensitivity sweep 4-panel
# -------------------------------------------------------------------- #
def fig_sensitivity():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    axes = axes.flatten()

    for ax, case in zip(axes, CASES):
        data = load_sweep_case(case)
        if not data:
            ax.text(0.5, 0.5, f'No data for {case}', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(CASE_TITLES[case])
            continue
        fs = sorted(data.keys())
        u = [data[f][0] for f in fs]; us = [data[f][1] for f in fs]
        v = [data[f][2] for f in fs]; vs = [data[f][3] for f in fs]
        p = [data[f][4] for f in fs]; ps = [data[f][5] for f in fs]

        ax.errorbar(fs, u, yerr=us, color=COLOR_U, marker='o',
                    label=r'$u$-velocity', capsize=2.5)
        ax.errorbar(fs, v, yerr=vs, color=COLOR_V, marker='s',
                    label=r'$v$-velocity', capsize=2.5)
        ax.errorbar(fs, p, yerr=ps, color=COLOR_P, marker='^',
                    label='pressure', capsize=2.5)

        ax.set_xlabel(r'Interface allocation fraction $f$')
        ax.set_ylabel(r'Relative $L_2$ Error (%)')
        ax.set_title(CASE_TITLES[case])
        ax.set_xticks(FRACS)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_ylim(bottom=0)

    axes[0].legend(loc='upper left', framealpha=0.9)
    fig.suptitle('Sensitivity of Reconstruction Error to Interface Allocation',
                 fontsize=13, y=1.02)
    out_svg = os.path.join(OUT, 'fig04_sensitivity_sweep.svg')
    out_png = out_svg.replace('.svg', '.png')
    fig.savefig(out_svg, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  wrote {out_svg}')


# -------------------------------------------------------------------- #
# Per-case N-scaling (Figures 5, 7, 9, and equivalent for FSI)
# -------------------------------------------------------------------- #
def fig_nscaling_per_case():
    for case in CASES:
        data = load_nscaling_case(case)
        if not data:
            continue
        Ns = sorted(data.keys())

        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
        for ax, (var, label, mi, si, color) in zip(axes, [
            ('u', r'$u$-velocity', 0, 1, COLOR_U),
            ('v', r'$v$-velocity', 2, 3, COLOR_V),
            ('p', r'Pressure $p$', 4, 5, COLOR_P),
        ]):
            means = [data[N][mi] for N in Ns]
            stds = [data[N][si] for N in Ns]
            b_mean, b_std = BASELINES[case][var]

            ax.axhspan(max(0, b_mean - b_std), b_mean + b_std,
                       color=COLOR_BASELINE_FILL, alpha=0.5, zorder=0,
                       label='XPINN baseline ±1σ')
            ax.axhline(b_mean, color=COLOR_BASELINE_LINE, linestyle='--',
                       linewidth=1.0, zorder=1, label='XPINN baseline (N=0)')

            ax.errorbar(Ns, means, yerr=stds, color=color, marker='s',
                        capsize=2.5, label=r'SPHINX ($f=0.1$)', zorder=3,
                        markerfacecolor='white', markeredgecolor=color,
                        markeredgewidth=1.4)

            ax.set_xscale('log')
            ax.set_xlabel(r'Number of sparse data points $N$')
            ax.set_ylabel(r'Relative $L_2$ Error (%)')
            ax.set_title(label)
            ax.set_xticks(Ns)
            ax.set_xticklabels([str(N) for N in Ns])
            ax.minorticks_off()
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.set_ylim(bottom=0, top=(b_mean + b_std) * 1.05)

        axes[0].legend(loc='center right', framealpha=0.9, fontsize=8)
        fig.suptitle(f'{CASE_TITLES[case]} — SPHINX Error Scaling at $f=0.1$',
                     fontsize=12, y=1.05)

        safe = case.replace('_', '-')
        out_svg = os.path.join(OUT, f'fig_nscaling_{safe}.svg')
        out_png = out_svg.replace('.svg', '.png')
        fig.savefig(out_svg, bbox_inches='tight')
        fig.savefig(out_png, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f'  wrote {out_svg}')


# -------------------------------------------------------------------- #
# Figure 12: SPHINX vs RBF u-velocity curves
# -------------------------------------------------------------------- #
def fig_sphinx_vs_rbf():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    axes = axes.flatten()

    for ax, case in zip(axes, CASES):
        sphinx = load_sweep_case(case)
        rbf = load_rbf_case(case)
        if not sphinx or not rbf:
            ax.text(0.5, 0.5, f'Missing data for {case}', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(CASE_TITLES[case])
            continue

        fs = sorted(set(sphinx.keys()) & set(rbf.keys()))
        u_sphinx = [sphinx[f][0] for f in fs]
        u_rbf = [rbf[f] for f in fs]

        ax.plot(fs, u_rbf, color='#b30000', marker='o', label='RBF interpolation')
        ax.plot(fs, u_sphinx, color=COLOR_U, marker='s', label='SPHINX')
        ax.set_xlabel(r'Interface allocation fraction $f$')
        ax.set_ylabel(r'$u$-velocity relative $L_2$ error (%)')
        ax.set_title(CASE_TITLES[case])
        ax.set_xticks(FRACS)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_ylim(bottom=0)

    axes[0].legend(loc='upper right', framealpha=0.9)
    fig.suptitle('SPHINX vs Radial Basis Function Interpolation', fontsize=13, y=1.02)

    out_svg = os.path.join(OUT, 'fig12_sphinx_vs_rbf.svg')
    fig.savefig(out_svg, bbox_inches='tight')
    fig.savefig(out_svg.replace('.svg', '.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  wrote {out_svg}')


# -------------------------------------------------------------------- #
# Figure 13: cross-case summary
# -------------------------------------------------------------------- #
def fig_cross_case_summary():
    # Numbers come straight from the manuscript / sweep at f = 0.1, N = 200
    labels = ['Cylinder\nRe = 40', 'NACA\nRe = 1,000', 'NACA\nRe = 10,000', 'FSI1\nRe = 20']
    baseline = [9.95, 24.40, 24.12, 49.86]
    sphinx = []
    for case in CASES:
        data = load_sweep_case(case)
        if 0.1 in data:
            sphinx.append(data[0.1][0])
        else:
            sphinx.append(0.0)
    reductions = [100 * (b - s) / b if b > 0 else 0 for b, s in zip(baseline, sphinx)]

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    b1 = ax.bar(x - w/2, baseline, w, label='Baseline XPINN', color=COLOR_BASELINE_LINE)
    b2 = ax.bar(x + w/2, sphinx, w, label='SPHINX (f=0.1, N=200)', color=COLOR_U)

    for i, (xi, bv, sv, rd) in enumerate(zip(x, baseline, sphinx, reductions)):
        ax.text(xi - w/2, bv + 1.5, f'{bv:.2f}', ha='center', fontsize=9)
        ax.text(xi + w/2, sv + 1.5, f'{sv:.3f}', ha='center', fontsize=9)
        ax.text(xi, max(bv, sv) + 7, f'-{rd:.1f}%', ha='center', fontsize=11,
                color='#1e7e34', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r'$u$-velocity relative $L_2$ error (%)')
    ax.set_title('SPHINX Performance Summary — u-velocity at N = 200, f = 0.1')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.set_ylim(0, max(baseline) * 1.25)

    out_svg = os.path.join(OUT, 'fig13_cross_case_summary.svg')
    fig.savefig(out_svg, bbox_inches='tight')
    fig.savefig(out_svg.replace('.svg', '.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  wrote {out_svg}')


# -------------------------------------------------------------------- #
def main():
    print('Generating manuscript figures...')
    print(f'  output directory: {OUT}')

    if not os.path.isdir(RESULTS):
        print(f'[!] results/ directory not found at {RESULTS}')
        print('    Run the sweep scripts first, then re-run this script.')
        return

    fig_sensitivity()
    fig_nscaling_per_case()
    fig_sphinx_vs_rbf()
    fig_cross_case_summary()
    print('Done.')


if __name__ == '__main__':
    main()
