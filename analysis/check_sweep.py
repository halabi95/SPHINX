"""
check_sweep.py — Summarize SPHINX sensitivity sweep, N-scaling, RBF, and ablation results.

Walks the local results/ directory tree (as laid out in the README) and prints:
  1. Sensitivity sweep completeness (4 cases × 6 fracs × 3 seeds = 72 trials per study)
  2. Per-trial errors with seed-to-seed mean ± std
  3. N-scaling extras at f = 0.1
  4. RBF baseline summary
  5. Ablation results (standard / adaptive / RAR)

Run from the repository root:
    python analysis/check_sweep.py

Expects results to be organized as:
    results/
      sensitivity_sweep/<case>/frac_<f>/seed_<s>/metrics.npz
      n_scaling_f01/<case>/N<n>/seed_<s>/metrics.npz
      rbf_baseline/<case>/frac_<f>/seed_<s>/metrics.npz
      ablation/<mode>/seed_<s>/metrics.npz  (or sphinx_ablation_<mode>_*.npz)

Each metrics.npz contains at minimum: u_err, v_err, p_err (relative L2 errors in percent).
"""

import os
import glob
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, 'results')

CASES = ['cylinder', 'naca_re1k', 'naca_re10k', 'fsi1']
FRACS = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']
SEEDS = [42, 123, 456]


def banner(title):
    print()
    print('=' * 72)
    print(f'  {title}')
    print('=' * 72)


def load_metrics(path):
    """Return (u, v, p) errors as floats, or None on failure."""
    try:
        d = dict(np.load(path, allow_pickle=True))
        return float(d['u_err']), float(d['v_err']), float(d['p_err'])
    except Exception:
        return None


# -------------------------------------------------------------------- #
# 1. Sensitivity sweep
# -------------------------------------------------------------------- #
def report_sensitivity():
    root = os.path.join(RESULTS, 'sensitivity_sweep')
    if not os.path.isdir(root):
        return

    banner(f'Sensitivity sweep: {root}')

    print('\n--- Per-trial errors ---')
    print(f"{'case':12s} {'frac':6s} {'seed':6s} {'u_err':>8s}  {'v_err':>8s}  {'p_err':>8s}")
    print('-' * 56)

    cell = {}  # (case, frac) -> list of (u, v, p)
    for case in CASES:
        for frac in FRACS:
            for seed in SEEDS:
                p = os.path.join(root, case, f'frac_{frac}', f'seed_{seed}', 'metrics.npz')
                if not os.path.exists(p):
                    continue
                m = load_metrics(p)
                if m is None:
                    continue
                u, v, pv = m
                cell.setdefault((case, frac), []).append((u, v, pv))
                print(f"{case:12s} {frac:6s} {seed:<6d} {u:>8.3f}  {v:>8.3f}  {pv:>8.3f}")

    print('\n--- Per-(case, frac) summaries (mean ± std across seeds) ---')
    print(f"{'case':12s} {'frac':6s} {'n':>3s}  {'u':>15s}  {'v':>15s}  {'p':>15s}")
    print('-' * 76)
    for case in CASES:
        for frac in FRACS:
            trials = cell.get((case, frac), [])
            if not trials:
                continue
            arr = np.array(trials)
            mu, sd = arr.mean(axis=0), arr.std(axis=0)
            print(f"{case:12s} {frac:6s} {len(trials):>3d}  "
                  f"{mu[0]:>6.3f} ± {sd[0]:5.3f}  "
                  f"{mu[1]:>6.3f} ± {sd[1]:5.3f}  "
                  f"{mu[2]:>6.3f} ± {sd[2]:5.3f}")


# -------------------------------------------------------------------- #
# 2. N-scaling extras at f = 0.1
# -------------------------------------------------------------------- #
def report_n_scaling():
    root = os.path.join(RESULTS, 'n_scaling_f01')
    if not os.path.isdir(root):
        return

    banner(f'N-scaling extras at f = 0.1: {root}')

    for case in CASES:
        case_root = os.path.join(root, case)
        if not os.path.isdir(case_root):
            continue
        n_dirs = sorted([d for d in os.listdir(case_root) if d.startswith('N')],
                        key=lambda x: int(x[1:]))
        if not n_dirs:
            continue
        print(f'\n  {case}:')
        for n_sub in n_dirs:
            try:
                N = int(n_sub[1:])
            except ValueError:
                continue
            trials = []
            for seed in SEEDS:
                p = os.path.join(case_root, n_sub, f'seed_{seed}', 'metrics.npz')
                if not os.path.exists(p):
                    continue
                m = load_metrics(p)
                if m:
                    trials.append(m)
            if not trials:
                continue
            arr = np.array(trials)
            mu, sd = arr.mean(axis=0), arr.std(axis=0)
            print(f"    N = {N:>4d} ({len(trials)} seeds): "
                  f"u = {mu[0]:6.3f} ± {sd[0]:5.3f}%, "
                  f"v = {mu[1]:6.3f} ± {sd[1]:5.3f}%, "
                  f"p = {mu[2]:6.3f} ± {sd[2]:5.3f}%")


# -------------------------------------------------------------------- #
# 3. RBF baseline
# -------------------------------------------------------------------- #
def report_rbf():
    root = os.path.join(RESULTS, 'rbf_baseline')
    if not os.path.isdir(root):
        return

    banner(f'RBF baseline: {root}')

    print(f"\n{'case':12s} {'frac':6s} {'u_err':>8s}  {'v_err':>8s}  {'p_err':>8s}")
    print('-' * 50)
    for case in CASES:
        for frac in FRACS:
            for seed in SEEDS:
                p = os.path.join(root, case, f'frac_{frac}', f'seed_{seed}', 'metrics.npz')
                if not os.path.exists(p):
                    continue
                m = load_metrics(p)
                if m is None:
                    continue
                # RBF is deterministic across seeds, so just show seed 42
                if seed == 42:
                    u, v, pv = m
                    print(f"{case:12s} {frac:6s} {u:>8.3f}  {v:>8.3f}  {pv:>8.3f}")
                break


# -------------------------------------------------------------------- #
# 4. Ablation results
# -------------------------------------------------------------------- #
def report_ablation():
    root = os.path.join(RESULTS, 'ablation')
    if not os.path.isdir(root):
        return

    banner(f'Ablation study at f = 0.1: {root}')

    # Two layouts supported:
    #   results/ablation/<mode>/seed_<s>/metrics.npz
    #   results/ablation/sphinx_ablation_<mode>_*.npz  (aggregate npz from training script)
    modes = ['standard', 'adaptive', 'rar']

    # Layout A: per-seed metrics
    for mode in modes:
        mode_root = os.path.join(root, mode)
        if not os.path.isdir(mode_root):
            continue
        trials = []
        for seed in SEEDS:
            p = os.path.join(mode_root, f'seed_{seed}', 'metrics.npz')
            if os.path.exists(p):
                m = load_metrics(p)
                if m:
                    trials.append(m)
        if trials:
            arr = np.array(trials)
            mu, sd = arr.mean(axis=0), arr.std(axis=0)
            print(f"  {mode:<10s}: u = {mu[0]:6.4f} ± {sd[0]:6.4f}%   ({len(trials)} seeds)")

    # Layout B: aggregate npz with mode-specific keys
    for f in sorted(glob.glob(os.path.join(root, 'sphinx_ablation_*.npz'))):
        try:
            d = dict(np.load(f, allow_pickle=True))
        except Exception:
            continue
        mode = os.path.basename(f).replace('sphinx_ablation_', '').split('_')[0]
        token = mode.upper() if mode == 'rar' else mode
        # Look for mode-specific keys, e.g. SPHINX_N200_adaptive_u_mean
        for key in d.keys():
            if token.lower() in key.lower() and 'u_mean' in key.lower() \
                    and 'standard' not in key.lower():
                mean = float(d[key])
                std_key = key.replace('_mean', '_std')
                std = float(d[std_key]) if std_key in d else 0.0
                print(f"  {mode:<10s}: u = {mean:6.4f} ± {std:6.4f}%   "
                      f"(from {os.path.basename(f)})")
                break


# -------------------------------------------------------------------- #
def main():
    print('SPHINX results summary')
    print(f'Repository root: {REPO_ROOT}')
    print(f'Results directory: {RESULTS}')

    if not os.path.isdir(RESULTS):
        print(f'\n[!] results/ directory not found at {RESULTS}')
        print('    Run the sweep scripts first, then re-run this script.')
        return

    report_sensitivity()
    report_n_scaling()
    report_rbf()
    report_ablation()
    print()


if __name__ == '__main__':
    main()
