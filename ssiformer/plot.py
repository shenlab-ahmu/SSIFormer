"""Region plots of SSIFormer predictions: predicted silencer-silencer interactions as arcs, with optional reference arcs
(for example HiChIP SSIs) and signal tracks.

  python -m ssiformer.plot samples/<name>/scores.tsv --region chr8:120,000,000-122,000,000 \
      --top 30 --reference hichip_ssi --bigwig H3K27me3=raw/bigwig/K562/K562_H3K27me3.bigWig --out region.pdf

Input: the score table of `python -m ssiformer.sample <sample.yaml> score` (columns chrom, a, b, logit, probability, ...),
or any table with chrom, a, b (10-kb bin indices) and a score column. Arcs above the axis: the pairs with the highest scores
inside the region (--top) or all pairs with probability >= --min-probability. Arcs below the axis: rows with a non-zero
value in the --reference column. Requires matplotlib; pyBigWig for --bigwig."""
import argparse
import re

import numpy as np
import pandas as pd

RES = 10_000


def parse_region(s):
    m = re.fullmatch(r'(\w+):([\d,]+)-([\d,]+)', s.strip())
    if not m:
        raise ValueError(f'region must look like chr8:120000000-122000000, got {s!r}')
    c, a, b = m.group(1), int(m.group(2).replace(',', '')), int(m.group(3).replace(',', ''))
    if b <= a:
        raise ValueError('region end must be larger than its start')
    return c, a, b


def select(T, region, score='logit', top=None, min_probability=None):
    """pairs with both anchors inside the region, ranked by score."""
    c, s, e = region
    m = (T.chrom == c) & (T.a * RES >= s) & ((T.b + 1) * RES <= e)
    R = T[m].copy()
    if min_probability is not None and 'probability' in R:
        R = R[R.probability >= min_probability]
    R = R.sort_values(score, ascending=False)
    return R.head(top) if top else R


def _arcs(ax, P, sign, color, alpha_from=None, lw=1.2):
    for _, r in P.iterrows():
        x0, x1 = (r.a + 0.5) * RES / 1e6, (r.b + 0.5) * RES / 1e6
        t = np.linspace(0, np.pi, 60)
        xs = (x0 + x1) / 2 + (x1 - x0) / 2 * np.cos(t)
        ys = sign * (x1 - x0) / 2 * np.sin(t)
        a = 0.85 if alpha_from is None else float(np.clip(0.25 + 0.75 * r[alpha_from], 0.25, 1.0))
        ax.plot(xs, ys, color=color, lw=lw, alpha=a, solid_capstyle='round')


def plot(T, region, out, score='logit', top=30, min_probability=None, reference=None, bigwigs=(), title=None,
         colors=('#c0392b', '#2c3e50', '#7f8c8d')):
    """draw the region figure and save it to `out` (format from the file extension)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c, s, e = region
    P = select(T, region, score, top if min_probability is None else None, min_probability)
    Rf = select(T[T[reference] > 0], region, score) if reference else None
    n_tr = len(bigwigs)
    fig = plt.figure(figsize=(7.2, 2.6 + 0.55 * n_tr))
    gs = fig.add_gridspec(1 + n_tr, 1, height_ratios=[4] + [1] * n_tr, hspace=0.08)
    ax = fig.add_subplot(gs[0])
    if 'probability' in P:
        _arcs(ax, P, +1, colors[0], alpha_from='probability')
    else:
        _arcs(ax, P, +1, colors[0])
    if Rf is not None and len(Rf):
        _arcs(ax, Rf, -1, colors[1])
    anchors = np.unique(np.r_[T[(T.chrom == c)].a.values, T[(T.chrom == c)].b.values])
    anchors = anchors[(anchors * RES >= s) & ((anchors + 1) * RES <= e)]
    ax.plot((anchors + 0.5) * RES / 1e6, np.zeros(len(anchors)), '|', color=colors[2], ms=6, mew=0.8)
    span = (e - s) / 1e6
    ax.set_xlim(s / 1e6, e / 1e6)
    top_h = ((P.b - P.a).max() * RES / 2e6) if len(P) else span * 0.05
    bot_h = ((Rf.b - Rf.a).max() * RES / 2e6) if (Rf is not None and len(Rf)) else span * 0.02
    ax.set_ylim(-bot_h * 1.15, top_h * 1.25)
    ax.axhline(0, color=colors[2], lw=0.6)
    ax.set_yticks([])
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    lab = f'top {len(P)} predicted pairs' if min_probability is None else f'{len(P)} pairs with probability >= {min_probability:g}'
    ax.text(0.01, 0.97, lab, transform=ax.transAxes, va='top', fontsize=8, color=colors[0])
    if Rf is not None and len(Rf):
        ax.text(0.01, 0.03, f'{len(Rf)} reference pairs ({reference})', transform=ax.transAxes, va='bottom', fontsize=8, color=colors[1])
    if title:
        ax.set_title(title, fontsize=9, loc='left')
    for k, (name, path) in enumerate(bigwigs):
        import pyBigWig
        axt = fig.add_subplot(gs[1 + k], sharex=ax)
        bw = pyBigWig.open(path)
        nb = max(1, (e - s) // 1000)
        v = np.array(bw.stats(c, s, s + nb * 1000, nBins=nb, type='mean'), dtype=float)
        x = (s + (np.arange(nb) + 0.5) * 1000) / 1e6
        axt.fill_between(x, np.nan_to_num(v), color=colors[2], lw=0)
        axt.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=7)
        axt.set_yticks([])
        for sp in ('top', 'right', 'left'):
            axt.spines[sp].set_visible(False)
        if k < n_tr - 1:
            plt.setp(axt.get_xticklabels(), visible=False)
    if n_tr:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(axis='x', length=0)
        ax.spines['bottom'].set_visible(False)
    fig.axes[-1].set_xlabel(f'{c} (Mb)', fontsize=8)
    for a_ in fig.axes:
        a_.tick_params(labelsize=7)
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return len(P), (0 if Rf is None else len(Rf))


def main():
    ap = argparse.ArgumentParser(description='Region plot of SSIFormer predictions')
    ap.add_argument('scores', help='score table (python -m ssiformer.sample ... score), or a table with chrom, a, b and a score column')
    ap.add_argument('--region', required=True, help='chrom:start-end, e.g. chr8:120000000-122000000')
    ap.add_argument('--score', default='logit', help='score column (default logit)')
    ap.add_argument('--top', type=int, default=30, help='number of highest-scoring pairs drawn (default 30)')
    ap.add_argument('--min-probability', type=float, help='draw all pairs with probability >= this value instead of --top')
    ap.add_argument('--reference', help='column whose non-zero rows are drawn below the axis (e.g. hichip_ssi)')
    ap.add_argument('--bigwig', action='append', default=[], help='name=path of a signal track (repeatable)')
    ap.add_argument('--title')
    ap.add_argument('--out', required=True, help='output figure (.pdf, .png or .svg)')
    a = ap.parse_args()
    T = pd.read_csv(a.scores, sep='\t')
    for col in ('chrom', 'a', 'b', a.score):
        if col not in T:
            raise SystemExit(f'column {col!r} missing in {a.scores}')
    bws = [tuple(x.split('=', 1)) for x in a.bigwig]
    n, nr = plot(T, parse_region(a.region), a.out, a.score, a.top, a.min_probability, a.reference, bws, a.title)
    print(f'{a.out}: {n} predicted pairs' + (f', {nr} reference pairs' if a.reference else ''))


if __name__ == '__main__':
    main()
