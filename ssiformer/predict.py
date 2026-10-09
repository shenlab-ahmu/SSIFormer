"""Score pairs of silencer anchor bins with trained SSIFormer models.

  python -m ssiformer.predict <dataset> <model> [--extra-loops] [--heldout] [--table pairs.tsv] [--strand-exchange] [--out scores.tsv]

<dataset> and <model> are keys of configs/ssiformer.yaml (e.g. GM12878_CTCF ssiformer_pretrained, mESC_H3K27ac ssiformer).
--extra-loops selects the models trained also on the loops without silencers of the dataset's HiChIP sample ('<model>_loops'; for the human
datasets, 'ssiformer_pretrained --extra-loops' is the extended configuration).

Default: the cross-validation table of the dataset (benchmark/tables); every pair is scored by the fold model whose test fold
contains its chromosome, so the scores are out-of-fold. --heldout: the table of the held-out chromosomes (benchmark/sealed;
file and option names in the code call these chromosomes 'sealed'), scored by the mean of the five fold models. --table: any
table of pairs of the same cell type with columns chrom, a, b (10-kb bin indices, a < b) and optionally label; the pairs are
scored by the mean of the five fold models (with --heldout the inputs of the held-out chromosomes are used). The anchor
bins of the pairs must be in the anchor-tensor file of the cell type (cache/anchor_bin_<cell>_10000.npz). Pairs of a new
sample are scored with ssiformer.sample.

The score is the logit averaged over the two orientations of a pair. --strand-exchange adds the change of this logit when the
CTCF plus- and minus-strand motif channels are exchanged over the whole region (convergent <-> divergent motif pairs,
positions unchanged): a more negative value means a stronger dependence of the pair on the CTCF orientation.

Each fold model uses the input normalisation it was trained with. Trained weights are stored in fp16."""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import config as K
from . import dataset as DS
from . import model as MD
from .utils import auprc, auroc


def weights_path(name, model, k):
    return K.fold_weights(name, model, k)[0]


def fold_norm(name, model, k):
    return json.load(open(K.fold_weights(name, model, k)[1]))['norm']


def load_fold_model(name, model, k, n_marks, n_channels, dev):
    import torch
    net = MD.build(K.MODELS[model]['family'], n_marks, n_channels).to(dev)
    sd = torch.load(weights_path(name, model, k), map_location=dev)
    net.load_state_dict({kk: v.float() if v.is_floating_point() else v for kk, v in sd.items()})
    net.eval()
    return net


def predict_rows(D, net, rows, equivariant, bs=32):
    """logits of the rows; with orientation symmetrisation, the mean over both orientations of each pair."""
    import torch
    f, r = np.zeros(len(rows), np.float32), np.zeros(len(rows), np.float32)
    pos = {x: i for i, x in enumerate(rows)}
    with torch.no_grad():
        for b in DS.make_batches(rows, D.L, bs, None):
            idx = np.array([pos[x] for x in b])
            for flip, out in ((None, f),) + (((np.ones(len(b), bool), r),) if equivariant else ()):
                A, Cc, x, L, ld, _ = D.batch(b, flip)
                with torch.autocast('cuda', dtype=torch.float16):
                    out[idx] = net(A, Cc, x, L, ld).float().cpu().numpy()
    return (f + r) / 2 if equivariant else f


def strand_exchange(D, net, rows, bs=8):
    """change of the orientation-averaged logit when the CTCF strand channels are exchanged over the whole region."""
    import torch
    i_pm, i_pc, i_mm, i_mc = D.swap_idx
    out = np.zeros(len(rows), np.float32)
    pos = {x: i for i, x in enumerate(rows)}

    def two_orient(A, Cc, x, L, ld):
        f = torch.ones(len(L), dtype=torch.bool, device=D.dev)
        x2, A2, C2 = DS.apply_flip(x.clone(), L, A.clone(), Cc.clone(), f, D.swap_idx)
        return (net(A, Cc, x, L, ld) + net(A2, C2, x2, L, ld)) / 2
    with torch.no_grad():
        for b in DS.make_batches(rows, D.L, bs, None):
            A, Cc, x, L, ld, _ = D.batch(b, None)
            A, Cc, x = A.float(), Cc.float(), x.float()
            xs = x.clone()
            xs[..., [i_pm, i_pc, i_mm, i_mc]] = x[..., [i_mm, i_mc, i_pm, i_pc]]
            out[[pos[r] for r in b]] = (two_orient(A, Cc, xs, L, ld) - two_orient(A, Cc, x, L, ld)).cpu().numpy()
    return out


def read_pairs(path):
    """a table of pairs (chrom, a, b[, label]) -> the columns the model input needs."""
    T = pd.read_csv(path, sep='\t')
    assert {'chrom', 'a', 'b'} <= set(T.columns), 'the table needs the columns chrom, a, b'
    assert (T.a < T.b).all(), 'a < b is required'
    T = T.reset_index(drop=True)
    T['d'] = (T.b - T.a) * K.RES
    if 'label' not in T.columns:
        T['label'] = -1
    T['row'] = np.arange(len(T))
    return T


def gpu():
    import torch
    frac = float(os.environ.get('SSIFORMER_GPU_FRACTION', '0.8'))   # an allocation beyond the device memory fails instead of
    if torch.cuda.is_available() and frac > 0:                       # spilling into shared system memory
        torch.cuda.set_per_process_memory_fraction(frac, 0)
    return torch.device('cuda:0')


def fold_scores(D, name, model, fold=None, swap=False):
    """scores of the rows of D with the five fold models of <name>/<model>, each with its own input normalisation.
    fold (array of the rows' fold labels): every row is scored by the model of its test fold (out-of-fold); None: mean of the
    five fold models. Returns (logit, strand-exchange change or None, logits of the single fold models (5, n))."""
    import torch
    n, eq = len(D.T), K.MODELS[model]['equivariant']
    logit, dsw = np.zeros(n, np.float64), np.zeros(n, np.float64)
    per = np.full((5, n), np.nan, np.float32)
    for k in range(5):
        rows = np.where(fold == str(k))[0] if fold is not None else np.arange(n)
        if not len(rows):
            continue
        D.set_norm(fold_norm(name, model, k))
        D.to_device()
        net = load_fold_model(name, model, k, len(D.marks), len(D.channels), D.dev)
        w = 1.0 if fold is not None else 0.2
        per[k, rows] = predict_rows(D, net, rows, eq)
        logit[rows] += w * per[k, rows]
        if swap:
            dsw[rows] += w * strand_exchange(D, net, rows)
        del net
        torch.cuda.empty_cache()
    return logit, (dsw if swap else None), per


def score(name, model, heldout=False, table=None, swap=False):
    dev = gpu()
    T = read_pairs(table) if table else None
    D = DS.SpanData(name, dev, table=T, sealed=heldout)
    oof = table is None and not heldout
    fold = D.T.fold.astype(str).values if oof else None
    logit, dsw, _ = fold_scores(D, name, model, fold, swap)
    R = D.T[['chrom', 'a', 'b']].copy()
    R['start_a'], R['start_b'] = R.a * K.RES, R.b * K.RES
    if (D.T.label >= 0).all():
        R['label'] = D.T.label.values.astype(int)
    if oof:
        R['fold'] = fold
    R['logit'] = logit
    R['probability'] = 1 / (1 + np.exp(-logit))
    if swap:
        R['strand_exchange_change'] = dsw
    return R


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('model')
    ap.add_argument('--extra-loops', action='store_true', help="models trained also on the loops without silencers of the sample ('<model>_loops')")
    ap.add_argument('--heldout', action='store_true')
    ap.add_argument('--table')
    ap.add_argument('--strand-exchange', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    model = K.model_key(a.model, a.extra_loops)
    R = score(a.dataset, model, a.heldout, a.table, a.strand_exchange)
    tag = 'pairs' if a.table else ('heldout' if a.heldout else 'cv')
    out = a.out or os.path.join(K.RUNS, 'scores', f'{a.dataset}__{model}__{tag}.tsv')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    R.to_csv(out, sep='\t', index=False, float_format='%.6f')
    msg = f'{a.dataset} {model} {tag}: {len(R)} pairs -> {out}'
    if 'label' in R.columns and R.label.nunique() == 2:
        msg += f' | AUROC {auroc(R.label.values, R.logit.values):.4f} AUPRC {auprc(R.label.values, R.logit.values):.4f}'
    print(msg)
