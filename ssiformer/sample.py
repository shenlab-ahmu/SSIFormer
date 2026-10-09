"""SSIFormer on a new sample: candidate pairs, model inputs, scores and, with HiChIP loops, training on the sample.

  python -m ssiformer.sample <sample.yaml> prepare                  # silencer loci, candidate pairs; with loops: training tables
  python -m ssiformer.sample <sample.yaml> inputs [procs]           # model inputs from the bigWig files and the genome (CPU)
  python -m ssiformer.sample <sample.yaml> score [--model <dataset>/<model>] [--strand-exchange] [--chroms c1,c2] [--per-fold]
  python -m ssiformer.sample <sample.yaml> train [--from-scratch] [--folds 0,1,2,3,4]

The sample file (template: configs/sample_example.yaml) names the inputs; relative paths are taken from the current folder.
  name        name of the sample; outputs go to samples/<name>/ of the repository (or to 'out')
  genome      hg38 or mm10
  silencers   BED file of silencer elements (chrom, start, end; further columns are ignored; .gz allowed)
  bigwig      {mark: bigWig file} for the histone marks of the model: the marks of configs/ssiformer.yaml except the one the model
              does not use (H3K27ac for the human models and for training from the mouse pretraining)
  loops       optional: HiChIP loops of the sample at 10 kb (Loop Catalog longrange file, or BEDPE: chrom1 start1 end1 chrom2
              start2 end2; anchors are assigned to the 10-kb bin of their start)
  model       model used by 'score': <dataset>/<model> of configs/ssiformer.yaml (e.g. K562_H3K27ac/ssiformer_pretrained),
              or <sample name>/<model> after 'train' (ssiformer_pretrained_loops, or ssiformer_loops with --from-scratch)
  optional    candidates: hichip | silencers; chromosomes: [list] (restricts every step); exclusion: [loop files of other
              samples of the cell, or {path, resolution}] excluded from the negatives of the training tables; loops_resolution
              (default 10000); design (negative design of the training tables, default perpos_eps001); fasta (genome
              sequence, only needed when cache/genome_<genome>_1kb.npz of the repository is missing); out

prepare: silencer elements are merged into loci (overlap >= 1 bp; locus = midpoint). Candidate pairs are pairs of 10-kb bins of
one chromosome with b - a >= 2 and 20 kb <= distance <= 2 Mb whose two bins are candidate anchors: the bins that anchor HiChIP
SSIs (loops with a silencer-locus midpoint in both anchor bins; 'candidates: hichip', the default with loops and the setting of
all evaluations of the paper) or every bin with a silencer-locus midpoint ('candidates: silencers', the default without loops).
Column hichip_ssi marks the HiChIP SSIs. With loops, the training tables are built by the rules of ssiformer.build: the SSI
table (SSIs of the cross-validation chromosomes with 1:1 negatives from pairs of their anchor bins, not near a loop of the sample
or of the exclusion files) and the table of loops without silencers; folds and held-out chromosomes of benchmark/folds_<genome>.tsv.
inputs: 1-kb tracks of the marks and of the silencer-locus midpoints, motif channels (from the genome inputs of the repository,
else computed from the fasta), anchor tensors of every bin of the candidates and tables.
score: each candidate pair is scored by the mean of the five fold models, each with the input normalisation it was trained
with, averaged over both orientations of the pair. Output samples/<name>/scores.tsv: chrom, a, b (10-kb bin indices), start_a,
start_b, distance, hichip_ssi, logit, probability, percentile_a and percentile_b (percentage of the candidate pairs of anchor
bin a, resp. b, scored at or below the pair: its rank among the partners of each anchor), with --strand-exchange
strand_exchange_change (change of the logit when the CTCF plus- and minus-strand motif channels are exchanged; more negative =
stronger dependence on the CTCF orientation), with --per-fold the logit of each fold model.
train: the extended configuration on the tables of the sample (mouse pretraining, then fine-tuning on the SSIs and the other
loops; --from-scratch without pretraining), five folds, early stopping and test on the SSI table; runs in
samples/<name>/runs."""
import argparse
import gzip
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml

from . import build as BU
from . import config as K
from .utils import auroc, rng_of

MIN_D, MAX_D = 20_000, 2_000_000


def load(path):
    S = yaml.safe_load(open(path))
    for k in ('name', 'genome', 'silencers', 'bigwig'):
        assert k in S, f'{path}: "{k}" is missing'
    assert S['genome'] in K.ALL_CHROMS, f'genome must be one of {list(K.ALL_CHROMS)}'
    S['out'] = S.get('out') or os.path.join(K.REPO, 'samples', S['name'])
    keep = set(S.get('chromosomes') or K.ALL_CHROMS[S['genome']])
    S['chromosomes'] = [c for c in K.ALL_CHROMS[S['genome']] if c in keep]
    S.setdefault('candidates', 'hichip' if S.get('loops') else 'silencers')
    assert S['candidates'] in ('hichip', 'silencers') and (S['candidates'] == 'silencers' or S.get('loops')), S['candidates']
    S.setdefault('design', 'perpos_eps001')
    S.setdefault('drop', next((m for m in K.MARKS if m not in S['bigwig']), 'H3K27ac'))
    return S


def path_of(S, *parts):
    return os.path.join(S['out'], *parts)


def register(S, model_dataset=None):
    """register the sample as a dataset; with the marks of the model's dataset when it scores with a model of another dataset."""
    drop = K.dataset(model_dataset)['drop'] if model_dataset and model_dataset != S['name'] else S['drop']
    K.register_sample(S['name'], S['out'], S['genome'], drop)
    missing = [m for m in K.dataset(S['name'])['marks'] if m not in S['bigwig']]
    assert not missing, f'bigWig files missing for {missing}'


# ------------------------------------------------------------------ prepare
def read_bed(path, chroms):
    rows = []
    with (gzip.open if path.endswith('.gz') else open)(path, 'rt') as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 3 and p[0] in chroms and p[1].isdigit() and p[2].isdigit():
                rows.append((p[0], int(p[1]), int(p[2])))
    return pd.DataFrame(rows, columns=['chrom', 'start', 'end'])


def read_loops(path, chroms, res=K.RES):
    """Loop Catalog longrange or BEDPE -> bin pairs (chrom, a, b) at resolution res, a <= b, intra-chromosomal, no duplicates."""
    rows = []
    with (gzip.open if path.endswith('.gz') else open)(path, 'rt') as f:
        for ln in f:
            p = ln.rstrip('\n').split('\t')
            m = BU.LOOP_RE.match(p[3]) if len(p) >= 4 else None
            if m:
                c, s, c2, s2 = p[0], p[1], m.group(1), m.group(2)
            elif len(p) >= 6:
                c, s, c2, s2 = p[0], p[1], p[3], p[4]
            else:
                continue
            c, c2 = (x[3:] if x.startswith('chrchr') else x for x in (c, c2))
            if c != c2 or c not in chroms or not (s.isdigit() and s2.isdigit()):
                continue
            rows.append((c, min(int(s), int(s2)), max(int(s), int(s2))))
    return BU.loops_to_bins(pd.DataFrame(rows, columns=['chrom', 's1', 's2']), res)


def exclusion(S):
    """pair keys near a loop of the sample or of the exclusion files (negatives of the training tables)."""
    g, chroms = S['genome'], set(S['chromosomes'])
    files = [(S['loops'], int(S.get('loops_resolution', K.RES)))]
    for e in S.get('exclusion') or []:
        files.append((e['path'], int(e.get('resolution', K.RES))) if isinstance(e, dict) else (e, K.RES))
    return np.unique(np.concatenate([BU.near_keys(read_loops(p, chroms, r), r, K.RES, g) for p, r in files]))


def prepare(S):
    g, res, chroms = S['genome'], K.RES, S['chromosomes']
    os.makedirs(path_of(S, 'tables'), exist_ok=True)
    E = read_bed(S['silencers'], set(chroms))
    E['method'] = 'bed'
    L = BU.merge_loci(E.sort_values(['chrom', 'start', 'end']).reset_index(drop=True))
    L.to_csv(path_of(S, 'loci.tsv.gz'), sep='\t', index=False)
    msg = [f'{len(E)} silencer elements -> {len(L)} loci']
    F = pd.read_csv(os.path.join(K.BENCH, f'folds_{g}.tsv'), sep='\t', dtype=str)
    fmap = dict(zip(F.chrom, F.fold))
    if S.get('loops'):
        loops = read_loops(S['loops'], set(chroms))
        Pall = BU.positives_of(loops, L, fmap, g, S['name'])
        msg.append(f'{len(loops)} loops, {len(Pall)} HiChIP SSIs')
    if S['candidates'] == 'hichip':
        bins = set(zip(Pall.chrom, Pall.a)) | set(zip(Pall.chrom, Pall.b))
    else:
        bins = set(zip(L.chrom, (L.mid // res).astype(np.int64)))
    by_chrom = {}
    for c, b in bins:
        by_chrom.setdefault(c, []).append(b)
    C = BU.enumerate_pairs({c: by_chrom[c] for c in chroms if c in by_chrom}, res, MIN_D, MAX_D)
    C = C[['chrom', 'a', 'b', 'd']].reset_index(drop=True)
    C['hichip_ssi'] = np.isin(BU.keys_of(C, g), BU.keys_of(Pall, g)).astype(int) if S.get('loops') else -1
    C.to_csv(path_of(S, 'candidates.tsv.gz'), sep='\t', index=False)
    msg.append(f'{len(bins)} candidate anchor bins ({S["candidates"]}) -> {len(C)} candidate pairs')
    if S.get('loops'):
        design = S['design']
        excl = exclusion(S)
        P = Pall[~Pall.sealed].reset_index(drop=True)
        dmin, dmax = max(BU.BC['min_distance'], int(P.d.min())), int(P.d.max())
        T, ret = BU.build_table(P, BU.candidates(P, Pall, res, dmin, dmax, g, excl), res, design,
                                rng_of('sample', S['name'], f'ssi_table_{design}'))
        T['fold'] = T.chrom.map(fmap).astype(str)
        X = BU.other_loops(loops, L, fmap, g, S['name'])
        T2, info = BU.loops_table(X, len(P), design, excl, g, BU.loops_draw_seed(S['name']), rng_of('sample', S['name'], f'loop_table_{design}'))
        T2['fold'] = T2.chrom.map(fmap).astype(str)
        for t, f in ((T, 'ssi'), (T2, 'loops')):
            t.to_csv(path_of(S, 'tables', f'{f}.tsv.gz'), sep='\t', index=False, compression={'method': 'gzip', 'mtime': 0})
        pd.DataFrame([dict(table='ssi', design=design, dmin=dmin, dmax=dmax, ssi_cv=len(P), positives=int(T.label.sum()),
                           negatives=int((T.label == 0).sum()), retained=round(ret, 4)),
                      dict(table='loops', positives=info['loops'], negatives=info['negatives'], **{k: v for k, v in info.items()
                                                                                                if k not in ('loops', 'negatives')})]
                     ).to_csv(path_of(S, 'tables', 'summary.tsv'), sep='\t', index=False)
        folds = sorted(set(T.fold) | set(T2.fold))
        msg.append(f'training tables: SSI table {int(T.label.sum())} + {int((T.label == 0).sum())}, table of loops without silencers {info["loops"]} + '
                   f'{info["negatives"]} (folds {",".join(folds)})')
        if len(folds) < 5:
            msg.append('WARNING: not every fold has rows; training needs all five folds')
    print(f'{S["name"]}: ' + '; '.join(msg) + f' -> {S["out"]}')


# ------------------------------------------------------------------ inputs
def _bigwig_job(args):
    mark, path, chrom, centres, n_absent = args
    got = BU.bigwig_inputs(path, chrom, centres)
    if got is None:   # chromosome absent from this bigWig -> all-zero signal
        return mark, chrom, np.zeros(n_absent, np.float16), np.zeros((len(centres), 3, 20), np.float16)
    return (mark, chrom) + got


def inputs(S, nproc=2):
    g, res, name = S['genome'], K.RES, S['name']
    cache = path_of(S, 'cache')
    os.makedirs(cache, exist_ok=True)
    tabs = [pd.read_csv(path_of(S, 'candidates.tsv.gz'), sep='\t')]
    tabs += [pd.read_csv(path_of(S, 'tables', f'{t}.tsv.gz'), sep='\t') for t in ('ssi', 'loops') if os.path.exists(path_of(S, 'tables', f'{t}.tsv.gz'))]
    bins = set()
    for T in tabs:
        bins |= set(zip(T.chrom, T.a.astype(np.int64))) | set(zip(T.chrom, T.b.astype(np.int64)))
    present = {c for c, _ in bins}
    chroms = [c for c in S['chromosomes'] if c in present]
    per_chrom = {c: np.array(sorted(b for cc, b in bins if cc == c), np.int64) for c in chroms}
    t0 = time.time()
    # motif channels: genome inputs of the repository (cross-validation and held-out chromosomes), else the fasta
    arrays, channels = {}, None
    for src in (os.path.join(K.CACHE, f'genome_{g}_1kb.npz'), os.path.join(K.CACHE, 'sealed', f'genome_{g}_1kb.npz')):
        if os.path.exists(src):
            z = np.load(src)
            channels = z['channels']
            for c in chroms:
                if f'{c}__motif' in z.files and f'{c}__motif' not in arrays:
                    for k in ('gc', 'nfrac', 'motif'):
                        arrays[f'{c}__{k}'] = z[f'{c}__{k}']
    todo = [c for c in chroms if f'{c}__motif' not in arrays]
    if todo:
        assert S.get('fasta'), f'motif channels of {todo} need the genome sequence ("fasta" in the sample file)'
        with Pool(nproc) as pool:
            for c, gcf, nf, mo in pool.imap_unordered(BU._genome_work, [(g, c, S['fasta']) for c in todo]):
                arrays[f'{c}__gc'], arrays[f'{c}__nfrac'], arrays[f'{c}__motif'] = gcf, nf, mo
        channels = np.array([f'{n}{s}_{k}' for n, s, *_ in BU._motifs() for k in ('count', 'maxrel')])
    np.savez_compressed(os.path.join(cache, f'genome_{g}_1kb.npz'), channels=channels, chroms=np.array(chroms), **arrays)
    # mark tracks and anchor tensors
    marks = [m for m in K.MARKS if m in S['bigwig']]
    jobs = [(m, S['bigwig'][m], c, per_chrom[c] * res + res // 2, len(arrays[f'{c}__motif'])) for c in chroms for m in marks]
    tracks, xbin = {}, {}
    with Pool(nproc) as pool:
        for m, c, tr, xb in pool.imap_unordered(_bigwig_job, jobs):
            tracks[(m, c)], xbin[(m, c)] = tr, xb
    L = pd.read_csv(path_of(S, 'loci.tsv.gz'), sep='\t')
    L = L.assign(kb=(L.mid // 1000).astype(np.int64))
    out = {}
    for c in chroms:
        n = max(len(tracks[(m, c)]) for m in marks)
        out[f'{c}__marks'] = np.stack([np.pad(tracks[(m, c)], (0, n - len(tracks[(m, c)]))) for m in marks], 1)
        out[f'{c}__silcnt'] = np.bincount(L.kb[L.chrom == c].values, minlength=n)[:n].astype(np.uint16)
    np.savez_compressed(os.path.join(cache, f'tracks_{name}_1kb.npz'), marks=np.array(marks), chroms=np.array(chroms), **out)
    np.savez_compressed(os.path.join(cache, f'anchor_bin_{name}_{res}.npz'), chrom=np.concatenate([[c] * len(per_chrom[c]) for c in chroms]),
                        bin=np.concatenate([per_chrom[c] for c in chroms]),
                        x=np.concatenate([np.stack([xbin[(m, c)] for m in marks], 1) for c in chroms]).astype(np.float16),
                        marks=np.array(marks), scales=np.array(['1kb@50bp', '10kb@500bp', '100kb@5kb']))
    print(f'{name}: inputs of {len(chroms)} chromosomes, {sum(len(v) for v in per_chrom.values())} anchor bins, marks {marks} -> {cache} '
          f'({time.time() - t0:.0f} s)')


# ------------------------------------------------------------------ score
def score(S, model=None, swap=False, chroms=None, per_fold=False, out=None):
    from . import dataset as DS
    from . import predict as PR
    mname, mkey = (model or S['model']).split('/')
    register(S, mname)
    T = pd.read_csv(path_of(S, 'candidates.tsv.gz'), sep='\t')
    if chroms:
        T = T[T.chrom.isin(chroms)]
    T = T.reset_index(drop=True)
    T['label'], T['row'] = -1, np.arange(len(T))
    D = DS.SpanData(S['name'], PR.gpu(), table=T)
    t0 = time.time()
    logit, dsw, per = PR.fold_scores(D, mname, mkey, None, swap)
    secs = time.time() - t0
    R = pd.DataFrame({'chrom': T.chrom, 'a': T.a, 'b': T.b, 'start_a': T.a * K.RES, 'start_b': T.b * K.RES, 'distance': T.d,
                      'hichip_ssi': T.hichip_ssi, 'logit': logit, 'probability': 1 / (1 + np.exp(-logit))})
    n = len(R)
    A = pd.DataFrame({'chrom': np.concatenate([R.chrom, R.chrom]), 'anchor': np.concatenate([R.a, R.b]), 'logit': np.concatenate([logit, logit])})
    pct = (A.groupby(['chrom', 'anchor']).logit.rank(method='max', pct=True) * 100).values
    R['percentile_a'], R['percentile_b'] = pct[:n], pct[n:]
    if swap:
        R['strand_exchange_change'] = dsw
    if per_fold:
        for k in range(5):
            R[f'logit_fold{k}'] = per[k]
    out = out or path_of(S, 'scores.tsv')
    R.to_csv(out, sep='\t', index=False, float_format='%.6f')
    msg = f'{S["name"]}: {n} candidate pairs scored with {mname}/{mkey} (five fold models) in {secs:.0f} s'
    if not swap:
        msg += f' ({5 * n / max(secs, 1e-9):.0f} pairs per second per fold model)'
    if (R.hichip_ssi >= 0).all() and R.hichip_ssi.nunique() == 2:
        msg += f'; AUROC of the HiChIP SSIs among all candidates {auroc(R.hichip_ssi.values, logit):.4f}'
    print(msg + f' -> {out}')


# ------------------------------------------------------------------ train
def train(S, from_scratch=False, folds=(0, 1, 2, 3, 4)):
    from . import train as TR
    assert os.path.exists(path_of(S, 'tables', 'loops.tsv.gz')), 'training needs the tables of the sample (loops in the sample file, then prepare)'
    register(S)
    model = 'ssiformer_loops' if from_scratch else 'ssiformer_pretrained_loops'
    if not from_scratch:
        need = K.dataset(K.CFG['pretraining']['source_dataset'])['marks']
        assert K.dataset(S['name'])['marks'] == need, f'the mouse pretraining uses the marks {need}'
    TR.train(S['name'], model, folds)
    a = TR.oof_auroc(S['name'], model)
    if a is not None:
        print(f'{S["name"]} {model}: out-of-fold AUROC on the SSI table {a:.4f}; score the candidates with --model {S["name"]}/{model}')


def main():
    ap = argparse.ArgumentParser(description='SSIFormer on a new sample')
    ap.add_argument('sample', help='sample file (see configs/sample_example.yaml)')
    ap.add_argument('step', choices=['prepare', 'inputs', 'score', 'train'])
    ap.add_argument('procs', nargs='?', type=int, default=2, help='inputs: number of processes')
    ap.add_argument('--model', help='score: <dataset>/<model> (default: model of the sample file)')
    ap.add_argument('--strand-exchange', action='store_true')
    ap.add_argument('--chroms', help='score: only the candidates of these chromosomes (comma-separated)')
    ap.add_argument('--per-fold', action='store_true', help='score: also write the logit of each fold model')
    ap.add_argument('--out', help='score: output table (default samples/<name>/scores.tsv)')
    ap.add_argument('--from-scratch', action='store_true', help='train: without the mouse pretraining')
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    S = load(a.sample)
    if a.step == 'prepare':
        prepare(S)
    elif a.step == 'inputs':
        inputs(S, a.procs)
    elif a.step == 'score':
        score(S, a.model, a.strand_exchange, a.chroms.split(',') if a.chroms else None, a.per_fold, a.out)
    else:
        train(S, a.from_scratch, [int(x) for x in a.folds.split(',')])


if __name__ == '__main__':
    main()
