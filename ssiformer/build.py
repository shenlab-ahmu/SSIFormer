"""Construction of the SSI benchmark from the raw inputs (configs/benchmark.yaml; raw files in $SSIFORMER_RAW).

  python -m ssiformer.build tables                       # universes, folds, positives, cross-validation and sealed tables
  python -m ssiformer.build loops [<dataset> ...]        # tables of the loops without silencers of the HiChIP samples (human datasets)
  python -m ssiformer.build genome <hg38|mm10> <procs> [--chroms=c1,c2] [--sealed]   # 1-kb motif channels
  python -m ssiformer.build marks <cell> <procs> [--chroms=c1,c2] [--sealed]         # 1-kb mark tracks + anchor tensors

Outputs go to $SSIFORMER_BUILD (default <repo>/build), never over the benchmark folder. raw/sources.tsv lists the raw files with
their URL and md5; the model inputs are written to build/cache (copy them to cache/ to use them).

Rules. Universe: SilencerDB elements merged into loci (overlap >= 1 bp), locus = midpoint. SSI (positive): a 10-kb bin
pair (a < b) joined by a FitHiChIP (stringent) loop of the dataset's HiChIP sample with >= 1 locus midpoint in each bin.
Folds: chromosomes assigned greedily (descending locus count) to the fold with the smallest running total.
Candidates: pairs of anchor bins of the dataset's SSIs (same chromosome, b - a >= 2, distance within
[max(20 kb, min SSI distance), max SSI distance]), not an SSI, not near any exclusion loop of the cell.
Negatives (1:1, degree-targeted): weights w(a) w(b), w = remaining SSI quota of the bin (decremented after each draw) or
eps once used up; 'strata' designs draw within chromosome x log10-distance quantile strata of the SSIs, 'perpos'
designs draw one negative per SSI within 0.05 log10 units of its distance (SSIs in random order; an SSI without a
remaining candidate is dropped). Sealed tables: the same rules on the sealed chromosomes only (distance range of the
cross-validation SSIs).
Tables of loops without silencers (benchmark/loops; training with --extra-loops): loops of the dataset's HiChIP sample (10-kb bins,
b - a >= 2) with no silencer-locus midpoint in either anchor bin, on the cross-validation chromosomes with 20 kb <= d <= 2 Mb;
at most 10 x the SSIs of the dataset on these chromosomes (before negative matching) and at most 30,000 of them, drawn at
random; negatives 1:1 from pairs of their anchor bins by the rules of the SSI table (design of the dataset, not a loop of
the sample, not near a loop of the exclusion set of the cell, extended by configs/benchmark.yaml other_loops.exclusion_added).
Mark tracks and anchor tensors: the bigWig signal is read in chunks with the float64 running sum carried over (the same
numbers as one cumulative sum over the whole chromosome)."""
import glob
import gzip
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml

from . import config as K
from .utils import chrom_index, natural_key, rng_of, seed_of

BC = yaml.safe_load(open(os.path.join(K.REPO, 'configs', 'benchmark.yaml')))
BUILD = os.environ.get('SSIFORMER_BUILD', os.path.join(K.REPO, 'build'))
RAW = K.RAW
DESIGNS = {'base': dict(mode='strata', nstrata=20, eps=0.05), 'perpos_eps001': dict(mode='perpos', eps=0.01)}
LOOP_RE = re.compile(r'^(\S+):(\d+)-(\d+),(\S+)$')


# ------------------------------------------------------------------ loops and keys
def read_longrange(path, genome):
    """Loop Catalog longrange bed -> DataFrame(chrom, s1, e1, s2, e2), s1 <= s2, intra-chromosomal, canonical, deduplicated."""
    chroms = set(K.ALL_CHROMS[genome])
    rows = []
    with gzip.open(path, 'rt') as f:
        for ln in f:
            p = ln.rstrip('\n').split('\t')
            if len(p) < 4:
                continue
            m = LOOP_RE.match(p[3])
            if not m:
                continue
            c, s, e = p[0], int(p[1]), int(p[2])
            c2, s2, e2 = m.group(1), int(m.group(2)), int(m.group(3))
            if c.startswith('chrchr'):
                c = c[3:]
            if c2.startswith('chrchr'):
                c2 = c2[3:]
            if c != c2 or c not in chroms:
                continue
            rows.append((c, s, e, s2, e2) if s <= s2 else (c, s2, e2, s, e))
    return pd.DataFrame(rows, columns=['chrom', 's1', 'e1', 's2', 'e2']).drop_duplicates().reset_index(drop=True)


def loops_to_bins(d, res):
    a, b = (d.s1 // res).astype(np.int64), (d.s2 // res).astype(np.int64)
    return pd.DataFrame({'chrom': d.chrom.values, 'a': np.minimum(a, b).values, 'b': np.maximum(a, b).values}).drop_duplicates().reset_index(drop=True)


def pair_key(ci, a, b):
    ci, a, b = (np.asarray(x, dtype=np.int64) for x in (ci, a, b))
    assert (a < (1 << 21)).all() and (b < (1 << 21)).all()
    return (ci << 42) | (a << 21) | b


def keys_of(df, genome):
    return pair_key(df.chrom.map(chrom_index(genome)).values, df.a.values, df.b.values)


def near_keys(loopbins, r_loop, r_task, genome):
    """all task-grid pair keys (A <= C) near any loop in loopbins (resolution r_loop)."""
    if len(loopbins) == 0:
        return np.zeros(0, np.int64)
    ci = loopbins.chrom.map(chrom_index(genome)).values.astype(np.int64)
    x, y = loopbins.a.values.astype(np.int64), loopbins.b.values.astype(np.int64)
    lo_x, hi_x = np.maximum((x - 1) * r_loop, 0) // r_task, ((x + 2) * r_loop - 1) // r_task
    lo_y, hi_y = np.maximum((y - 1) * r_loop, 0) // r_task, ((y + 2) * r_loop - 1) // r_task
    wx, wy = hi_x - lo_x + 1, hi_y - lo_y + 1
    keys = []
    for dx in range(int(wx.max())):
        for dy in range(int(wy.max())):
            m = (dx < wx) & (dy < wy)
            A, Cc = lo_x[m] + dx, lo_y[m] + dy
            keys.append(pair_key(ci[m], np.minimum(A, Cc), np.maximum(A, Cc)))
    return np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)


def loop_file(sample, res, kind='fhs'):
    return {'fhs': os.path.join(RAW, 'loops', 'fhs', f'{sample}.{res}.fithichip_hp.longrange.bed.gz'),
            'fhl': os.path.join(RAW, 'loops', 'fhl', f'{sample}.{res}.fithichip_hp.longrange.bed.gz'),
            'hiccups': os.path.join(RAW, 'loops', 'hiccups', f'{sample}.{res}.hiccups.longrange.bed.gz')}[kind]


def unavailable():
    s = pd.read_csv(os.path.join(RAW, 'sources.tsv'), sep='\t', dtype=str).fillna('')
    return {os.path.basename(p) for p, n in zip(s.path, s.note) if n.startswith('UNAVAILABLE')}


def exclusion_files(cell, extra=()):
    out = []
    for ent in list(BC['exclusion'][cell]) + list(extra):
        for s in ent.get('samples', []):
            for r in ent.get('fhs_res', []):
                out.append((loop_file(s, r, 'fhs'), r))
            if ent.get('fhl_5kb') is True:
                out.append((loop_file(s, 5000, 'fhl'), 5000))
            if ent.get('hiccups_10kb') is True:
                out.append((loop_file(s, 10000, 'hiccups'), 10000))
        if isinstance(ent.get('fhl_5kb'), list):
            for s in ent['fhl_5kb']:
                out.append((loop_file(s, 5000, 'fhl'), 5000))
    na, seen, res = unavailable(), set(), []
    for p, r in out:
        if p in seen:
            continue
        seen.add(p)
        if not os.path.exists(p):
            if os.path.basename(p) in na:
                continue
            raise FileNotFoundError(p)
        res.append((p, r))
    return res


def exclusion_keys(cell, res, genome, extra=()):
    return np.unique(np.concatenate([near_keys(loops_to_bins(read_longrange(p, genome), r), r, res, genome)
                                     for p, r in exclusion_files(cell, extra)]))


# ------------------------------------------------------------------ universes, folds, positives
def merge_loci(d):
    out = []
    for c, g in d.groupby('chrom', sort=False):
        s, e, meth = g.start.values, g.end.values, g.method.values
        order = np.lexsort((e, s))
        s, e, meth = s[order], e[order], meth[order]
        cur_s, cur_e, ms, n = s[0], e[0], {meth[0]}, 1
        for i in range(1, len(s)):
            if s[i] < cur_e:
                cur_e = max(cur_e, e[i])
                ms.add(meth[i])
                n += 1
            else:
                out.append((c, cur_s, cur_e, ','.join(sorted(ms)), n))
                cur_s, cur_e, ms, n = s[i], e[i], {meth[i]}, 1
        out.append((c, cur_s, cur_e, ','.join(sorted(ms)), n))
    L = pd.DataFrame(out, columns=['chrom', 'start', 'end', 'methods', 'n_elements'])
    L['length'] = L.end - L.start
    L['mid'] = (L.start + L.end) // 2
    L = L.sort_values(['chrom', 'start'], key=lambda s: s.map(natural_key) if s.name == 'chrom' else s)
    return L.reset_index(drop=True)


def universe(u):
    spec = BC['universes'][u]
    d = pd.read_csv(os.path.join(RAW, spec['file']), sep='\t', header=None, usecols=[0, 1, 2, 3], names=['chrom', 'start', 'end', 'name'])
    d['eid'], d['method'] = d.name.str.split('|').str[0], d.name.str.split('|').str[1]
    d = d[d.chrom.isin(K.ALL_CHROMS[spec['genome']])]
    if spec['methods'] != 'all':
        d = d[d.method.isin(spec['methods'])]
    L = merge_loci(d.sort_values(['chrom', 'start', 'end']).reset_index(drop=True))
    L.insert(0, 'locus_id', np.arange(len(L)))
    return L


def folds(genome, univ):
    cnt = pd.concat([univ[u].chrom for u in BC['fold_balance'][genome]]).value_counts()
    elig = [c for c in K.ALL_CHROMS[genome] if c not in K.SEALED[genome]]
    tot, assign = [0] * 5, {}
    for c in sorted(elig, key=lambda c: (-int(cnt.get(c, 0)), natural_key(c))):
        k = int(np.argmin(tot))
        assign[c] = k
        tot[k] += int(cnt.get(c, 0))
    return pd.DataFrame([dict(chrom=c, n_loci=int(cnt.get(c, 0)), fold=assign.get(c, 'sealed')) for c in K.ALL_CHROMS[genome]])


def positives(name, L, fold_map):
    d = K.dataset(name)
    return positives_of(loops_to_bins(read_longrange(loop_file(d['label'], K.RES), d['genome']), K.RES), L, fold_map, d['genome'], name)


def positives_of(loops, L, fold_map, g, name):
    """SSIs: loops (10-kb bins) with a silencer-locus midpoint of the loci L in both anchor bins."""
    res = K.RES
    Lb = L.assign(bin=(L.mid // res).astype(np.int64))
    bt = Lb.groupby(['chrom', 'bin']).size()
    bset = set(zip(bt.index.get_level_values(0), bt.index.get_level_values(1)))
    ok = np.array([(c, a) in bset and (c, b) in bset for c, a, b in zip(loops.chrom, loops.a, loops.b)], bool) & (loops.b.values > loops.a.values)
    P = loops[ok].copy()
    P['d'] = (P.b - P.a) * res
    P['fold'] = P.chrom.map(fold_map).astype(str)
    P['sealed'] = P.chrom.isin(K.SEALED[g])
    P['n_loci_a'] = bt.reindex(pd.MultiIndex.from_arrays([P.chrom, P.a])).values
    P['n_loci_b'] = bt.reindex(pd.MultiIndex.from_arrays([P.chrom, P.b])).values
    P = P.sort_values(['chrom', 'a', 'b'], key=lambda s: s.map(natural_key) if s.name == 'chrom' else s).reset_index(drop=True)
    P.insert(0, 'loop_id', [f'{name}:{c}:{a}:{b}' for c, a, b in zip(P.chrom, P.a, P.b)])
    return P


# ------------------------------------------------------------------ candidates and negatives
def enumerate_pairs(bins_by_chrom, res, dmin, dmax):
    out = []
    lo, hi = max(2, int(np.ceil(dmin / res))), int(np.floor(dmax / res))
    for c, B in bins_by_chrom.items():
        B = np.unique(np.asarray(B, dtype=np.int64))
        if len(B) < 2:
            continue
        j0, j1 = np.searchsorted(B, B + lo, 'left'), np.searchsorted(B, B + hi, 'right')
        n = j1 - j0
        if n.sum() == 0:
            continue
        ia = np.repeat(np.arange(len(B)), n)
        offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        out.append(pd.DataFrame({'chrom': c, 'a': B[ia], 'b': B[np.repeat(j0, n) + offs]}))
    if not out:
        return pd.DataFrame(columns=['chrom', 'a', 'b', 'd'])
    D = pd.concat(out, ignore_index=True)
    D['d'] = (D.b - D.a) * res
    return D


def strata_edges(d_pos, n):
    return np.unique(np.quantile(np.log10(d_pos), np.linspace(0, 1, n + 1)))[1:-1]


def _weighted_draw(rng, wa, wc):
    w = wa * wc
    return int(rng.choice(len(w), p=w / w.sum()))


def build_table(P, cand, res, design, rng):
    """degree-targeted 1:1 table (SSIs then negatives, in draw order)."""
    dz = DESIGNS[design]
    eps = dz['eps']
    pos_keep, neg_take = [], []
    inner = strata_edges(P.d.values, dz.get('nstrata', 20))
    P = P.assign(ld=np.log10(P.d.values), sbin=np.digitize(np.log10(P.d.values), inner))
    cand = cand.assign(ld=np.log10(cand.d.values), sbin=np.digitize(np.log10(cand.d.values), inner))
    for c in sorted(P.chrom.unique(), key=natural_key):
        Pc, Cc = P[P.chrom == c], cand[cand.chrom == c]
        bins = np.unique(np.concatenate([Pc.a.values, Pc.b.values, Cc.a.values, Cc.b.values]))
        bidx = {b: i for i, b in enumerate(bins)}
        quota = np.zeros(len(bins))
        for x in np.concatenate([Pc.a.values, Pc.b.values]):
            quota[bidx[x]] += 1
        ca = np.array([bidx[x] for x in Cc.a.values], dtype=np.int64)
        cc = np.array([bidx[x] for x in Cc.b.values], dtype=np.int64)
        used = np.zeros(len(Cc), bool)
        cidx = Cc.cand_idx.values
        w = lambda ix: np.where(quota[ix] > 0, quota[ix], eps)  # noqa: E731
        if dz['mode'] == 'strata':
            slots = []
            for s, Ps in Pc.groupby('sbin'):
                cs = np.where(Cc.sbin.values == s)[0]
                if len(cs) < len(Ps):  # unfillable stratum: downsample the SSIs, take all candidates
                    keep = rng.choice(len(Ps), size=len(cs), replace=False) if len(cs) else np.array([], int)
                    pos_keep.extend(Ps.loop_id.values[np.sort(keep)])
                    for k in cs:
                        used[k] = True
                        quota[ca[k]] -= 1
                        quota[cc[k]] -= 1
                        neg_take.append(cidx[k])
                else:
                    pos_keep.extend(Ps.loop_id.values)
                    slots.extend([s] * len(Ps))
            slots = rng.permutation(np.array(slots, dtype=np.int64)) if slots else []
            by_s = {s: np.where(Cc.sbin.values == s)[0] for s in np.unique(slots)} if len(slots) else {}
            for s in slots:
                cs = by_s[s]
                cs = cs[~used[cs]]
                k = cs[_weighted_draw(rng, w(ca[cs]), w(cc[cs]))]
                used[k] = True
                quota[ca[k]] -= 1
                quota[cc[k]] -= 1
                neg_take.append(cidx[k])
        else:
            order = rng.permutation(len(Pc))
            srt = np.argsort(Cc.ld.values, kind='stable')
            cld_s = Cc.ld.values[srt]
            for i in order:
                p = Pc.iloc[i]
                cs = srt[np.searchsorted(cld_s, p.ld - 0.05, 'left'):np.searchsorted(cld_s, p.ld + 0.05, 'right')]
                cs = cs[~used[cs]]
                if len(cs) == 0:
                    continue
                k = cs[_weighted_draw(rng, w(ca[cs]), w(cc[cs]))]
                used[k] = True
                quota[ca[k]] -= 1
                quota[cc[k]] -= 1
                pos_keep.append(p.loop_id)
                neg_take.append(cidx[k])
    Pk = P[P.loop_id.isin(set(pos_keep))]
    Nt = cand.set_index('cand_idx').loc[neg_take].reset_index()
    return pd.concat([pd.DataFrame({'chrom': Pk.chrom.values, 'a': Pk.a.values, 'b': Pk.b.values, 'd': Pk.d.values, 'label': 1}),
                      pd.DataFrame({'chrom': Nt.chrom.values, 'a': Nt.a.values, 'b': Nt.b.values, 'd': Nt.d.values, 'label': 0})],
                     ignore_index=True), len(Pk) / max(1, len(P))


def candidates(P_sub, Pall, res, dmin, dmax, genome, excl):
    pb = {}
    for c, a, b in zip(P_sub.chrom, P_sub.a, P_sub.b):
        pb.setdefault(c, []).extend([a, b])
    cand0 = enumerate_pairs(pb, res, dmin, dmax)
    k0 = keys_of(cand0, genome)
    cand = cand0[~np.isin(k0, keys_of(Pall, genome)) & ~np.isin(k0, excl)].reset_index(drop=True)
    cand.insert(0, 'cand_idx', np.arange(len(cand)))
    return cand


def tables():
    for d in ('universe', 'positives', 'tables', 'sealed'):
        os.makedirs(os.path.join(BUILD, d), exist_ok=True)
    univ = {u: universe(u) for u in BC['universes']}
    for u, L in univ.items():
        L.to_csv(os.path.join(BUILD, 'universe', f'{u}.loci.tsv.gz'), sep='\t', index=False)
    fmap = {}
    for g in ('hg38', 'mm10'):
        F = folds(g, univ)
        F.to_csv(os.path.join(BUILD, f'folds_{g}.tsv'), sep='\t', index=False)
        fmap[g] = dict(zip(F.chrom, F.fold))
    summary = []
    for name in K.MOUSE + K.HUMAN:
        d = K.dataset(name)
        g, res, design = d['genome'], K.RES, BC['design'][name]
        Pall = positives(name, univ[BC['cell_universe'][d['cell']]], fmap[g])
        Pall.to_csv(os.path.join(BUILD, 'positives', f'{name}.tsv.gz'), sep='\t', index=False)
        excl = exclusion_keys(d['cell'], res, g)
        P = Pall[~Pall.sealed].reset_index(drop=True)
        dmin, dmax = max(BC['min_distance'], int(P.d.min())), int(P.d.max())
        T, ret = build_table(P, candidates(P, Pall, res, dmin, dmax, g, excl), res, design, np.random.default_rng(table_seed(name, 'crossval')))
        T['fold'] = T.chrom.map(fmap[g]).astype(str)
        T.to_csv(os.path.join(BUILD, 'tables', f'{name}.tsv.gz'), sep='\t', index=False, compression={'method': 'gzip', 'mtime': 0})
        row = dict(dataset=name, design=design, dmin=dmin, dmax=dmax, ssi_all=len(P), ssi=int(T.label.sum()), negatives=int((T.label == 0).sum()),
                   retained=round(ret, 4))
        if name != 'HeLaS3_CTCF':  # HeLa-S3 CTCF has no held-out table
            Ps = Pall[Pall.sealed].reset_index(drop=True)
            Ts, _ = build_table(Ps, candidates(Ps, Pall, res, dmin, dmax, g, excl), res, design,
                                np.random.default_rng(table_seed(name, 'heldout')))
            Ts['fold'] = 'sealed'
            Ts.to_csv(os.path.join(BUILD, 'sealed', f'{name}.tsv.gz'), sep='\t', index=False, compression={'method': 'gzip', 'mtime': 0})
            row.update(sealed_ssi=int(Ts.label.sum()), sealed_negatives=int((Ts.label == 0).sum()))
        summary.append(row)
        print(row, flush=True)
    pd.DataFrame(summary).to_csv(os.path.join(BUILD, 'tables_summary.tsv'), sep='\t', index=False)


# ------------------------------------------------------------------ tables of loops without silencers
def other_loops(loopbins, L, fold_map, genome, prefix):
    """loops of a HiChIP sample (10-kb bins) with b - a >= 2 and no silencer-locus midpoint of the loci L in either anchor bin."""
    res = K.RES
    sb = set(zip(L.chrom, (L.mid // res).astype(np.int64)))
    ok = np.array([(c, a) not in sb and (c, b) not in sb for c, a, b in zip(loopbins.chrom, loopbins.a, loopbins.b)], bool) & \
        (loopbins.b.values - loopbins.a.values >= 2)
    C = loopbins[ok].copy()
    C['d'] = (C.b - C.a) * res
    C['fold'] = C.chrom.map(fold_map).astype(str)
    C['sealed'] = C.chrom.isin(K.SEALED.get(genome, set()))
    C = C.sort_values(['chrom', 'a', 'b']).reset_index(drop=True)
    C.insert(0, 'loop_id', [f'{prefix}:{c}:{a}:{b}' for c, a, b in zip(C.chrom, C.a, C.b)])
    return C


def loops_table(C, n_ssi, design, excl, genome, draw_seed, rng):
    """table of loops without silencers: the loops C on cross-validation chromosomes with min_distance <= d <= max_distance, at most
    factor x n_ssi and max_loops of them (numpy default_rng(draw_seed)), with 1:1 negatives (build_table with generator rng)."""
    OL, res = BC['other_loops'], K.RES
    P0 = C[(~C.sealed) & (C.d >= OL['min_distance']) & (C.d <= OL['max_distance'])].reset_index(drop=True)
    assert len(P0), 'no loop without silencers on the cross-validation chromosomes'
    dmin, dmax = max(BC['min_distance'], int(P0.d.min())), int(P0.d.max())
    N = min(len(P0), int(OL['factor']) * n_ssi, int(OL['max_loops']))
    keep = np.sort(np.random.default_rng(draw_seed).choice(len(P0), N, replace=False))
    P = P0.iloc[keep].reset_index(drop=True)
    T, ret = build_table(P, candidates(P, C, res, dmin, dmax, genome, excl), res, design, rng)
    return T, dict(design=design, dmin=dmin, dmax=dmax, available_loops=len(P0), ssi_cv=n_ssi, drawn=len(P), loops=int(T.label.sum()),
                   negatives=int((T.label == 0).sum()), retained=round(ret, 4))


def table_seed(name, which):
    """seed of the negative draw of a table ('crossval' or 'heldout'): configs/benchmark.yaml for the datasets of the paper, else
    derived from the name."""
    s = BC['table_seed'][which]
    return int(s[name]) if name in s else seed_of('table', name, which)


def loops_draw_seed(name):
    """seed of the random draw of loops: configs/benchmark.yaml for the datasets of the paper, else derived from the name."""
    s = BC['other_loops']['draw_seed']
    return int(s[name]) if name in s else seed_of('loops', name, 'draw')


def loops_tables(names=None):
    """tables of loops without silencers of the human datasets -> build/loops/<dataset>.tsv.gz and build/loops_summary.tsv."""
    names = names or K.HUMAN
    base = BUILD if os.path.exists(os.path.join(BUILD, 'universe')) else K.BENCH   # universes and folds of 'tables', else provided
    os.makedirs(os.path.join(BUILD, 'loops'), exist_ok=True)
    summary, excl = [], {}
    for name in names:
        d = K.dataset(name)
        g, cell, res, design = d['genome'], d['cell'], K.RES, BC['design'][name]
        L = pd.read_csv(os.path.join(base, 'universe', f'{BC["cell_universe"][cell]}.loci.tsv.gz'), sep='\t')
        F = pd.read_csv(os.path.join(base, f'folds_{g}.tsv'), sep='\t', dtype=str)
        fmap = dict(zip(F.chrom, F.fold))
        if cell not in excl:
            excl[cell] = exclusion_keys(cell, res, g, BC['other_loops'].get('exclusion_added', {}).get(cell, []))
        Pall = positives(name, L, fmap)
        C = other_loops(loops_to_bins(read_longrange(loop_file(d['label'], res), g), res), L, fmap, g, name)
        T, info = loops_table(C, int((~Pall.sealed).sum()), design, excl[cell], g, loops_draw_seed(name), rng_of('CTRL', name, f'tables_{design}'))
        T['fold'] = T.chrom.map(fmap).astype(str)
        T.to_csv(os.path.join(BUILD, 'loops', f'{name}.tsv.gz'), sep='\t', index=False, compression={'method': 'gzip', 'mtime': 0})
        summary.append(dict(dataset=name, **info))
        print(summary[-1], flush=True)
    pd.DataFrame(summary).to_csv(os.path.join(BUILD, 'loops_summary.tsv'), sep='\t', index=False)


# ------------------------------------------------------------------ caches
def _chroms(genome, sealed, only):
    cs = [c for c in K.ALL_CHROMS[genome] if (c in K.SEALED[genome]) == sealed]
    return [c for c in cs if not only or c in only]


def read_jaspar(p):
    lines = gzip.open(p, 'rt').read().strip().splitlines()
    mid, name = lines[0][1:].split()[:2]
    rows = {}
    for ln in lines[1:]:
        rows[ln.split('[')[0].strip()] = [float(x) for x in ln.split('[')[1].split(']')[0].split()]
    return mid, name, np.array([rows[b] for b in 'ACGT']).T


def _motifs():
    out = []
    for mid in ['MA0139.1', 'MA0138', 'MA0095', 'MA0088']:
        f = sorted(glob.glob(os.path.join(RAW, 'jaspar', f'{mid}*.jaspar.gz')))[-1]
        m, name, M = read_jaspar(f)
        W = np.log2(((M + 0.2) / (M.sum(1, keepdims=True) + 0.8)) / 0.25)
        for strand, Wx in (('+', W), ('-', W[::-1, ::-1])):   # reverse complement: positions reversed, ACGT reversed
            T = np.concatenate([Wx, np.full((len(Wx), 1), -1e4)], 1).astype(np.float32)   # N column
            out.append((f'{m}_{name}', strand, T, Wx.min(1).sum(), Wx.max(1).sum()))
    return out


def _genome_work(args):
    """motif channels of one chromosome; args = (genome, chrom[, fasta path]) (default fasta raw/fa/<genome>.fa)."""
    import pyfaidx
    genome, chrom = args[:2]
    fasta = args[2] if len(args) > 2 else os.path.join(RAW, 'fa', f'{genome}.fa')
    LUT = np.full(256, 4, np.uint8)
    for ch, v in zip(b'ACGTacgt', [0, 1, 2, 3, 0, 1, 2, 3]):
        LUT[ch] = v
    MOT, THR, CHUNK = _motifs(), 0.85, 5_000_000
    seq = pyfaidx.Fasta(fasta, rebuild=False, read_ahead=10000)[chrom][:].seq
    codes = LUT[np.frombuffer(seq.encode(), np.uint8)]
    del seq
    L = len(codes)
    nb = (L + 999) // 1000
    bc = np.concatenate([codes, np.full(nb * 1000 - L, 4, np.uint8)]).reshape(nb, 1000)
    nn = (bc == 4).sum(1)
    gcf, nfrac = ((bc == 1) | (bc == 2)).sum(1) / np.maximum(1, 1000 - nn), nn / 1000.0
    out = np.zeros((nb, len(MOT) * 2), np.float32)
    maxL = max(len(m[2]) for m in MOT)
    ext = np.concatenate([codes, np.full(maxL, 4, np.uint8)])
    for c0 in range(0, nb * 1000, CHUNK):
        n = min(CHUNK, nb * 1000 - c0)
        b0 = c0 // 1000
        for mi, (_, _, T, smin, smax) in enumerate(MOT):
            Lm = len(T)
            S = np.zeros(n, np.float32)
            seg = ext[c0:c0 + n + Lm]
            if len(seg) < n + Lm:
                seg = np.concatenate([seg, np.full(n + Lm - len(seg), 4, np.uint8)])
            for k in range(Lm):
                S += T[k][seg[k:k + n]]
            rel = np.maximum((S - smin) / (smax - smin), 0).reshape(-1, 1000)
            out[b0:b0 + n // 1000, 2 * mi] = (rel >= THR).sum(1)
            out[b0:b0 + n // 1000, 2 * mi + 1] = rel.max(1)
    return chrom, gcf.astype(np.float16), nfrac.astype(np.float16), out.astype(np.float16)


def genome_cache(genome, nproc, sealed=False, only=None):
    chroms = _chroms(genome, sealed, only)
    arrays = {}
    with Pool(nproc) as pool:
        for chrom, gcf, nf, mo in pool.imap_unordered(_genome_work, [(genome, c) for c in chroms]):
            arrays[f'{chrom}__gc'], arrays[f'{chrom}__nfrac'], arrays[f'{chrom}__motif'] = gcf, nf, mo
    arrays['channels'] = np.array([f'{n}{s}_{k}' for n, s, *_ in _motifs() for k in ('count', 'maxrel')])
    arrays['chroms'] = np.array(chroms)
    out = os.path.join(BUILD, 'cache', 'sealed' if sealed else '', f'genome_{genome}_1kb.npz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **arrays)
    print('wrote', out)


CHUNK_BP = 4_000_000   # bigWig bases read at once


def bigwig_inputs(path, chrom, centres):
    """1-kb track (log1p of the mean signal per kb; NaN = 0) and anchor tensors (log1p of window means at the scales of
    configs/benchmark.yaml, centred on the given positions) of one chromosome of a bigWig; None if the chromosome is absent.
    The chromosome is read in chunks; the float64 running sum is carried from chunk to chunk, which gives the same numbers as
    one cumulative sum over the whole chromosome."""
    import pyBigWig
    bw = pyBigWig.open(path)
    L = bw.chroms().get(chrom)
    if L is None:
        bw.close()
        return None
    centres = np.asarray(centres, np.int64)
    nb = (L + 999) // 1000
    e1, s1 = np.minimum(np.arange(1, nb + 1) * 1000, L), np.arange(nb) * 1000
    S, E = [], []
    for W, n in BC['tracks']['anchor_scales']:
        starts = centres[:, None] - W // 2 + (W // n) * np.arange(n)[None, :]
        S.append(np.clip(starts, 0, L))
        E.append(np.clip(starts + W // n, 0, L))
    need = np.unique(np.concatenate([s1, e1] + [x.ravel() for x in S + E]))
    cs_need = np.zeros(len(need))   # cumulative signal before each needed position (0 at position 0)
    carry, pos = 0.0, 0
    while pos < L:
        n = min(CHUNK_BP, L - pos)
        cs = np.cumsum(np.concatenate([[carry], np.nan_to_num(bw.values(chrom, pos, pos + n, numpy=True), nan=0.0).astype(np.float64)]))
        m = (need > pos) & (need <= pos + n)
        cs_need[m] = cs[need[m] - pos]
        carry = cs[-1]
        pos += n
    bw.close()

    def window_sum(s, e):
        return cs_need[np.searchsorted(need, e)] - cs_need[np.searchsorted(need, s)]
    track = np.log1p(np.maximum(window_sum(s1, e1) / 1000.0, 0)).astype(np.float16)
    xb = (np.stack([window_sum(s, e) / (W // n) for (W, n), s, e in zip(BC['tracks']['anchor_scales'], S, E)], 1) if len(centres)
          else np.zeros((0, 3, 20)))
    return track, np.log1p(np.maximum(xb, 0)).astype(np.float16)


def _marks_work(args):
    cell, genome, mark, chrom, bin_centres = args
    got = bigwig_inputs(os.path.join(RAW, 'bigwig', cell, f'{cell}_{mark}.bigWig'), chrom, bin_centres)
    if got is None:   # chromosome absent from this bigWig -> all-zero signal over the fasta length
        fai = {ln.split('\t')[0]: int(ln.split('\t')[1]) for ln in open(os.path.join(RAW, 'fa', f'{genome}.fa.fai'))}
        return mark, chrom, np.zeros((fai[chrom] + 999) // 1000, np.float16), np.zeros((len(bin_centres), 3, 20), np.float16)
    return (mark, chrom) + got


def mark_cache(cell, nproc, sealed=False, only=None):
    """1-kb tracks of the 7 marks (+ silencer-locus midpoint count) and anchor tensors of the SSI anchor bins and of the anchor
    bins of the tables of loops without silencers (build/loops, else benchmark/loops)."""
    genome = [d['genome'] for d in K.DATASETS.values() if d['cell'] == cell][0]
    chroms = _chroms(genome, sealed, only)
    L = pd.read_csv(os.path.join(BUILD if os.path.exists(os.path.join(BUILD, 'universe')) else K.BENCH, 'universe',
                                 f'{BC["cell_universe"][cell]}.loci.tsv.gz'), sep='\t')
    L = L[L.chrom.isin(chroms)]
    pbins = set()
    for name, d in K.DATASETS.items():
        if d['cell'] != cell:
            continue
        pf = os.path.join(BUILD, 'positives', f'{name}.tsv.gz')
        lf = os.path.join(BUILD, 'loops', f'{name}.tsv.gz')
        lf = lf if os.path.exists(lf) else K.table_path(name, loops=True)
        for f in ([] if name in K.IT and not os.path.exists(pf) else [pf]) + ([lf] if os.path.exists(lf) else []):
            P = pd.read_csv(f, sep='\t')   # the independent-test tables are provided in benchmark/ (no positives file)
            P = P[P.chrom.isin(chroms)]
            pbins |= set(zip(P.chrom, P.a)) | set(zip(P.chrom, P.b))
    per_chrom = {c: np.array(sorted(b for cc, b in pbins if cc == c), dtype=np.int64) for c in chroms}
    jobs = [(cell, genome, m, c, per_chrom[c] * K.RES + K.RES // 2) for c in chroms for m in K.MARKS]
    tracks, xbin = {}, {}
    with Pool(nproc) as pool:
        for m, c, tr, xb in pool.imap_unordered(_marks_work, jobs):
            tracks[(m, c)], xbin[(m, c)] = tr, xb
    arrays = {}
    Ls = L.assign(kb=(L.mid // 1000).astype(np.int64))
    for c in chroms:
        tr = np.stack([tracks[(m, c)] for m in K.MARKS], 1)
        arrays[f'{c}__marks'] = tr
        arrays[f'{c}__silcnt'] = np.bincount(Ls.kb[Ls.chrom == c].values, minlength=len(tr))[:len(tr)].astype(np.uint16)
    arrays['marks'], arrays['chroms'] = np.array(K.MARKS), np.array(chroms)
    od = os.path.join(BUILD, 'cache', 'sealed' if sealed else '')
    os.makedirs(od, exist_ok=True)
    np.savez_compressed(os.path.join(od, f'tracks_{cell}_1kb.npz'), **arrays)
    ch = [np.array([c] * len(per_chrom[c])) for c in chroms if len(per_chrom[c])]
    xs = [np.stack([xbin[(m, c)] for m in K.MARKS], 1) for c in chroms if len(per_chrom[c])]
    np.savez_compressed(os.path.join(od, f'anchor_bin_{cell}_{K.RES}.npz'), chrom=np.concatenate(ch),
                        bin=np.concatenate([per_chrom[c] for c in chroms if len(per_chrom[c])]), x=np.concatenate(xs).astype(np.float16),
                        marks=np.array(K.MARKS), scales=np.array(['1kb@50bp', '10kb@500bp', '100kb@5kb']))
    print('wrote', od, cell)


if __name__ == '__main__':
    a = sys.argv[1:]
    only = next((x.split('=', 1)[1].split(',') for x in a if x.startswith('--chroms=')), None)
    if a[0] == 'tables':
        tables()
    elif a[0] == 'loops':
        loops_tables(a[1:] or None)
    elif a[0] == 'genome':
        genome_cache(a[1], int(a[2]), '--sealed' in a, only)
    elif a[0] == 'marks':
        mark_cache(a[1], int(a[2]), '--sealed' in a, only)
