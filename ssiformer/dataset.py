"""Model inputs of one dataset: 1-kb loop-domain tokens (gathered on the GPU from one concatenated genome track),
multi-scale anchor tensors and log distance; per-fold normalisation fitted on training rows; the orientation flip F."""
import os

import numpy as np
import pandas as pd

from . import config as K
from .utils import natural_key

SPAN_MOTIF = ['CTCF+_maxrel', 'CTCF+_count', 'CTCF-_maxrel', 'CTCF-_count', 'REST_count', 'YY1_count', 'ZNF143_count']
MOTIF_IDS = {'CTCF': 'MA0139.1', 'REST': 'MA0138', 'YY1': 'MA0095', 'ZNF143': 'MA0088'}


def load_table(name, sealed=False, loops=False):
    T = pd.read_csv(K.table_path(name, sealed, loops), sep='\t', dtype={'fold': str})
    T = T.reset_index(drop=True)
    T['row'] = np.arange(len(T))
    return T


def fold_rows(T, k):
    f = T.fold.astype(str).values
    te = np.where(f == str(k))[0]
    va = np.where(f == str((k + 1) % 5))[0]
    tr = np.where((f != str(k)) & (f != str((k + 1) % 5)))[0]
    return tr, va, te


def make_batches(rows, L, bs, rng=None, budget=None):
    """Length-bucketed batches. rng None -> deterministic (prediction); else jitter within equal L and permuted order.
    budget (prediction only): a batch is further split so that the sum of the squared input lengths of a part stays within
    budget (rows are scored independently; the split changes outputs only by fp16 rounding)."""
    rows = np.asarray(rows)
    if rng is None:
        order = rows[np.lexsort((rows, L[rows]))]
    else:
        jit = rng.random(len(rows))
        order = rows[np.lexsort((jit, L[rows]))]
    b = [order[i:i + bs] for i in range(0, len(order), bs)]
    if rng is not None:
        b = [b[i] for i in rng.permutation(len(b))]
    elif budget:
        parts = []
        for x in b:
            cur, tot = [], 0
            for r in x:
                c = int(L[r]) ** 2
                if cur and tot + c > budget:
                    parts.append(np.asarray(cur))
                    cur, tot = [], 0
                cur.append(r)
                tot += c
            if cur:
                parts.append(np.asarray(cur))
        b = parts
    return b


class SpanData:
    """inputs of one table: the SSI table of a dataset (cross-validation or held-out chromosomes), its table of loops without silencers
    (loops=True) or a given table of pairs of the same cell."""

    def __init__(self, name, device, table=None, sealed=False, loops=False):
        import torch
        self.torch = torch
        d = K.dataset(name)
        self.name, self.cell, self.g, self.res = name, d['cell'], d['genome'], K.RES
        self.rk = self.res // 1000
        self.marks = d['marks']
        self.T = T = table if table is not None else load_table(name, sealed, loops)
        self.dev = device
        cache = K.cache_dir(name, sealed)
        tr = np.load(os.path.join(cache, f'tracks_{self.cell}_1kb.npz'))
        gn = np.load(os.path.join(cache, f'genome_{self.g}_1kb.npz'))
        mi = [list(tr['marks']).index(m) for m in self.marks]
        ch = list(gn['channels'])
        ci = lambda s: ch.index(s)  # noqa: E731
        ctcf = [ci('MA0139.1_CTCF+_maxrel'), ci('MA0139.1_CTCF+_count'), ci('MA0139.1_CTCF-_maxrel'), ci('MA0139.1_CTCF-_count')]
        grp = {k: [i for i, c in enumerate(ch) if c.startswith(p) and c.endswith('_count')]
               for k, p in (('REST', 'MA0138'), ('YY1', 'MA0095'), ('ZNF143', 'MA0088'))}
        assert all(len(v) == 2 for v in grp.values()), grp
        self.channels = self.marks + SPAN_MOTIF + ['silcnt', 'anchor_ind']
        self.n_trk = len(self.channels) - 1  # anchor_ind is built on the fly
        self.swap_idx = (len(self.marks) + 0, len(self.marks) + 1, len(self.marks) + 2, len(self.marks) + 3)  # +max,+cnt,-max,-cnt
        chroms = sorted(set(T.chrom), key=natural_key)
        blocks, self.off, self.clen, o = [], {}, {}, 0
        for c in chroms:
            M = tr[f'{c}__marks'][:, mi].astype(np.float32)
            Mo = gn[f'{c}__motif'].astype(np.float32)
            S = tr[f'{c}__silcnt'].astype(np.float32)[:, None]
            n = min(len(M), len(Mo))
            assert abs(len(M) - len(Mo)) <= 1, (c, len(M), len(Mo))
            X = np.concatenate([M[:n], Mo[:n][:, ctcf]] + [Mo[:n][:, grp[k]].sum(1, keepdims=True) for k in ('REST', 'YY1', 'ZNF143')]
                               + [S[:n]], 1)
            blocks.append(X)
            self.off[c], self.clen[c] = o, n
            o += n
        self.raw = np.concatenate(blocks).astype(np.float32)  # (N, n_trk)
        self.N = len(self.raw)
        cs = np.zeros((self.N + 1, self.n_trk))
        cs[1:] = np.cumsum(self.raw.astype(np.float64), 0)
        cs2 = np.zeros((self.N + 1, self.n_trk))
        cs2[1:] = np.cumsum(self.raw.astype(np.float64) ** 2, 0)
        self._cs, self._cs2 = cs, cs2
        self.L = ((T.b.values - T.a.values + 1) * self.rk + 2 * K.FLANK).astype(np.int64)
        self.start_kb = (T.a.values * self.rk - K.FLANK).astype(np.int64)
        self.row_off = T.chrom.map(self.off).values.astype(np.int64)
        self.row_clen = T.chrom.map(self.clen).values.astype(np.int64)
        ab = np.load(os.path.join(cache, f'anchor_bin_{self.cell}_{self.res}.npz'))
        ami = [list(ab['marks']).index(m) for m in self.marks]
        key = {(str(c), int(b)): i for i, (c, b) in enumerate(zip(ab['chrom'], ab['bin']))}
        ia = np.array([key[(c, int(a))] for c, a in zip(T.chrom, T.a)])
        ic = np.array([key[(c, int(b))] for c, b in zip(T.chrom, T.b)])
        ubin = np.unique(np.concatenate([ia, ic]))
        remap = {u: i for i, u in enumerate(ubin)}
        self.ia = np.array([remap[x] for x in ia])
        self.ic = np.array([remap[x] for x in ic])
        self.anc_raw = ab['x'][ubin][:, ami].astype(np.float32).transpose(0, 2, 1, 3)  # (nb, 3, 6, 20)
        self.logd = np.log10(T.d.values.astype(np.float64))
        self.y = T.label.values.astype(np.float32)
        self.norm = None

    def fit_norm(self, tr):
        s = self.start_kb[tr]
        e = s + self.L[tr]
        s_c = np.clip(s, 0, self.row_clen[tr]) + self.row_off[tr]
        e_c = np.clip(e, 0, self.row_clen[tr]) + self.row_off[tr]
        n = (e_c - s_c).sum()
        tot = (self._cs[e_c] - self._cs[s_c]).sum(0)
        tot2 = (self._cs2[e_c] - self._cs2[s_c]).sum(0)
        mu = tot / n
        sd = np.sqrt(np.maximum(tot2 / n - mu ** 2, 0))
        sd[sd < 1e-6] = 1.0
        ub = np.unique(np.concatenate([self.ia[tr], self.ic[tr]]))
        A = self.anc_raw[ub]
        amu = A.mean((0, 3))
        asd = A.std((0, 3))
        asd[asd < 1e-6] = 1.0
        dmu, dsd = self.logd[tr].mean(), max(self.logd[tr].std(), 1e-6)
        self.norm = dict(span_mean=mu, span_sd=sd, span_tokens=int(n), anchor_mean=amu, anchor_sd=asd, anchor_bins=len(ub),
                         logd_mean=dmu, logd_sd=dsd)
        return self.norm

    def set_norm(self, norm):
        """use stored normalisation statistics (e.g. those a fold model was trained with)."""
        self.norm = {k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v) for k, v in norm.items()}

    def to_device(self):
        torch = self.torch
        nm = self.norm
        Z = (self.raw - np.asarray(nm['span_mean'])[None].astype(np.float32)) / np.asarray(nm['span_sd'])[None].astype(np.float32)
        Z = np.concatenate([Z, np.zeros((1, self.n_trk), np.float32)])  # zero row = out of chromosome / padding
        self.trk = torch.tensor(Z, dtype=torch.float16, device=self.dev)
        self.zero_row = self.N
        A = (self.anc_raw - np.asarray(nm['anchor_mean'])[None, :, :, None]) / np.asarray(nm['anchor_sd'])[None, :, :, None]
        self.anc = torch.tensor(A.astype(np.float32), device=self.dev)
        d = lambda x, dt: torch.tensor(x, dtype=dt, device=self.dev)  # noqa: E731
        self.t_L = d(self.L, torch.long)
        self.t_start = d(self.start_kb, torch.long)
        self.t_off = d(self.row_off, torch.long)
        self.t_clen = d(self.row_clen, torch.long)
        self.t_ia = d(self.ia, torch.long)
        self.t_ic = d(self.ic, torch.long)
        self.t_logd = d(((self.logd - nm['logd_mean']) / nm['logd_sd']).astype(np.float32)[:, None], torch.float32)
        self.t_y = d(self.y, torch.float32)

    def batch(self, rows, flip=None):
        """rows: np int array. flip: np bool array (per row) or None. Returns A, Cc, span, lens, logd, y."""
        torch = self.torch
        r = torch.as_tensor(rows, device=self.dev, dtype=torch.long)
        L = self.t_L[r]
        Lmax = int(L.max())
        t = torch.arange(Lmax, device=self.dev)[None, :]
        pos = self.t_start[r][:, None] + t
        valid = t < L[:, None]
        inchr = (pos >= 0) & (pos < self.t_clen[r][:, None])
        gidx = torch.where(valid & inchr, self.t_off[r][:, None] + pos, torch.full_like(pos, self.zero_row))
        x = self.trk[gidx]
        rk = self.rk
        ind = (((t >= K.FLANK) & (t < K.FLANK + rk)) | ((t >= L[:, None] - K.FLANK - rk) & (t < L[:, None] - K.FLANK))) & valid
        x = torch.cat([x, ind[..., None].to(x.dtype)], -1)
        A = self.anc[self.t_ia[r]]
        Cc = self.anc[self.t_ic[r]]
        if flip is not None and flip.any():
            f = torch.as_tensor(flip, device=self.dev)
            x, A, Cc = apply_flip(x, L, A, Cc, f, self.swap_idx)
        return A, Cc, x, L, self.t_logd[r], self.t_y[r]


def reverse_valid(x, lengths):
    """reverse each sequence within its valid length (B, L, C); padding positions stay in place."""
    import torch
    B, L = x.shape[:2]
    t = torch.arange(L, device=x.device)[None, :].expand(B, L)
    idx = torch.where(t < lengths[:, None], lengths[:, None] - 1 - t, t)
    return torch.gather(x, 1, idx[..., None].expand_as(x))


def swap_ctcf(x, swap_idx):
    pmax, pcnt, mmax, mcnt = swap_idx
    perm = list(range(x.shape[-1]))
    perm[pmax], perm[mmax] = mmax, pmax
    perm[pcnt], perm[mcnt] = mcnt, pcnt
    return x[..., perm]


def apply_flip(x, L, A, Cc, f, swap_idx):
    """orientation flip F (reverse the region, swap CTCF strands, swap and reverse the anchors) on rows where f is True."""
    import torch
    xf = swap_ctcf(reverse_valid(x, L), swap_idx)
    x = torch.where(f[:, None, None], xf, x)
    Af, Cf = Cc.flip(-1), A.flip(-1)
    A2 = torch.where(f[:, None, None, None], Af, A)
    C2 = torch.where(f[:, None, None, None], Cf, Cc)
    return x, A2, C2
