"""Seeds, genomic keys and metrics shared by all modules. Seeds are derived with SHA-256."""
import hashlib
import json
import os

import numpy as np

from .config import ALL_CHROMS


def seed_of(stage, key, purpose):
    """int(sha256('<stage>|<key>|<purpose>')[:8], 16) -- the only seed rule used."""
    return int(hashlib.sha256(f'{stage}|{key}|{purpose}'.encode()).hexdigest()[:8], 16)


def rng_of(stage, key, purpose):
    return np.random.default_rng(seed_of(stage, key, purpose))


def natural_key(c):
    s = c.replace('chr', '')
    return (0, int(s)) if s.isdigit() else (1, s)


def chrom_index(genome):
    return {c: i for i, c in enumerate(ALL_CHROMS[genome])}


def bin_key(ci, a):
    return (np.asarray(ci, dtype=np.int64) << 32) | np.asarray(a, dtype=np.int64)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def auprc(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(np.asarray(y), s))


def json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    os.replace(tmp, path)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()
