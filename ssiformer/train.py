"""Train (or pretrain / fine-tune) SSIFormer on one dataset, folds 0-4, seed 0.

  python -m ssiformer.train <dataset> <model> [--folds 0,1,2,3,4] [--pretrain] [--extra-loops]

<model> is a key of configs/ssiformer.yaml models. --pretrain fits the model on mESC_H3K27ac (folds 1-4 train, fold 0
early stopping) and writes runs/pretrain/<model>/best.pt, the initialisation of 'ssiformer_pretrained' (without a
local pretraining run, 'ssiformer_pretrained' starts from weights/pretrain/ssiformer.pt, the pretrained model of the paper).

--extra-loops trains the version of <model> that also learns from the loops without silencers of the dataset's HiChIP sample (model key
'<model>_loops'; table benchmark/loops/<dataset>.tsv.gz, see ssiformer.build loops). The training folds of the SSI table and of
the loop table are used in every epoch (batches are drawn within one table and interleaved in random order; each table keeps
its own input normalisation, fitted on its training rows); early stopping and testing use the SSI table only.
'ssiformer_pretrained --extra-loops' (mouse pretraining, then fine-tuning on the SSIs and the loops without silencers) is the extended
configuration.

Outputs per fold (runs/cv/<dataset>/<model>/fold<k>/): config.json, log.txt, val.npz, oof.npz (test-fold logits of both
orientations and their mean, SSI table), best.pt (fp16 state_dict), done.json."""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd

from . import config as K
from . import dataset as DS
from . import model as MD
from .utils import auroc, json_dump, seed_of

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256,garbage_collection_threshold:0.6')
SOURCE = {'ssiformer_pretrained': 'ssiformer', 'ssiformer_pretrained_loops': 'ssiformer'}
SEEDS = os.path.join(K.REPO, 'configs', 'seeds.tsv')


def fold_seeds(name, model, fold, seed):
    """(torch seed, batch-order seed) of one fold: configs/seeds.tsv for seed 0 (the integers used for the models of the paper),
    else derived from the names."""
    if seed == 0 and os.path.exists(SEEDS):
        S = pd.read_csv(SEEDS, sep='\t', dtype={'fold': str})
        r = S[(S.dataset == name) & (S.model == model) & (S.fold == str(fold))]
        if len(r):
            return int(r.torch_seed.iloc[0]), int(r.order_seed.iloc[0])
    key, p = f'{name}|{model}', f'fold{fold}|seed{seed}'
    return seed_of('train', key, p), seed_of('train', key, f'{p}|order')


def gpu():
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    dev = torch.device('cuda:0')
    frac = float(os.environ.get('SSIFORMER_GPU_FRACTION', '0.8'))   # an allocation beyond this share of the device memory fails
    if frac > 0:                                                     # instead of spilling into shared system memory
        torch.cuda.set_per_process_memory_fraction(frac, 0)
    return dev


def memory_budget(default):
    """micro-batch budget (sum of squared input lengths per micro-batch; the optimisation step is the same). SSIFORMER_MEMORY_BUDGET
    overrides it, e.g. on a GPU with less memory (only the floating-point summation order of the gradients changes)."""
    return int(os.environ.get('SSIFORMER_MEMORY_BUDGET') or default)


def schedule(lr, warm, total):
    """linear warm-up over the first `warm` steps, then cosine to 0 at step `total`."""
    def lr_at(step):
        if step < warm:
            return lr * (step + 1) / warm
        p = min(1.0, (step - warm) / max(1, total - warm))
        return lr * 0.5 * (1 + math.cos(math.pi * p))
    return lr_at


def predict_logits(model, D, rows, noflip, bs, budget=None):
    """logits of the rows in the forward orientation and after the orientation flip F."""
    import torch
    model.eval()
    out_f = np.zeros(len(rows), np.float32)
    out_r = np.zeros(len(rows), np.float32)
    pos = {r: i for i, r in enumerate(rows)}
    with torch.no_grad():
        for b in DS.make_batches(rows, D.L, bs, None, budget):
            idx = np.array([pos[r] for r in b])
            for flip, out in ((None, out_f),) + (() if noflip else ((np.ones(len(b), bool), out_r),)):
                A, Cc, x, L, ld, _ = D.batch(b, flip)
                with torch.autocast('cuda', dtype=torch.float16):
                    lo = model(A, Cc, x, L, ld)
                out[idx] = lo.float().cpu().numpy()
    return (out_f, out_f.copy()) if noflip else (out_f, out_r)


def train_batch(model, opt, scaler, D, b, flips, lr, budget, grad_clip, tot_loss):
    """one optimisation step on batch b (split into micro-batches when long)."""
    import torch
    import torch.nn.functional as F
    for g in opt.param_groups:
        g['lr'] = lr
    opt.zero_grad(set_to_none=True)
    for sub in np.array_split(b, max(1, math.ceil(len(b) * int(D.L[b].max()) ** 2 / budget))):
        A, Cc, x, L, ld, y = D.batch(sub, flips[sub])
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(A, Cc, x, L, ld)
        loss = F.binary_cross_entropy_with_logits(out.float(), y, reduction='sum') / len(b)
        scaler.scale(loss).backward()
        tot_loss += loss.detach() * len(b)
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(opt)
    scaler.update()


def new_model(spec, model, D, dev):
    import torch
    net = MD.build(spec['family'], len(D.marks), len(D.channels)).to(dev)
    if spec['pretrained']:
        src = K.pretrained_weights(SOURCE[model])
        net.load_state_dict({kk: v.float() if v.is_floating_point() else v for kk, v in torch.load(src, map_location=dev).items()})
    return net


def save_outputs(od, T, va, te, yl, val, test, best_state, done):
    import torch
    (vf, vr), (tf, trv) = val, test
    np.savez_compressed(os.path.join(od, 'val.npz'), row=va, label=yl[va].astype(np.int8), logit_fwd=vf, logit_flip=vr,
                        logit=(vf + vr) / 2)
    if te is not None:
        np.savez_compressed(os.path.join(od, 'oof.npz'), row=te, chrom=T.chrom.values[te].astype('U5'), a=T.a.values[te],
                            b=T.b.values[te], label=yl[te].astype(np.int8), logit_fwd=tf, logit_flip=trv, logit=(tf + trv) / 2)
    torch.save({kk: (v.half() if v.is_floating_point() else v) for kk, v in best_state.items()}, os.path.join(od, 'best.pt'))
    json_dump(done, os.path.join(od, 'done.json'))


def norm_json(norm):
    return {k_: (v.tolist() if hasattr(v, 'tolist') else v) for k_, v in norm.items()}


def train(name, model, folds=(0, 1, 2, 3, 4), pretrain=False):
    if K.MODELS[model].get('extra_loops'):
        assert not pretrain, '--pretrain does not apply to training with loops without silencers'
        return train_with_loops(name, model, folds)
    import torch
    TR = K.CFG['training']
    PT = K.CFG['pretraining']
    spec = K.MODELS[model]
    NOFLIP = not spec['equivariant']
    assert not pretrain or (name == PT['source_dataset'] and not spec['pretrained'])
    seed = int(TR['seed'])
    PATIENCE, BS, MAXEP = int(TR['patience']), int(TR['batch']), int(TR['max_epochs'])
    BUDGET = memory_budget(TR['memory_budget'])
    dev = gpu()
    D = DS.SpanData(name, dev)

    for k in (['P'] if pretrain else list(folds)):
        od = os.path.join(K.RUNS, 'pretrain', model) if pretrain else K.run_dir(name, model, k)
        if os.path.exists(os.path.join(od, 'done.json')):
            print('skip (done)', od)
            continue
        os.makedirs(od, exist_ok=True)
        if pretrain:
            fo = D.T.fold.astype(str).values
            va = np.where(fo == str(PT['early_stopping_fold']))[0]
            tr = np.where(np.isin(fo, [str(x) for x in PT['train_folds']]))[0]
            te = np.zeros(0, np.int64)
        else:
            tr, va, te = DS.fold_rows(D.T, k)
        norm = D.fit_norm(tr)
        D.to_device()
        tseed, oseed = fold_seeds(name, model, k, seed)
        torch.manual_seed(tseed)
        rng = np.random.default_rng(oseed)
        net = new_model(spec, model, D, dev)
        npar = MD.n_params(net)
        lr = float(TR['optimiser']['lr'])
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=float(TR['optimiser']['weight_decay']))
        spe = math.ceil(len(tr) / BS)
        warm = spe   # linear warm-up over the first epoch
        lr_at = schedule(lr, warm, MAXEP * spe)
        scaler = torch.cuda.amp.GradScaler()
        json_dump(dict(dataset=name, model=model, seed=seed, fold=k, torch_seed=tseed, lr=lr, warmup_steps=warm,
                       steps_per_epoch=spe, n_params=npar, channels=D.channels, marks=D.marks, n_train=len(tr),
                       n_val=len(va), n_test=len(te), norm=norm_json(norm), torch=torch.__version__), os.path.join(od, 'config.json'))
        logf = open(os.path.join(od, 'log.txt'), 'w')
        best_auc, best_state, best_ep, bad, step, epochs = -np.inf, None, -1, 0, 0, 0
        yl = D.T.label.values
        for ep in range(MAXEP):
            net.train()
            t0 = time.time()
            batches = DS.make_batches(tr, D.L, BS, rng)
            flips = rng.random(len(D.T)) < 0.5
            if NOFLIP:
                flips[:] = False
            tot_loss, nseen = torch.zeros((), device=dev), 0
            for b in batches:
                train_batch(net, opt, scaler, D, b, flips, lr_at(step), BUDGET, float(TR['grad_clip']), tot_loss)
                nseen += len(b)
                step += 1
            vf, vr = predict_logits(net, D, va, NOFLIP, BS)
            auc = auroc(yl[va], (vf + vr) / 2)
            epochs += 1
            improved = auc > best_auc + 1e-6
            if improved:
                best_auc, best_ep, bad = auc, ep + 1, 0
                best_state = {kk: v.detach().to('cpu', copy=True) for kk, v in net.state_dict().items()}
            else:
                bad += 1
            logf.write(f'epoch {ep + 1} train_loss {float(tot_loss) / nseen:.5f} val_auroc {auc:.5f}{" *" if improved else ""} '
                       f'seconds {time.time() - t0:.1f}\n')
            logf.flush()
            if bad >= PATIENCE:
                break
        net.load_state_dict(best_state)
        val = predict_logits(net, D, va, NOFLIP, BS)
        test = predict_logits(net, D, te, NOFLIP, BS) if len(te) else (np.zeros(0, np.float32), np.zeros(0, np.float32))
        save_outputs(od, D.T, va, None if pretrain else te, yl, val, test, best_state,
                     dict(best_epoch=best_ep, best_val_auroc=best_auc, epochs_run=epochs, n_params=npar))
        logf.close()
        del net, opt, best_state
        torch.cuda.empty_cache()


def train_with_loops(name, model, folds):
    """training on the SSI table and the table of loops without silencers of the dataset (seed rule and settings of configs/ssiformer.yaml
    extra_loops)."""
    import torch
    TR, XL = K.CFG['training'], K.CFG['extra_loops']
    spec = K.MODELS[model]
    NOFLIP = not spec['equivariant']
    seed = int(TR['seed'])
    PATIENCE, BS, MAXEP = int(TR['patience']), int(TR['batch']), int(TR['max_epochs'])
    BUDGET, PBUDGET = memory_budget(XL['memory_budget']), int(XL['prediction_budget'])
    dev = gpu()
    Ds = [DS.SpanData(name, dev), DS.SpanData(name, dev, loops=True)]   # SSI table, table of loops without silencers
    assert Ds[0].marks == Ds[1].marks and Ds[0].channels == Ds[1].channels
    for k in folds:
        od = K.run_dir(name, model, k)
        if os.path.exists(os.path.join(od, 'done.json')):
            print('skip (done)', od)
            continue
        os.makedirs(od, exist_ok=True)
        split, norms = [], []
        for D in Ds:
            split.append(DS.fold_rows(D.T, k))
            norms.append(D.fit_norm(split[-1][0]))
            D.to_device()
        tseed, oseed = fold_seeds(name, model, k, seed)
        torch.manual_seed(tseed)
        rng = np.random.default_rng(oseed)
        net = new_model(spec, model, Ds[0], dev)
        npar = MD.n_params(net)
        lr = float(TR['optimiser']['lr'])
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=float(TR['optimiser']['weight_decay']))
        spe = sum(math.ceil(len(s[0]) / BS) for s in split)
        lr_at = schedule(lr, spe, MAXEP * spe)   # linear warm-up over the first epoch
        scaler = torch.cuda.amp.GradScaler()
        (tr, va, te), (trl, val_l, _) = split
        json_dump(dict(dataset=name, model=model, seed=seed, fold=k, torch_seed=tseed, lr=lr, warmup_steps=spe, steps_per_epoch=spe,
                       n_params=npar, channels=Ds[0].channels, marks=Ds[0].marks, n_train=len(tr), n_train_loops=len(trl),
                       n_val=len(va), n_test=len(te), norm=norm_json(norms[0]), norm_loops=norm_json(norms[1]),
                       torch=torch.__version__), os.path.join(od, 'config.json'))
        logf = open(os.path.join(od, 'log.txt'), 'w')
        best_auc, best_state, best_ep, bad, step, epochs = -np.inf, None, -1, 0, 0, 0
        yl = Ds[0].T.label.values
        for ep in range(MAXEP):
            net.train()
            t0 = time.time()
            plan, flips = [], []
            for i, (D, s) in enumerate(zip(Ds, split)):
                plan += [(i, b) for b in DS.make_batches(s[0], D.L, BS, rng)]
                f = rng.random(len(D.T)) < 0.5
                if NOFLIP:
                    f[:] = False
                flips.append(f)
            plan = [plan[j] for j in rng.permutation(len(plan))]
            tot_loss, nseen = torch.zeros((), device=dev), 0
            for i, b in plan:
                train_batch(net, opt, scaler, Ds[i], b, flips[i], lr_at(step), BUDGET, float(TR['grad_clip']), tot_loss)
                nseen += len(b)
                step += 1
            vf, vr = predict_logits(net, Ds[0], va, NOFLIP, BS, PBUDGET)
            auc = auroc(yl[va], (vf + vr) / 2)   # early stopping on the SSI table
            epochs += 1
            improved = auc > best_auc + 1e-6
            if improved:
                best_auc, best_ep, bad = auc, ep + 1, 0
                best_state = {kk: v.detach().to('cpu', copy=True) for kk, v in net.state_dict().items()}
            else:
                bad += 1
            logf.write(f'epoch {ep + 1} train_loss {float(tot_loss) / nseen:.5f} val_auroc {auc:.5f}{" *" if improved else ""} '
                       f'seconds {time.time() - t0:.1f}\n')
            logf.flush()
            if bad >= PATIENCE:
                break
        net.load_state_dict(best_state)
        val = predict_logits(net, Ds[0], va, NOFLIP, BS, PBUDGET)
        test = predict_logits(net, Ds[0], te, NOFLIP, BS, PBUDGET)
        save_outputs(od, Ds[0].T, va, te, yl, val, test, best_state,
                     dict(best_epoch=best_ep, best_val_auroc=best_auc, epochs_run=epochs, n_params=npar))
        logf.close()
        print(f'{name} {model} fold {k}: best epoch {best_ep}, early-stopping AUROC {best_auc:.4f}, test-fold AUROC '
              f'{auroc(yl[te], sum(test) / 2):.4f}', flush=True)
        del net, opt, best_state
        torch.cuda.empty_cache()


def oof_auroc(name, model):
    """AUROC of the pooled out-of-fold logits when all five folds are done, else None."""
    ps = [os.path.join(K.run_dir(name, model, k), 'oof.npz') for k in range(5)]
    if not all(os.path.exists(p) for p in ps):
        return None
    z = [np.load(p) for p in ps]
    return auroc(np.concatenate([x['label'] for x in z]), np.concatenate([x['logit'] for x in z]))


def main():
    ap = argparse.ArgumentParser(description='train SSIFormer on one dataset (five chromosome folds)')
    ap.add_argument('dataset')
    ap.add_argument('model')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--pretrain', action='store_true')
    ap.add_argument('--extra-loops', action='store_true', help='also train on the loops without silencers of the HiChIP sample')
    args = ap.parse_args()
    model = K.model_key(args.model, args.extra_loops)
    train(args.dataset, model, [int(x) for x in args.folds.split(',')], args.pretrain)
    a = None if args.pretrain else oof_auroc(args.dataset, model)
    if a is not None:
        print(f'{args.dataset} {model}: out-of-fold AUROC of the five folds {a:.4f}')


if __name__ == '__main__':
    main()
