"""Paths and settings. Every path can be moved with an environment variable:
SSIFORMER_BENCHMARK (dataset tables; default <repo>/benchmark), SSIFORMER_WEIGHTS (trained models; default <repo>/weights),
SSIFORMER_CACHE (model inputs; default <repo>/cache), SSIFORMER_RUNS (outputs of local runs; default <repo>/runs),
SSIFORMER_RAW (raw downloads for the construction of datasets and inputs; default <repo>/raw).
A new sample (ssiformer.sample) is registered at run time with register_sample(): its tables, inputs and runs live in its own
folder."""
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(REPO, 'configs', 'ssiformer.yaml')))

BENCH = os.environ.get('SSIFORMER_BENCHMARK', os.path.join(REPO, 'benchmark'))
WEIGHTS = os.environ.get('SSIFORMER_WEIGHTS', os.path.join(REPO, 'weights'))
CACHE = os.environ.get('SSIFORMER_CACHE', os.path.join(REPO, 'cache'))
RUNS = os.environ.get('SSIFORMER_RUNS', os.path.join(REPO, 'runs'))
RAW = os.environ.get('SSIFORMER_RAW', os.path.join(REPO, 'raw'))

IT = list(CFG['independent_test_datasets'])   # independent-test datasets (not in MOUSE / HUMAN)
DATASETS = dict(CFG['datasets'], **CFG['independent_test_datasets'])
MODELS = CFG['models']
MARKS = list(CFG['marks'])
RES = int(CFG['resolution'])
FLANK = int(CFG['input']['flank_tokens'])
MOUSE, HUMAN = list(CFG['mouse_datasets']), list(CFG['human_datasets'])
ALL_CHROMS = {g: list(v['chroms']) for g, v in CFG['genomes'].items()}
SEALED = {g: set(v) for g, v in CFG['sealed_chromosomes'].items()}
SAMPLES = {}   # name of a registered new sample -> its folder


def dataset(name):
    d = dict(DATASETS[name])
    d['name'] = name
    d['marks'] = [m for m in MARKS if m != d['drop']]
    return d


def register_sample(name, root, genome, drop):
    """make a new sample usable like a dataset: tables in <root>/tables, inputs in <root>/cache, runs in <root>/runs."""
    DATASETS[name] = dict(cell=name, genome=genome, label=None, drop=drop)
    SAMPLES[name] = root


def table_path(name, sealed=False, loops=False):
    """SSI table of a dataset (cross-validation or held-out chromosomes) or its table of loops without silencers (loops=True)."""
    if name in SAMPLES:
        return os.path.join(SAMPLES[name], 'tables', 'loops.tsv.gz' if loops else 'ssi.tsv.gz')
    return os.path.join(BENCH, 'loops' if loops else 'sealed' if sealed else 'tables', f'{name}.tsv.gz')


def cache_dir(name, sealed=False):
    """folder of the model inputs of a dataset."""
    if name in SAMPLES:
        return os.path.join(SAMPLES[name], 'cache')
    return os.path.join(CACHE, 'sealed') if sealed else CACHE


def run_dir(name, model, fold, sealed=False):
    root = os.path.join(SAMPLES[name], 'runs') if name in SAMPLES else RUNS
    return os.path.join(root, 'sealed' if sealed else 'cv', name, model, f'fold{fold}')


def fold_weights(name, model, fold):
    """(weights, json with 'norm') of one fold model: a local training run if present, else the trained model of the repository."""
    od = run_dir(name, model, fold)
    if os.path.exists(os.path.join(od, 'best.pt')):
        return os.path.join(od, 'best.pt'), os.path.join(od, 'config.json')
    w = os.path.join(WEIGHTS, name, model)
    return os.path.join(w, f'fold{fold}.pt'), os.path.join(w, f'fold{fold}.norm.json')


def pretrained_weights(source_model):
    """initial weights of a '*_pretrained' model: a local pretraining run if present, else the mESC H3K27ac model of the repository."""
    p = os.path.join(RUNS, 'pretrain', source_model, 'best.pt')
    return p if os.path.exists(p) else os.path.join(WEIGHTS, 'pretrain', f'{source_model}.pt')


def model_key(model, extra_loops=False):
    """'--extra-loops' selects the version of a model trained with the loops without silencers of the sample (key '<model>_loops')."""
    if extra_loops and not MODELS[model].get('extra_loops'):
        model = f'{model}_loops'
    assert model in MODELS, f'unknown model {model}'
    return model
