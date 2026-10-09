# SSIFormer: a loop-domain Transformer predicts silencer–silencer interactions across cell types and species

SSIFormer predicts silencer–silencer interactions (SSIs) from the whole loop domain between two silencer anchors. A
shared multi-scale anchor encoder reads the histone marks around each anchor; a Transformer with rotary position
embeddings reads the 1-kb loop-domain tokens (histone marks, CTCF motif strength and count on each strand, REST, YY1 and
ZNF143 motifs, silencer loci, anchor indicator); a symmetric readout combines both anchors, the loop domain and the
genomic distance. A pair is presented in a random orientation during training, and the prediction is the mean over both
orientations. On mouse datasets SSIFormer is trained from scratch; on human datasets it is pretrained on mESC H3K27ac
and fine-tuned. For a human sample with HiChIP loops, the paper recommends the extended configuration: the
mouse-pretrained model fine-tuned on the SSIs and the loops without silencers of the sample (evaluated by cross-validation).

This repository contains the source code of SSIFormer, the trained models of the paper, the SSI datasets and the
workflow for a new sample.

## Contents

| Path | Content |
|---|---|
| `ssiformer/model.py`, `dataset.py` | the model and its inputs |
| `ssiformer/train.py` | training, pretraining and fine-tuning; `--extra-loops`: training with the loops without silencers of the sample |
| `ssiformer/predict.py` | scoring of the pairs of a dataset with trained models, CTCF strand exchange |
| `ssiformer/sample.py` | a new sample: candidate pairs, model inputs, scores, training on the sample |
| `ssiformer/build.py` | construction of the SSI datasets, of the tables of loops without silencers and of the model inputs from the raw files |
| `configs/` | datasets, model and training settings (`ssiformer.yaml`); construction rules of the datasets (`benchmark.yaml`); template of a new sample (`sample_example.yaml`) |
| `weights/` | trained models: five fold models per dataset and model (`fold<k>.pt`, fp16, with the input normalisation `fold<k>.norm.json`), the mESC H3K27ac model used for pretraining (`pretrain/ssiformer.pt`); `SHA256SUMS` |
| `benchmark/tables/` | SSI datasets: pairs of 10-kb bins with as many matched negatives as SSIs (`chrom, a, b, d, label, fold`) |
| `benchmark/sealed/` | the same for the held-out chromosomes (human chr8 and chr17; mouse chr4, chr12 and chr19); HeLa-S3 CTCF has no held-out table |
| `benchmark/loops/` | tables of the loops without silencers of the HiChIP sample of each human dataset (no silencer in either anchor bin; with matched negatives; same columns and folds) |
| `benchmark/folds_*.tsv`, `benchmark/universe/` | chromosome folds; silencer loci |
| `raw/` | JASPAR matrices and the list of the public raw files of the benchmark (`sources.tsv`: URL, size and, where recorded, md5) |

Datasets: `mESC_H3K27me3`, `mESC_H3K27ac`, `GM12878_H3K27ac`, `GM12878_SMC1A`, `GM12878_CTCF`, `K562_H3K27ac`,
`HeLaS3_CTCF`, and the independent-test datasets `IT_mESC_SMC1A`, `IT_H9_RAD21`, `IT_H9_SMC1A`. Models:

| Model | Training | Trained models in `weights/` |
|---|---|---|
| `ssiformer` | the SSIs of the dataset alone | the mouse datasets (the model of the paper for mouse) |
| `ssiformer_pretrained` | pretrained on mESC H3K27ac, fine-tuned on the SSIs of the dataset | the human datasets (the standard configuration, used for all comparisons and the independent test) |
| `ssiformer_pretrained --extra-loops` | pretrained on mESC H3K27ac, fine-tuned on the SSIs and the loops without silencers of the sample (key `ssiformer_pretrained_loops`) | the five human datasets (the extended configuration) |
| `ssiformer --extra-loops` | the SSIs and the loops without silencers of the sample, without pretraining (key `ssiformer_loops`) | none (`ssiformer.train`) |

In file and option names of the code, `sealed` means the held-out chromosomes.

## Installation

```bash
git clone https://github.com/shenlab-ahmu/SSIFormer
cd SSIFormer
conda env create -f environment.yml      # Python 3.10, PyTorch 2.0.0 (CUDA 11.8) and the other packages
conda activate ssiformer
```

The code needs an NVIDIA GPU (the models of the paper were trained and applied on RTX 3090 and RTX 3070 cards with
Python 3.10 and PyTorch 2.0.0). The construction of datasets and inputs (`ssiformer.build`, `ssiformer.sample prepare`
and `inputs`) runs on the CPU.

## Model inputs

The model inputs of the datasets (1-kb histone-mark and motif tracks and the anchor tensors) are release assets. Unzip
them in the repository root; they create the folder `cache/`.

| File | Content |
|---|---|
| `ssiformer_inputs.zip` (282 MB) | the inputs of all datasets |
| `ssiformer_inputs_loops.zip` (60 MB) | anchor tensors of GM12878, K562 and HeLa-S3 that also contain the anchor bins of the tables of loops without silencers; unzip after `ssiformer_inputs.zip` and replace the three files (needed for training with `--extra-loops`; the inputs of every other table are unchanged) |

```bash
curl -L -O https://github.com/shenlab-ahmu/SSIFormer/releases/download/v1.2.0/ssiformer_inputs.zip && unzip ssiformer_inputs.zip
curl -L -O https://github.com/shenlab-ahmu/SSIFormer/releases/download/v1.2.0/ssiformer_inputs_loops.zip && unzip -o ssiformer_inputs_loops.zip
```

(or download the files from the release pages of the repository).

## Quick start

After the installation and the model inputs:

```bash
python -m ssiformer.predict K562_H3K27ac ssiformer_pretrained                 # out-of-fold scores of the K562 H3K27ac table (AUROC 0.841)
python -m ssiformer.predict K562_H3K27ac ssiformer_pretrained --extra-loops   # the same with the extended configuration (AUROC 0.857)
```

The scores are written to `runs/scores/`. For your own sample, see [Using SSIFormer on a new sample](#using-ssiformer-on-a-new-sample).

## Scoring pairs with the trained models

```bash
python -m ssiformer.predict GM12878_CTCF ssiformer_pretrained            # cross-validation table, out-of-fold scores
python -m ssiformer.predict GM12878_CTCF ssiformer_pretrained --extra-loops   # extended configuration
python -m ssiformer.predict IT_H9_RAD21 ssiformer_pretrained --heldout   # held-out chromosomes, mean of the five fold models
python -m ssiformer.predict HeLaS3_CTCF ssiformer_pretrained --table pairs.tsv --strand-exchange --out scores.tsv
```

The output is a table with the logit and the probability of every pair (default `runs/scores/`). With labels in the
table, AUROC and AUPRC are printed. `--table` takes any pairs of the same cell type (columns `chrom`, `a`, `b`: 10-kb
bin indices with `a < b`); the anchor bins must be in the anchor-tensor file of the cell type. These pairs are scored by the
mean of the five fold models, so scores of pairs that are in the training tables are not out-of-fold; pairs on the held-out
chromosomes need `--heldout` and a table of their own. `--strand-exchange` adds
the change of the logit when the CTCF plus- and minus-strand motif channels are exchanged over the whole region; a more
negative value means a stronger dependence of the pair on the CTCF orientation.

Because the models were trained on tables with one negative per SSI, the scores are best used to rank candidate pairs
(the partners of one anchor, or all pairs of a window) rather than with a fixed threshold.

Expected output with the trained models of this repository (`predict` stands for `python -m ssiformer.predict`):

| Dataset | Command | AUROC |
|---|---|---|
| mESC H3K27me3 | `predict mESC_H3K27me3 ssiformer` | 0.955 |
| mESC H3K27ac | `predict mESC_H3K27ac ssiformer` | 0.941 |
| GM12878 H3K27ac | `predict GM12878_H3K27ac ssiformer_pretrained` | 0.888 |
| GM12878 SMC1A | `predict GM12878_SMC1A ssiformer_pretrained` | 0.927 |
| GM12878 CTCF | `predict GM12878_CTCF ssiformer_pretrained` | 0.902 |
| K562 H3K27ac | `predict K562_H3K27ac ssiformer_pretrained` | 0.841 |
| HeLa-S3 CTCF | `predict HeLaS3_CTCF ssiformer_pretrained` | 0.863 |
| mESC SMC1A (independent test) | `predict IT_mESC_SMC1A ssiformer --heldout` | 0.946 |
| H9 hESC RAD21 (independent test) | `predict IT_H9_RAD21 ssiformer_pretrained --heldout` | 0.902 |
| H9 hESC SMC1A (independent test) | `predict IT_H9_SMC1A ssiformer_pretrained --heldout` | 0.952 |

Extended configuration (cross-validation; `predict <dataset> ssiformer_pretrained --extra-loops`):

| Dataset | Loops without silencers in training | AUROC |
|---|---|---|
| GM12878 H3K27ac | 28,603 | 0.933 |
| GM12878 SMC1A | 21,425 | 0.958 |
| GM12878 CTCF | 17,254 | 0.938 |
| K562 H3K27ac | 29,800 | 0.857 |
| HeLa-S3 CTCF | 8,532 | 0.886 |

## Training

Every model uses seed 0 and five chromosome folds (test fold k, early stopping on fold (k + 1) mod 5, training on the
other three). Outputs go to `runs/cv/<dataset>/<model>/fold<k>/`; for each fold, `ssiformer.predict` uses the local run of
that fold when it exists and the trained model of `weights/` otherwise. The pooled out-of-fold AUROC is printed when
all five folds are done.

```bash
python -m ssiformer.train mESC_H3K27ac ssiformer               # a mouse dataset, from scratch
python -m ssiformer.train GM12878_CTCF ssiformer_pretrained    # a human dataset: fine-tuning of weights/pretrain/ssiformer.pt
python -m ssiformer.train GM12878_CTCF ssiformer_pretrained --extra-loops   # extended configuration (needs ssiformer_inputs_loops.zip)
python -m ssiformer.train GM12878_CTCF ssiformer --extra-loops              # SSIs and loops without silencers, without pretraining
python -m ssiformer.train mESC_H3K27ac ssiformer --pretrain    # the pretraining itself (folds 1-4 train, fold 0 early stopping)
```

With `--extra-loops`, the training folds of the SSI table and of the table of loops without silencers of the same HiChIP sample
(`benchmark/loops/<dataset>.tsv.gz`) are used in every epoch; batches are drawn within one table and interleaved in random
order, each table keeps its own input normalisation, and early stopping and testing use the SSI table only. These
tables hold loops without a silencer-locus midpoint in either anchor bin, on the cross-validation chromosomes, 20 kb to
2 Mb apart, up to ten times the SSIs of the dataset on these chromosomes before negative matching and at most 30,000,
with negatives drawn 1:1 from pairs of their anchor bins by the rules of the SSI table (`configs/benchmark.yaml`,
`other_loops`).

Retraining on another GPU or PyTorch version can change the last digits. On an RTX 3070, the five folds of the extended
configuration take about 20 min (HeLa-S3 CTCF) to 80 min (GM12878 H3K27ac); `SSIFORMER_MEMORY_BUDGET` lowers the
micro-batch size on a GPU with less memory (same optimisation steps).

## Using SSIFormer on a new sample

SSIFormer needs a silencer annotation (BED), the six histone-modification tracks of the model (bigWig: H3K4me3, H3K4me1,
H3K9ac, H3K36me3, H3K9me3 and H3K27me3 for the human models) and the genome (hg38 or mm10; the motif channels of both
genomes are in the model inputs). HiChIP loops of the sample are optional: they define the candidate pairs used in the
paper and allow training on the sample. Copy `configs/sample_example.yaml`, set the paths and run:

```bash
python -m ssiformer.sample my_sample.yaml prepare             # silencer loci, candidate pairs (+ training tables with loops)
python -m ssiformer.sample my_sample.yaml inputs 2            # model inputs from the bigWig files (CPU, 2 processes)
python -m ssiformer.sample my_sample.yaml score --strand-exchange      # scores with the released models (GPU)
python -m ssiformer.sample my_sample.yaml train               # optional, with loops: extended configuration on the sample
python -m ssiformer.sample my_sample.yaml score --model <name>/ssiformer_pretrained_loops --strand-exchange
```

1. **Candidate pairs.** Silencer elements are merged into loci (overlap of at least 1 bp; locus = midpoint). Candidate
   pairs are pairs of 10-kb bins on one chromosome, 20 kb to 2 Mb apart, whose two bins are candidate anchors: with
   loops, the bins that anchor HiChIP SSIs (loops with a silencer-locus midpoint in both anchor bins; all evaluations of
   the paper used such candidates); without loops, every bin with a silencer-locus midpoint.
2. **Inputs.** 1-kb tracks of the marks (log1p of the mean signal per kb) and of the silencer loci, motif channels and the
   anchor tensors of all candidate bins, computed exactly as for the datasets of the paper.
3. **Scores.** Each candidate pair is scored by the mean of the five fold models of the chosen model (`model` in the sample
   file, e.g. `K562_H3K27ac/ssiformer_pretrained`; for a cell line with few SSIs, the model of another cell line
   profiled with the same HiChIP target), each with the input normalisation it was trained with, averaged over both
   orientations of the pair. The output `samples/<name>/scores.tsv` has one row per pair: `chrom, a, b` (bin indices), `start_a, start_b, distance`,
   `hichip_ssi` (1 for a HiChIP SSI), `logit`, `probability`, `percentile_a` and `percentile_b` (the percentage of the
   candidate pairs of anchor bin a, resp. b, scored at or below the pair, i.e. its rank among the partners of each
   anchor), and with `--strand-exchange` `strand_exchange_change` (the change of the logit when the CTCF strand
   channels are exchanged; more negative = stronger dependence on the CTCF orientation). Use the scores to rank
   candidates, for example the partners of one anchor or all pairs in a window, rather than with a fixed threshold.
4. **Training on the sample (optional, with loops).** `prepare` also builds the SSI table and the table of loops without silencers of
   the sample by the rules of the benchmark (folds and held-out chromosomes of `benchmark/folds_<genome>.tsv`; loops of
   further samples of the cell can be excluded from the negatives with `exclusion`). `train` fits the extended
   configuration (mouse pretraining, then fine-tuning on the SSIs and the loops without silencers; `--from-scratch` without
   pretraining) and prints the out-of-fold AUROC on the SSI table of the sample; `score --model
   <name>/ssiformer_pretrained_loops` then scores the candidates with these models. For a human sample with HiChIP loops
   this is the configuration the paper recommends; it was evaluated by cross-validation.

## Expected runtimes

On an NVIDIA RTX 3070 (8 GB; PyTorch 2.0.0, Linux under WSL2), scoring 10,000 pairs of a human table (GM12878 H3K27ac
pairs, 20 kb to 2 Mb, median 370 kb, drawn with replacement) with the five fold models, both orientations of every pair,
took 99 s for the whole command (`python -m ssiformer.predict ... --table`, including start-up and reading the inputs
from the cache, 3 s), about 500 pairs per second per fold model, and 367 s with `--strand-exchange` (about 140 pairs per
second per fold model); peak GPU memory 2.1 GB. Times vary with the load of the computer (scoring alone took 61 to 325 s
in repeated runs).
The out-of-fold scores of a benchmark dataset take 6 to 25 s. Training the five folds takes about 20 to 80 min per human dataset
with `--extra-loops` (above). The inputs of a new sample take about 1 min per 500 Mb of chromosomes with two CPU
processes (69 s for six human chromosomes with 469 Mb, below 0.3 GB of memory), about 8 min for a whole human genome.

## Datasets and inputs from the raw files

`ssiformer/build.py` builds the SSI tables, the tables of loops without silencers and the model inputs from public files (loop calls
of the Loop Catalog, SilencerDB elements, ENCODE histone-modification bigWigs, the hg38 and mm10 genome sequences, JASPAR
matrices); the rules are in its docstring and in `configs/benchmark.yaml`, and `raw/sources.tsv` lists the files of the
seven benchmark datasets with their URL (expected under `raw/`, e.g. `raw/loops/`, `raw/bigwig/<cell>/`, `raw/fa/`).
Outputs go to `build/`.

```bash
python -m ssiformer.build tables                 # silencer loci, folds, SSIs, cross-validation and held-out tables of the benchmark datasets
python -m ssiformer.build loops                  # tables of loops without silencers of the five human datasets
python -m ssiformer.build genome hg38 16         # 1-kb motif channels (also mm10); 16 = number of processes
python -m ssiformer.build marks GM12878 16       # 1-kb mark tracks and anchor tensors of the SSI and loop-table anchor bins (also K562, HeLa-S3, mESC)
```

The tables and inputs of the independent-test datasets are provided (`benchmark/`, release assets).

## Citation

SSIFormer: a loop-domain Transformer predicts silencer–silencer interactions across cell types and species
(manuscript under review).

## License

MIT (see `LICENSE`). Third-party resources keep their own licences: the Loop Catalog, SilencerDB, ENCODE and JASPAR.
