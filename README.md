# Protein Mutation Impact Prediction

A machine learning pipeline for predicting the functional impact of mutations in **Green Fluorescent Protein (GFP)**, using embeddings and zero-shot scores from the ESM2 protein language model.

---

## Overview

This project predicts how mutations affect GFP fluorescence, using the full [ProteinGym](https://proteingym.org/) deep mutational scan (Sarkisyan et al., 2016): 51,714 variants (both single and combinatorial multi-mutants) with a continuous experimental fitness score.

The pipeline uses ESM2 in two ways:

- **Delta embeddings** (mutant − wild-type, mean-pooled) — the input to a supervised Random Forest regressor trained on the labeled assay. This is the model.
- **A zero-shot score** (masked-marginal log-odds) — no training data required, computed directly from the language model's per-position amino-acid preferences. Used as a baseline: adding it as an extra regressor input was tested and made no difference (Spearman 0.6398 with it vs 0.6405 without), so the final model leaves it out.

The primary target is the **continuous DMS fitness score**, evaluated by **Spearman correlation** — the standard ProteinGym evaluation metric — rather than a thresholded binary label.

---

## Results

| Model | Spearman ρ (test set, n=10,343) |
|---|---|
| Zero-shot ESM2 (masked-marginal, no training) | 0.140 |
| Baseline: number of mutations (more mutations → dimmer) | 0.550 |
| Supervised RF regressor (delta embeddings) | **0.641** |

**The mutation-count baseline matters.** GFP tolerates a few mutations and then abruptly stops folding: the fraction of variants that still fluoresce drops from 92% with one mutation, to 54% with four, to under 1% with ten (threshold-like epistasis). So simply counting mutations already ranks variants well, and the headline 0.641 is only ~0.09 above that. The real test of what the model learned is **within a fixed number of mutations**, where counting can't help:

| Mutations per variant | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Regressor Spearman ρ | 0.02 | 0.38 | 0.43 | 0.45 | 0.41 | 0.23 |

The model captures real signal about *which* combinations break GFP (ρ ≈ 0.4 for 2–5 mutations), but **it has learned essentially nothing about single mutants** — the hardest and most useful case.

A binary classifier (functional vs. non-functional) on the same features reaches AUROC 0.888. That is *not* directly comparable to the original project's AUROC 0.553, which was measured on single mutants only.

### Project history

The first version trained a Random Forest classifier on only the 1,084 single-substitution variants — 91.6% of which still fluoresce. It reported 89–92% accuracy but AUROC 0.55 (near-random): accuracy on a 91.6%-majority dataset reflects the base rate, not learning. Its decision threshold was also tuned on the test set, and its write-up had the labels inverted (calling the functional majority "deleterious").

The rework used all 51,714 variants (the original filtering discarded 98% of the assay), predicted the continuous fitness score instead of a thresholded label, scored models by Spearman correlation (ProteinGym's standard metric), and added the zero-shot and mutation-count baselines to show what the supervised model actually adds.

### Limitations and next steps

- **Single mutants are unsolved.** Mean-pooling over 238 positions dilutes a one-residue change to ~1/238 of the embedding; using the embedding at the mutated position(s), or a larger ESM2 checkpoint, are the obvious next experiments.
- **Random split is optimistic.** Test multi-mutants often share individual mutations with training variants. Holding out whole positions (as ProteinGym's stricter supervised splits do) would give a harder, more honest estimate.
- Single train/test split (no cross-validation), and the zero-shot score treats mutations as independent, which GFP's threshold behavior violates.

Both models are trained on a 15,000-variant random subsample of the 41,371-row training split (`MAX_TRAIN_SAMPLES` in `src/config.py`) with shallow trees (`max_depth=15`), to keep a full training run to a few minutes on a laptop CPU. The full test set (10,343 variants) is always used for evaluation.

---

## Pipeline

```
scripts/run_pipeline.py
├─ src/features/build_features.py
│   ├─ src/data/load_data.py        Load the GFP assay from ProteinGym
│   ├─ src/features/embeddings.py   Delta ESM2 embeddings (mutant - WT)
│   └─ src/features/zero_shot.py    Masked-marginal zero-shot scores
├─ src/models/split.py           Stratified train/test split
├─ src/models/regressor.py       Primary model: RF regressor on continuous score
├─ src/models/classifier.py      Ablation: RF classifier on binary label
├─ src/models/validate.py        Evaluation plots
└─ src/reporting/report.py       Text summary report
```

Each pipeline stage is also runnable standalone via its `scripts/*.py` entrypoint.

---

## Usage

```bash
pip install -r requirements.txt
pip install -e .

python scripts/run_pipeline.py          # full pipeline, end to end

# or individual stages:
python scripts/build_features.py
python scripts/split_data.py
python scripts/train_regressor.py
python scripts/predict.py --mutations A206T S65T Y66H
```

---

## Tech Stack

- Python, PyTorch, [fair-esm](https://github.com/facebookresearch/esm) (ESM2-t12-35M, 480-dim embeddings)
- scikit-learn (Random Forest regression/classification, AUROC)
- pandas, numpy, scipy (Spearman correlation)
- matplotlib for evaluation plots

---

## Dataset

- **Source:** ProteinGym DMS substitutions (Sarkisyan et al., 2016)
- **Protein:** Green Fluorescent Protein (GFP), 238 residues
- **Variants:** 51,714 (1,084 single substitutions, up to 15 combined mutations per variant)
- **Target:** continuous DMS fitness score (log fluorescence, 1.28–4.12), plus ProteinGym's binary label (functional if score ≥ 2.5) for the classifier ablation
