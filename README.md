# Protein Mutation Impact Prediction

A machine learning pipeline for predicting the functional impact of mutations in **Green Fluorescent Protein (GFP)**, using embeddings and zero-shot scores from the ESM2 protein language model.

---

## Overview

This project predicts how mutations affect GFP fluorescence, using the full [ProteinGym](https://proteingym.org/) deep mutational scan (Sarkisyan et al., 2016): 51,714 variants (both single and combinatorial multi-mutants) with a continuous experimental fitness score.

The pipeline combines two complementary signals from ESM2:

- **A zero-shot score** (masked-marginal log-odds) — no training data required, computed directly from the language model's per-position amino-acid preferences.
- **Delta embeddings** (mutant − wild-type, mean-pooled) — fed into a supervised Random Forest regressor trained on the labeled assay.

The primary target is the **continuous DMS fitness score**, evaluated by **Spearman correlation** — the standard ProteinGym evaluation metric — rather than a thresholded binary label.

---

## Results

| Model | Spearman ρ (test set, n=10,343) |
|---|---|
| Zero-shot ESM2 (masked-marginal, no training) | 0.140 |
| Supervised RF regressor (delta embeddings + zero-shot) | **0.640** |

For reference, a binary classifier (tolerated/deleterious) trained the same way reaches **AUROC 0.890** / AUPRC 0.906 — up from AUROC 0.553 (near-random) in the original single-mutant-only version.

### Why this framing, and what changed

The first version of this project trained a Random Forest **classifier** on a binary tolerated/deleterious label, using only the 1,084 single-substitution variants in the assay (91.6% one class). That model reached **AUROC 0.55** — barely above random guessing — despite reporting 89–92% accuracy, because accuracy on a 91.6%-majority-class dataset is dominated by the base rate, not by anything the model learned.

Two changes fixed this:

1. **Used the full dataset.** The raw assay contains 51,714 variants, including combinatorial mutants (up to 15 substitutions); the original pipeline discarded 98% of it by filtering to single substitutions only. Multi-mutant sequences embed exactly the same way as single mutants, so no new data collection was needed — just using what was already downloaded.
2. **Matched the evaluation to the data.** GFP fluorescence is a continuous, highly non-linear function of stability, and ProteinGym itself evaluates DMS models by Spearman correlation on the continuous score, not accuracy on an arbitrarily thresholded binary label. Reframing as regression removed the class-imbalance artifact entirely.

The zero-shot ESM2 score alone stays weak on this particular assay (Spearman well under 0.2) — GFP brightness is known to be a near-binary function of folding stability, which pure language-model likelihood doesn't capture well. The supervised regressor, trained on real mutant embeddings rather than an additive approximation, is what drives the improvement.

Both models are trained on a 15,000-variant random subsample of the 41,371-row training split (`MAX_TRAIN_SAMPLES` in `src/models/regressor.py` / `classifier.py`) with shallow trees (`max_depth=15`), to keep a full training run to a few minutes on a laptop CPU. The full training set and deeper trees were tested and did not change the qualitative result.

---

## Pipeline

```
scripts/run_pipeline.py
├─ src/data/load_data.py        Load & validate the GFP assay from ProteinGym
├─ src/features/build_features.py
│   ├─ src/features/embeddings.py   Delta ESM2 embeddings (mutant - WT)
│   └─ src/features/zero_shot.py    Masked-marginal zero-shot scores
├─ src/models/split.py           Stratified train/test split
├─ src/models/regressor.py       Primary model: RF regressor on continuous score
├─ src/models/classifier.py      Comparison baseline: RF classifier on binary label
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
- scikit-learn (Random Forest regression/classification), imbalanced-learn (SMOTE, for the comparison classifier)
- pandas, numpy, scipy (Spearman/Pearson correlation)
- matplotlib for evaluation plots

---

## Dataset

- **Source:** ProteinGym DMS substitutions (Sarkisyan et al., 2016)
- **Protein:** Green Fluorescent Protein (GFP), 238 residues
- **Variants:** 51,714 (1,084 single substitutions, up to 15 combined mutations per variant)
- **Target:** continuous DMS fitness score (log fluorescence), plus a binary tolerated/deleterious label for comparison
