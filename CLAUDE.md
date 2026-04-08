# CLAUDE.md — ms-pred (PyG Version)

## Project Overview

Mass spectrum predictor toolkit: predicts tandem mass spectra (MS/MS) from molecular structures.

**Primary models:** ICEBERG, MARASON, SCARF
**Baselines:** NEIMS-FFN, NEIMS-GNN, MassFormer, 3DMolMS, GrAFF-MS
**Datasets:** NIST20 (commercial) and MassSpecGym (open-source)

---

## Environment & Installation

```bash
mamba env create -f environment.yml
mamba activate ms-gen
pip install -r requirements.txt
python3 setup.py develop   # compiles MassFormer Cython modules
```

Key versions: Python 3.8, PyTorch 1.9.1+cu111, PyTorch Lightning 1.6, RDKit 2021.03, DGL 0.8.2

---

## Repository Structure

```
src/ms_pred/
  common/          # shared chem/data utilities (mol graphs, spectra, formulas)
  nn_utils/        # GNN layers, formula embedder, transformer blocks
  dag_pred/        # ICEBERG (fragment DAG generator + intensity predictor)
  marason/         # MARASON (retrieval-augmented intensity prediction)
  scarf_pred/      # SCARF (formula-level prediction)
  ffn_pred/        # NEIMS-FFN baseline
  gnn_pred/        # NEIMS-GNN baseline
  massformer_pred/ # MassFormer baseline
  graff_ms/        # GrAFF-MS baseline
  molnetms/        # 3DMolMS baseline
  autoregr_gen/    # autoregressive generator
  magma/           # MAGMA fragmentation engine
  retrieval/       # retrieval benchmarking + bootstrap metrics

configs/           # OmegaConf YAML configs per model
run_scripts/       # numbered shell/Python scripts per pipeline step
data_scripts/      # preprocessing (MAGMA, subformula, splits, PubChem)
launcher_scripts/  # experiment launcher + SLURM helpers
analysis/          # evaluation metric scripts
notebooks/         # Jupyter demos (ICEBERG, MARASON, SCARF)
data/              # spec_datasets/, pubchem/, exp_specs/
```

---

## Architecture Patterns

- **Two-stage training**: Stage 1 = fragment generator, Stage 2 = intensity predictor
- All models subclass `pl.LightningModule`
- Graph NNs via DGL (GGNN, PNA, GraphSAGE); molecular graphs constructed with RDKit
- Config management via OmegaConf YAML
- Ray Tune for distributed hyperparameter search

---

## Key Entry Points

### ICEBERG (primary model)
```bash
# Full pipeline
bash run_scripts/iceberg/run_all.sh

# Stage 1: fragment DAG generator
python src/ms_pred/dag_pred/train_gen.py

# Stage 2: intensity predictor
python src/ms_pred/dag_pred/train_inten.py

# Inference (SMILES → spectrum)
python src/ms_pred/dag_pred/predict_smis.py

# Retrieval evaluation
python run_scripts/iceberg/06_run_retrieval.py
```

### General
```bash
# Launch from config
python launcher_scripts/run_from_config.py <config.yaml>

# Evaluate predictions
python analysis/spec_pred_eval.py
```

---

## Data Pipeline

```
NIST20 / MassSpecGym
  → MAGMA fragmentation (generates fragment DAGs)
  → subformula assignment
  → binned spectrum conversion (15,000 m/z bins, 0–1500 Da)
  → dataset splits
  → training
```

Data preprocessing scripts live in `data_scripts/` and are numbered by step.

---

## Testing & Validation

No formal test suite. Validation uses:
- Evaluation scripts in `analysis/` (cosine similarity, top-k retrieval)
- Bootstrap confidence intervals: `retrieval/bootstrap_metrics.py`
- Demo notebooks in `notebooks/` for end-to-end sanity checks
