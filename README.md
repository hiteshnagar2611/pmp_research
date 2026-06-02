# PMP Research: DynaMo + PMPGen

A complete PyTorch implementation of **DynaMo** (Dynamic Membrane Oracle) for **Phase 1** peripheral membrane protein (PMP) binding residue prediction, and **PMPGen** for **Phase 2** de novo PMP generation.

Designed to target **NeurIPS / ICLR / ICML** with novel contributions in:
- Conformational ensemble-aware attention using MD trajectories
- Membrane geometry-guided cross-attention fusion
- SE(3) flow matching with MD-informed anisotropic noise schedule
- Membrane plane gradient guidance during diffusion sampling

---

## Project Structure

```
pmp_research/
├── pyproject.toml              # dependencies, build config
├── requirements.txt            # pip install -r
├── Makefile                    # shortcuts: make train-phase1, make test
├── configs/                    # Hydra YAML configs
│   ├── data/
│   ├── model/
│   └── train/
├── data/
│   ├── raw/                    # PDB, OPM, MD trajectories, labels
│   ├── processed/              # cached features, graphs
│   └── splits/                 # train/val/test protein lists
├── src/
│   ├── data/
│   │   ├── graph_builder.py    # kNN residue graph construction
│   │   ├── feature_extractor.py # scalar + vector per-residue features
│   │   ├── pmp_dataset.py      # PyG Dataset class
│   │   └── ...
│   ├── models/
│   │   ├── shared/             # SE(3) utils, GVP primitives
│   │   ├── dynamo/             # Phase 1 model
│   │   │   ├── dynamo.py
│   │   │   ├── fusion.py       # PLM + structure fusion
│   │   │   ├── conf_attention.py # MD ensemble pooling
│   │   │   ├── cross_attention.py # structure-dynamics fusion
│   │   │   ├── geometry_path.py # membrane encoder
│   │   │   └── phys_gate.py    # physicochemical gating
│   │   └── pmpgen/             # Phase 2 model
│   ├── training/
│   │   ├── trainer_phase1.py
│   │   ├── losses.py           # focal, patch, contrastive, flow
│   │   └── metrics.py
│   ├── evaluation/
│   │   ├── benchmark_phase1.py
│   │   └── ablation.py
│   └── utils/
├── scripts/
│   ├── preprocess.py           # build graphs, cache features
│   ├── train_phase1.py         # entry point
│   ├── train_phase2.py
│   └── generate.py
├── tests/
│   ├── test_equivariance.py    # critical: verify SE(3) invariance
│   ├── test_gvp_shapes.py
│   └── test_losses.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 03_attention_maps.ipynb # interpretability
│   └── ...
└── outputs/                    # git-ignored, generated files
    ├── checkpoints/
    ├── generated_pmps/
    └── figures/
```

---

## Installation

```bash
git clone <repo>
cd pmp_research
pip install -e ".[dev]"
make install

# Copy and edit config
cp .env.example .env
# Edit: WANDB_API_KEY, DATA_ROOT, etc.
```

---

## Quick Start

### 1. Preprocess Data

```bash
make preprocess
```

Builds:
- kNN residue graphs
- Caches ESM-2 embeddings
- Computes RMSF from MD trajectories
- Extracts structural features

### 2. Train Phase 1 (DynaMo)

```bash
make train-phase1
```

Trains binding residue predictor with:
- Conformational attention on T MD snapshots
- Membrane geometry path
- Cross-attention fusion
- Physicochemical gating

Monitor on Weights & Biases: `https://wandb.ai/your-entity/pmp-research`

### 3. Generate Phase 2 (PMPGen)

```bash
make generate
```

Uses trained DynaMo to condition de novo generation via:
- SE(3) flow matching with MD-informed noise schedule
- Membrane plane gradient guidance
- ProteinMPNN sequence decoder
- 3-stage validation cascade

---

## Key Features

### Phase 1: DynaMo

**Novel Components:**
- **Conformational Attention Pool**: Pools T MD snapshots using RMSF-adaptive temperature
  - High RMSF → broad attention over conformations
  - Low RMSF → sharp attention on stable states
  - Learned joint query from static + RMSF embedding

- **Membrane Geometry Path**: Encodes OPM depth, tilt, amphipathic score
  - Geometric prior on residue-membrane orientation
  - Cross-attention: "structure asks, dynamics answers"

- **Physicochemical Gate**: Learned modulation via KD hydrophobicity + charge
  - Respects domain knowledge: hydrophobic + depth → binding
  - Fallback when model encounters new families

**Losses:**
- Focal loss (γ=2) for severe class imbalance
- Patch contiguity loss: smooth spatial clusters
- Contrastive loss: family-level binding representation alignment

### Phase 2: PMPGen

**Novel Components:**
- **MD-Informed Anisotropic Noise Schedule**: σ_i(t) ∝ RMSF_i
  - Flexible loops: more noise → creative sampling
  - Rigid helices: less noise → conservative sampling

- **Membrane Plane Guidance**: Gradient of geometric energy during sampling
  - No retraining required
  - Cheap: just MSE between depth profiles

- **3-Stage Validation**:
  1. ESMFold foldability (pLDDT > 70)
  2. DynaMo binding patch recall > 0.8
  3. Rosetta ΔG + structural diversity ranking

---

## Data Requirements

### Input Files

```
data/raw/
├── pdb/
│   ├── 2abc.pdb  (or .cif)
│   └── ...
├── opm/
│   ├── 2abc.json  { "normal": [...], "depth": [...] }
│   └── ...
├── md_trajectories/
│   ├── 2abc.xtc   (MDAnalysis-compatible)
│   └── ...
└── labels/
    ├── 2abc_binding.csv  { residue_id, is_binding }
    └── ...
```

### Split Files

```
data/splits/
├── train.txt   (protein codes, one per line)
├── val.txt
└── test.txt    (held-out PMP families)
```

---

## Training

### Config Example

```yaml
# configs/train/phase1.yaml
defaults:
  - data: pmp_dataset
  - model: dynamo

train:
  max_epochs: 100
  lr: 1.0e-4
  warmup_steps: 500
  lambda_focal: 1.0
  lambda_patch: 0.2
  lambda_contrast: 0.1
```

### Run with Overrides

```bash
python scripts/train_phase1.py \
  data.batch_size=8 \
  train.lr=5e-5 \
  data.n_snapshots=30
```

---

## Evaluation

### Metrics

**Phase 1:**
- MCC (Matthews correlation coefficient)
- AUROC
- Patch-level F1 (recall + precision on contiguous clusters)
- Attention map interpretability

**Phase 2:**
- pLDDT (ESMFold confidence)
- TM-score to generated structure
- Patch recall (does binding patch appear?)
- Sequence novelty vs training set

### Run Benchmarks

```bash
python scripts/evaluate.py \
  --phase 1 \
  --checkpoint outputs/checkpoints/dynamo_best.ckpt \
  --test_set data/splits/test.txt
```

Compare vs baselines:
- ScanNet (geometric interface prediction)
- MBPred / Membrain (PMP-specific methods)
- ProteinMPNN (sequence-only)

---

## Testing

```bash
make test              # run all tests
pytest tests/test_equivariance.py -v   # critical: SE(3) check
```

**Key Test:** Verify SE(3) equivariance

```python
# Rotate input protein by random SO(3), check output is rotation-invariant
R = roma.random_rotmat()
batch_rot = rotate_batch(batch, R)
H_orig = model(batch)
H_rot = model(batch_rot)
assert torch.allclose(H_orig, H_rot, atol=1e-5)
```

---

## Paper Submission

### Phase 1 Paper (Prediction)

*Title*: "DynaMo: Conformational Ensemble-Aware Prediction of Peripheral Membrane Protein Binding Sites"

**Figures:**
1. Full architecture diagram (DynaMo_architecture.pdf)
2. AUROC comparison vs baselines
3. Attention maps showing learned ensemble pooling
4. Ablation: +conformational pool, +geometry path, +cross-attention
5. Performance on held-out PMP families

**Tables:**
- MCC, AUROC, F1 on test set vs ScanNet, MBPred
- Ablation results
- Hyperparameter sensitivity

### Phase 2 Paper (Generation)

*Title*: "PMPGen: Membrane-Guided SE(3) Flow Matching for De Novo Peripheral Membrane Protein Design"

**Figures:**
1. PMPGen architecture diagram
2. Generated protein backbones (3D visualizations)
3. Patch recall vs scaffold fidelity trade-off
4. MD simulation validation of generated proteins
5. Comparison to RFdiffusion on membrane geometry

---

## Dependencies

Key packages:
- `torch>=2.2`
- `torch-geometric>=2.5`
- `gvp-pytorch` (GVP layer primitives)
- `fair-esm>=2.0` (ESM-2 embeddings)
- `biopython>=1.83` (PDB parsing)
- `MDAnalysis>=2.7` (MD trajectory processing)
- `freesasa>=2.1` (SASA computation)
- `hydra-core>=1.3` (config management)
- `wandb` (experiment tracking)
- `roma` (SO(3) / SE(3) math)

---

## Troubleshooting

### SE(3) Equivariance Test Fails

**Problem**: Rotating input protein changes scalar output

**Cause**: Vector features in scalar channel, or nonlinearity applied directly to 3D coords

**Fix**: 
- Verify node_vectors only contain 3D unit vectors
- Check graph_builder.py: edge_attr_s should never include raw coordinates
- Run `python tests/test_equivariance.py -v` to pinpoint the layer

### Out of Memory

**Reduce batch size**:
```bash
python scripts/train_phase1.py data.batch_size=2
```

**Reduce MD snapshots**:
```bash
python scripts/train_phase1.py data.n_snapshots=10
```

### PLM Embedding Dominates

**Problem**: Training loss plateaus, other features ignored

**Cause**: ESM-2 1280-dim embeddings dominate gradient signal

**Fix**: Verify fusion.py has independent LayerNorm on each stream before concat:
```python
psi = self.plm_norm(self.plm_proj(plm_emb))   # <- required
s_struct_norm = self.struct_norm(s_struct)    # <- required
```

---

## Contributing

Code style: `ruff check src/ && ruff format src/`

Before committing:
1. Pass all tests: `make test`
2. Equivariance check: `pytest tests/test_equivariance.py`
3. Format: `make lint`

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{dynamo2024,
  title={DynaMo: Conformational Ensemble-Aware Prediction of Peripheral Membrane Protein Binding Sites},
  author={[Hitesh]},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2027}
}

@inproceedings{pmpgen2024,
  title={PMPGen: Membrane-Guided SE(3) Flow Matching for De Novo Peripheral Membrane Protein Design},
  author={[Hitesh]},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2027}
}
```

---

## License

MIT

---

## Contact

Questions? Create an issue or contact the authors.

**Happy research!** 🧬
