
# GPSforTMDs: Membrane Protein Localization Prediction

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Built with TorchDrug](https://img.shields.io/badge/Built%20with-TorchDrug-ff69b4.svg)](https://github.com/DeepGraphLearning/torchdrug)

Predicting membrane protein localization using graph neural networks on protein structure and chemistry.

**Project link:** https://github.com/bivekpok/gearnet_nodupes

![GPSforTMDs Architecture](model_archi.png)

## Overview

**Tech used:** Python, PyTorch, TorchDrug, Graph Neural Networks, CUDA, BioPython, Foldseek

GPSforTMDs implements a modified GEARNET architecture built on the TorchDrug framework for classifying membrane proteins into their native environments. The model processes protein structures as graphs where nodes represent alpha-carbons and edges capture spatial, sequential, and chemical relationships. We trained on a curated dataset from the OPM database and evaluated the model with rigorous structural de-duplication, class-balancing strategies, and a 6-fold nested cross-validation pipeline designed to reduce data leakage.

## Repository Layout

- [`torchdrug/`](./torchdrug): bundled **modified TorchDrug fork** used by this project
- [`gpstmd/github_uploads/train_splits.py`](./gpstmd/github_uploads/train_splits.py): main training script
- [`gpstmd/github_uploads/gearnet_modules.py`](./gpstmd/github_uploads/gearnet_modules.py): model, loader, and utility code
- [`gpstmd/github_uploads/datasplitterv2.py`](./gpstmd/github_uploads/datasplitterv2.py): nested split generation
- [`gpstmd/github_uploads/production_splitsv2/`](./gpstmd/github_uploads/production_splitsv2): production nested cross-validation split manifests
- [`gpstmd/filter_foldseek_redundant_by_membrane.py`](./gpstmd/filter_foldseek_redundant_by_membrane.py): Foldseek-based structural redundancy removal
- [`gpstmd/readme.md`](./gpstmd/readme.md): notes specific to the training and split utilities

## Data Preprocessing and Splitting Pipeline

To ensure robust evaluation, the dataset pipeline is divided into two phases.

### Phase 1: Structural Redundancy Removal with Foldseek

Sequence-based clustering alone is often insufficient for structural biology tasks. GPSforTMDs uses **Foldseek** to perform 3D structural clustering with strict thresholds:

- Sequence identity `>= 0.3`
- TM-score `>= 0.3`
- Coverage `>= 0.8`

If a Representative protein and a Member protein fall into the same structural cluster **and** share the same membrane label, the redundant Member is discarded. This forces the model to learn generalizable features rather than memorizing structural templates.

### Phase 2: Hybrid Nested Cross-Validation

We use a nested split strategy to separate hyperparameter tuning from final model evaluation.

- **Outer loop (model evaluation):** the full dataset is divided into **6 independent outer folds**, each holding out about 16% of the data for testing with stratification to preserve class balance.
- **Inner loop (training and tuning):**
  - **Outer Fold 1** contains 3 inner folds for hyperparameter sweeping.
  - **Outer Folds 2-6** each contain 1 inner fold for final production training.

The production split manifests used by the training scripts are included directly in this repository under [`gpstmd/github_uploads/production_splitsv2/`](./gpstmd/github_uploads/production_splitsv2).

## Optimizations

- Structural redundancy filtering with Foldseek
- Hybrid 6-fold nested cross-validation
- Early stopping with 55-epoch patience
- Weighted loss functions to mitigate class imbalance
- Graph construction with 7 edge types for richer structural representation
- Confidence-based filtering for low-quality predictions

## Dataset, Models, and Logs

The complete dataset, including PDB structures, trained model checkpoints, and training logs, is hosted on Hugging Face:

[https://huggingface.co/datasets/bivek77/protein-dataset](https://huggingface.co/datasets/bivek77/protein-dataset)

Example structure:

```text
protein-dataset/
├── splits/
│   ├── Outer_Fold_1/
│   │   ├── test_manifest.csv
│   │   ├── Inner_Fold_1/
│   │   │   ├── train_manifest.csv
│   │   │   └── valid_manifest.csv
│   │   └── ...
│   └── Outer_Fold_2/
└── pdbs_m15_ac/
```

To download the dataset snapshot:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bivek77/protein-dataset",
    repo_type="dataset",
    local_dir="./data"
)
```

## Installation

This project ships a **modified local fork of TorchDrug** in [`./torchdrug`](./torchdrug).
You install that local fork, not upstream `torchdrug` from PyPI. It is installed
as distribution `torchdrug-custom` but is still imported as `import torchdrug`.

Installation is done in **four steps** because two pieces of the PyTorch
ecosystem (`torch-scatter`, `torch-cluster`) need OS-specific handling to avoid
the C++ ABI `Symbol not found: __ZN3c1017RegisterOperatorsD1Ev` error on macOS.

### Prerequisites

- Conda (Miniconda or Anaconda)
- Linux / Windows: NVIDIA GPU with CUDA 12.1 for GPU training (CPU works too)
- macOS: works CPU-only on Apple Silicon and Intel
- A C/C++ toolchain (for source-compiling `torch-scatter` / `torch-cluster` on macOS):
  - macOS: `xcode-select --install`
  - Linux: `build-essential` (or your distro's equivalent)

If any previous TorchDrug is installed globally, remove it first so imports
don't shadow the bundled one:

```bash
pip uninstall -y torchdrug torchdrug-custom
```

### Step 1 — Clone the repo

```bash
git clone https://github.com/bivekpok/gearnet_nodupes
cd gearnet_nodupes
```

### Step 2 — Create the Conda environment

**Linux / Windows (CUDA 12.1):**

```bash
conda env create -f environment.yml
conda activate gearnet
```

**macOS (Apple Silicon or Intel):**

`pytorch-cuda=12.1` does not exist on macOS, so drop that one line before
creating the env:

```bash
# macOS sed needs the empty '' arg; on Linux use `sed -i '/pytorch-cuda/d' environment.yml`
sed -i '' '/pytorch-cuda/d' environment.yml
conda env create -f environment.yml
conda activate gearnet
```

### Step 3 — Install `torch-scatter` and `torch-cluster` (OS-specific)

These two packages contain C++ extensions that must match the exact PyTorch
build installed above. Pick the command block for your OS.

**Linux / Windows — use the PyG pre-built wheel index (fast, no compile):**

```bash
pip install torch-scatter==2.1.2 torch-cluster==1.6.3 \
    -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

For a **CPU-only** Linux/Windows install, use `+cpu` instead:

```bash
pip install torch-scatter==2.1.2 torch-cluster==1.6.3 \
    -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
```

**macOS — compile from source against the installed torch:**

The PyG wheel index does not publish macOS wheels. Pre-built wheels from PyPI
are built against a different torch and will fail at import with
`Symbol not found: __ZN3c1017RegisterOperatorsD1Ev`. The reliable fix is to
build locally with `--no-build-isolation`, which forces the build to use the
torch already in this env instead of pulling a fresh one into a sandbox:

```bash
pip uninstall -y torch-scatter torch-cluster   # in case a bad wheel is cached
pip install --no-build-isolation --no-cache-dir \
    torch-scatter==2.1.2 \
    torch-cluster==1.6.3
```

If the compile fails on macOS, set the SDK target and retry:

```bash
export MACOSX_DEPLOYMENT_TARGET=11.0
export CC=clang
export CXX=clang++
pip install --no-build-isolation --no-cache-dir \
    torch-scatter==2.1.2 torch-cluster==1.6.3
```

### Step 4 — Install the bundled modified TorchDrug

Do this **last**, so its build sees the correct `torch` + `torch-scatter` +
`torch-cluster` already in the env:

```bash
pip install -e ./torchdrug
```

> **Optional — ESM support.** GearNet does **not** require the ESM model, and
> the fork now imports `torchdrug.models.EvolutionaryScaleModeling` lazily, so
> `fair-esm` is optional. Only install it if you actually want to use ESM:
>
> ```bash
> pip install -e "./torchdrug[esm]"
> ```

## Running the Streamlit demo app

A web UI for single-protein inference lives in
[`web_demo_test/app.py`](../web_demo_test/app.py). It lets you upload a PDB +
checkpoint, renders the structure in 3D, and shows ranked localization
probabilities.

The demo needs three extra packages on top of the main environment. They are
already listed in the `pip:` section of `environment.yml`, so if you followed
the steps above they are installed. To install them manually into an existing
env:

```bash
pip install streamlit py3Dmol stmol ipython_genutils
```

> **Why `ipython_genutils`?** `stmol` depends on `ipywidgets`, which on modern
> Jupyter stacks no longer pulls `ipython_genutils` transitively. Without it
> the viewer raises `ModuleNotFoundError: No module named 'ipython_genutils'`.

Launch the app (from the repo parent that contains `web_demo_test/`):

```bash
conda activate gearnet
cd web_demo_test
streamlit run app.py
```

### Features

- **File uploader** for `.pdb` / `.ent` structures and trained `.pth` / `.pt`
  checkpoints. FASTA-format uploads are detected and the user is told to fold
  the sequence first.
- **3D viewer** (`py3Dmol` + `stmol`): any valid PDB is rendered inline in an
  `"👁️ View 3D Structure"` expander with cartoon / spectrum colouring.
- **Demo Mode** toggle in the sidebar: bypasses the file uploaders and loads
  a bundled sample PDB + checkpoint from fixed local paths so you can exercise
  the full pipeline without uploading anything. Useful while developing.
- **Ranked results**: top prediction + confidence as `st.metric`, full
  probability distribution as bar chart and ranked table, plus a CSV download.
- **Graceful errors**: missing optional deps (`py3Dmol`, `stmol`, `fair-esm`),
  bad checkpoint / config mismatches, and FASTA-instead-of-PDB uploads all
  produce friendly `st.error` / `st.warning` messages instead of red
  tracebacks.

### Demo Mode paths

When Demo Mode is ON, the app reads these two files from disk:

```text
web_demo_test/1a0s.pdb
web_demo_test/Outer_Fold_1_Inner_Fold_1_best_model.pth
```

If you clone the repo elsewhere, edit `DEMO_PDB_PATH` and `DEMO_CKPT_PATH`
at the top of `web_demo_test/app.py` to point at your local copies — or just
leave Demo Mode OFF and use the uploaders.

### Verify the installation

```bash
python -c "import torch, torch_scatter, torch_cluster; \
print('torch', torch.__version__, '| scatter', torch_scatter.__version__, '| cluster', torch_cluster.__version__)"

python -c "import torchdrug; print('TorchDrug', torchdrug.__version__, 'from', torchdrug.__file__)"
```

The second line must print a path inside **this repository's** `torchdrug/`
directory (e.g. `.../gearnet_nodupes/torchdrug/torchdrug/__init__.py`), not
anything under `site-packages/`. That confirms the editable install of the
local fork is in use.

## Usage

The primary training utilities live under [`gpstmd/github_uploads/`](./gpstmd/github_uploads).

### Main training script

Run training from the `gpstmd/github_uploads` directory so that local module imports resolve cleanly:

```bash
cd gpstmd/github_uploads

python train_splits.py \
  --pdb_folder <path_to_membrane_pdbs> \
  --soluble_folder_ac <path_to_soluble_pdbs> \
  --split_dir production_splitsv2/Outer_Fold_1/Inner_Fold_1 \
  --output_dir <path_for_results>
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--pdb_folder` | Path to membrane protein PDB files | Required |
| `--soluble_folder_ac` | Path to soluble protein PDB files | Required |
| `--split_dir` | Path to the inner fold directory | Required |
| `--output_dir` | Directory to save training results | Required |
| `--num_epochs` | Number of training epochs | `2500` |
| `--batch_size` | Training batch size | `20` |
| `--learning_rate` | Initial learning rate | `1e-3` |
| `--hidden_dim` | Hidden dimension size | `512` |
| `--num_gearnet_layers` | Number of GearNet layers | `3` |
| `--mlp_dropout` | MLP dropout rate | `0.2` |
| `--weight_decay` | AdamW weight decay | `1e-5` |
| `--readout` | Graph readout function (`sum`, `mean`) | `sum` |
| `--concat_hidden` | Concatenate hidden layers | `True` |
| `--knn_k` | K for KNN edge construction | `10` |
| `--spatial_radius` | Radius for spatial edges in angstroms | `10.0` |
| `--activation` | Activation function (`relu`, `silu`, `gelu`) | `relu` |
| `--seed` | Random seed | `56` |

### Example production run

```bash
cd gpstmd/github_uploads

python train_splits.py \
  --pdb_folder <path_to_membrane_proteins> \
  --soluble_folder_ac <path_to_soluble_proteins> \
  --split_dir production_splitsv2/Outer_Fold_1/Inner_Fold_1 \
  --output_dir <path_for_results> \
  --learning_rate 0.00000394644981433921 \
  --weight_decay 0.00008113944975079617 \
  --mlp_dropout 0.2903210512935248 \
  --readout mean \
  --num_gearnet_layers 5 \
  --batch_size 32 \
  --hidden_dim 512 \
  --concat_hidden False \
  --knn_k 25 \
  --spatial_radius 12.0 \
  --num_epochs 2500 \
  --seed 56 \
  --activation relu
```

The script resolves:

- `train_manifest.csv` and `valid_manifest.csv` from `--split_dir`
- `test_manifest.csv` from the parent outer-fold directory

Best checkpoints and training outputs are written to `--output_dir`.

## Foldseek-Based Redundancy Filtering

The preprocessing helper is:

[`gpstmd/filter_foldseek_redundant_by_membrane.py`](./gpstmd/filter_foldseek_redundant_by_membrane.py)

Example usage:

```bash
python gpstmd/filter_foldseek_redundant_by_membrane.py \
  --clusters-tsv path/to/s03tm03_clusters_cluster.tsv \
  --dataset-csv path/to/final_pdb_csv2.csv \
  --output-csv path/to/final_pdb_csv2_nodupes.csv
```

Optional arguments:
- `--pdb-column`
- `--label-column`

## Citation

If you use this codebase in your research, please cite the following references.

```bibtex
@inproceedings{zhang2022protein,
  title={Protein representation learning by geometric structure pretraining},
  author={Zhang, Zuobai and Xu, Minghao and Jamasb, Arian and Chenthamarakshan, Vijil and Lozano, Aurelie and Das, Payel and Tang, Jian},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@article{zhang2023enhancing,
  title={A Systematic Study of Joint Representation Learning on Protein Sequences and Structures},
  author={Zhang, Zuobai and Wang, Chuanrui and Xu, Minghao and Chenthamarakshan, Vijil and Lozano, Aurelie and Das, Payel and Tang, Jian},
  journal={arXiv preprint arXiv:2303.06275},
  year={2023}
}

@article{zhu2022torchdrug,
  title={TorchDrug: A powerful and flexible machine learning platform for drug discovery},
  author={Zhu, Zhaocheng and Shi, Chence and Zhang, Peifa and Liu, Shengchao and Xu, Mai and Yuan, Xinyu and Wang, Jiacheng and Zhang, Biao and Liu, Jie and Luo, Ying and others},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={1},
  pages={1--8},
  year={2022}
}

@article{pokhrel2024gpsfortmds,
  title={GPSforTMDs: Predicting Membrane Protein Localization by Deep Learning on Structure and Chemistry},
  author={Pokhrel, Bivek and Munley, Christian and Lyman, Edward and Pedraza, Miguel},
  journal={Nature Communications},
  year={2024},
  publisher={Nature Publishing Group}
}
