GPSforTMDs: Membrane Protein Localization Prediction
https://img.shields.io/badge/License-Apache%25202.0-blue.svg
https://img.shields.io/badge/python-3.10-blue.svg
https://img.shields.io/badge/Built%2520with-TorchDrug-ff69b4.svg

GPSforTMDs (Graph-based Protein Structure for TransMembrane Domains) is a deep learning framework for predicting the native membrane environment of integral membrane proteins based on their 3D structure and chemistry. This repository contains a custom implementation based on the GEARNET architecture and a modified fork of the TorchDrug library.

📋 Overview
This work is built upon and contains modifications of the TorchDrug library, a powerful PyTorch-based machine learning toolbox for drug discovery.

Zhu, Z., Shi, C., Zhang, P., Liu, S., Xu, M., Yuan, X., ... & Tang, J. (2022). TorchDrug: A powerful and flexible machine learning platform for drug discovery. Journal of Machine Learning Research (JMLR).

Our implementation extends TorchDrug with:

Custom modifications to the GEARNET graph neural network architecture

Specialized data loaders for membrane and soluble protein structures

Training and evaluation pipelines for whole-graph classification tasks

📥 Installation
Prerequisites
Conda (Miniconda or Anaconda)

NVIDIA GPU with CUDA 12.1 (recommended)

Quick Installation
bash
# Clone the repository
git clone https://github.com/bivekpok/gearnet_nodupes
cd gearnet_nodupes

# Create and activate the conda environment
conda env create -f environment.yml
conda activate gearnet
⚡ Usage
Run the membrane protein classification training with:

bash
python script.py \
    --pdb_folder <path_to_membrane_proteins> \
    --soluble_folder <path_to_soluble_proteins> \
    --csv_path <path_to_metadata_csv> \
    --output_dir <path_for_results> \
    [--num_epochs 2500]
Key Arguments
--pdb_folder: Path to directory containing membrane protein PDB files

--soluble_folder: Path to directory containing soluble protein PDB files

--csv_path: Path to CSV file with protein metadata and localization labels

--output_dir: Directory to save training results and model checkpoints

--num_epochs: Number of training epochs (default: 2500)

🔧 Modifications to TorchDrug
This project includes a modified version of TorchDrug with the following key changes:

Custom Data Loading: Enhanced protein graph construction for membrane proteins

Architecture Tweaks: Modified GEARNET implementation for whole-graph classification

Training Pipeline: Custom training loops and evaluation metrics for localization prediction

Membrane-specific Features: Specialized handling of transmembrane domain properties

For detailed information about the original TorchDrug framework, please refer to the official repository.

📊 Dataset
The model is trained on curated membrane protein structures from the OPM database, including:

Eukaryotic plasma membrane proteins

Bacterial inner and outer membrane proteins

Organellar membranes (mitochondrial, ER, Golgi, etc.)

Soluble proteins as negative controls

📝 License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

Note: This project incorporates and modifies code from the TorchDrug library, which is also licensed under Apache 2.0. The original copyright notices have been preserved in accordance with the license requirements.

🙏 Acknowledgments
We acknowledge and thank the developers of the TorchDrug library, which served as the foundation for our implementation:

Zhu, Z., Shi, C., Zhang, P., Liu, S., Xu, M., Yuan, X., ... & Tang, J. (2022). TorchDrug: A powerful and flexible machine learning platform for drug discovery. Journal of Machine Learning Research (JMLR).

📧 Contact
For questions about this implementation, please open an issue on GitHub or contact:

[Your Name] - [your.email@domain.com]

Project Repository: https://github.com/bivekpok/gearnet_nodupes


# GEARNET with Custom TorchDrug

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

Molecular property prediction with a custom GEARNET implementation and modified TorchDrug fork.

## 📥 Installation

### Prerequisites
- Conda (Miniconda or Anaconda)
- NVIDIA GPU with CUDA 12.1 (recommended)

### Quick Installation
```bash
git clone https://github.com/bivekpok/gearnet_nodupes
cd gearnet_nodupes
conda env create -f environment.yml  # Creates 'gearnet' environment
conda activate gearnet

### ⚡ Usage

Run the protein classification training with:

python script.py \
    --pdb_folder <path_to_membrane_proteins> \
    --soluble_folder <path_to_soluble_proteins> \
    --csv_path <path_to_metadata_csv> \
    --output_dir <path_for_results> \
    [--num_epochs 2500]


