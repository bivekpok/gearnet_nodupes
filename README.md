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
