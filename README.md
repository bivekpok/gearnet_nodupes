🙏 Acknowledgments
Based on the TorchDrug library:

Zhu, Z., Shi, C., Zhang, P., Liu, S., Xu, M., Yuan, X., ... & Tang, J. (2022). TorchDrug: A powerful and flexible machine learning platform for drug discovery. Journal of Machine Learning Research.


# GPSforTMDs: Membrane Protein Localization Prediction (custom Torchdrug + Gearnet)
📝 License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

Note: This work is built upon the TorchDrug library (Apache 2.0). Please cite the original authors when using this code.


[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Built with TorchDrug](https://img.shields.io/badge/Built%20with-TorchDrug-ff69b4.svg)](https://github.com/DeepGraphLearning/torchdrug)

Predicting membrane protein localization using graph neural networks on protein structure and chemistry.

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

Argument	Description	Default
--pdb_folder	Path to membrane protein PDB files	Required
--soluble_folder	Path to soluble protein PDB files	Required
--csv_path	Path to metadata CSV with localization labels	Required
--output_dir	Directory to save training results	Required
--num_epochs	Number of training epochs	2500
--batch_size	Training batch size	32
--learning_rate	Initial learning rate	1e-4
--gpus	GPU IDs to use (comma-separated)	0


python train.py \
    --pdb_folder <path_to_membrane_proteins> \
    --soluble_folder <path_to_soluble_proteins> \
    --csv_path <path_to_metadata_csv> \
    --output_dir <path_for_results> \
    [--num_epochs 2500]



## 📚 Citation

If you use this codebase in your research, please cite the following papers:

```bibtex
@inproceedings{zhang2022protein,
  title={Protein representation learning by geometric structure pretraining},
  author={Zhang, Zuobai and Xu, Minghao and Jamasb, Arian and Chenthamarakshan, Vijil and Lozano, Aurelie and Das, Payel and Tang, Jian},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
bibtex
@article{zhang2023enhancing,
  title={A Systematic Study of Joint Representation Learning on Protein Sequences and Structures},
  author={Zhang, Zuobai and Wang, Chuanrui and Xu, Minghao and Chenthamarakshan, Vijil and Lozano, Aurelie and Das, Payel and Tang, Jian},
  journal={arXiv preprint arXiv:2303.06275},
  year={2023}
}
bibtex
@article{pokhrel2024gpsfortmds,
  title={GPSforTMDs: Predicting Membrane Protein Localization by Deep Learning on Structure and Chemistry},
  author={Pokhrel, Bivek and Munley, Christian and Lyman, Edward and Pedraza, Miguel},
  journal={Nature Communications},
  year={2024},
  publisher={Nature Publishing Group}
}
Also cite the original TorchDrug paper:

bibtex
@article{zhu2022torchdrug,
  title={TorchDrug: A powerful and flexible machine learning platform for drug discovery},
  author={Zhu, Zhaocheng and Shi, Chence and Zhang, Peifa and Liu, Shengchao and Xu, Mai and Yuan, Xinyu and Wang, Jiacheng and Zhang, Biao and Liu, Jie and Luo, Ying and others},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={1},
  pages={1--8},
  year={2022}
}

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


