"""
predict_single.py — Inference on a single PDB using a trained GearNet checkpoint.

Usage:
    python3 predict_single.py
"""
import sys
import os

# Tell Python exactly where to find gearnet_modules.py
sys.path.append("/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover")

# Keep Torch / Matplotlib caches on a writable path for sandboxed runs.
python_bin = os.path.dirname(sys.executable)
os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from torchdrug import core, data, utils
from torchdrug.data import DataLoader
from torchdrug.data.dataloader import graph_collate
from torch.utils import data as torch_data

from gearnet_modules import initialize_model, transform

# Replace with your actual 9 class names in alphabetical sort order
# (the order _make_dict() assigned indices during training)
LABEL_NAMES = [
    "class_0",
    "class_1",
    "class_2",
    "class_3",
    "class_4",
    "class_5",
    "class_6",
    "class_7",
    "class_8",
]


# ---------------------------------------------------------------------------
# Minimal single-sample dataset
# ---------------------------------------------------------------------------
class _SinglePDBDataset(torch_data.Dataset):
    def __init__(self, pdb_path: str):
        protein = data.Protein.from_pdb(
            pdb_path,
            atom_feature="position",
            bond_feature="length",
            residue_feature="symbol",
        )
        if protein is None:
            raise RuntimeError(f"Failed to load protein from: {pdb_path}")
        self._protein = protein
        self._name = os.path.basename(pdb_path)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        item = {"graph": self._protein, "label": 0, "name": self._name}
        return transform(item)


class _BootstrapLabelDataset(torch_data.Dataset):
    """Tiny dataset used only to build the classification head for checkpoint load."""

    def __init__(self, label: int = 0):
        self.label = label

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return {"label": self.label}


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def run_local_inference(pdb_path: str, checkpoint_path: str, config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = [torch.cuda.current_device()] if device.type == "cuda" else None
    print(f"Device: {device}")

    # 1. Build task
    print("Building model architecture...")
    task = initialize_model(
        num_classes=config["num_classes"],
        class_weights=None,
        mlp_dropout=config["mlp_dropout"],
        hidden_dim=config["hidden_dim"],
        num_gearnet_layers=config["num_gearnet_layers"],
        readout=config["readout"],
        concat_hidden=config["concat_hidden"],
        knn_k=config["knn_k"],
        spatial_radius=config["spatial_radius"],
        activation=config["activation"],
    )

    # 2. Load PDB and build MLP head
    print(f"Loading PDB: {pdb_path}")
    ds = _SinglePDBDataset(pdb_path)
    bootstrap_ds = _BootstrapLabelDataset()

    print("Bootstrapping task metadata...")
    task.preprocess(bootstrap_ds, None, None)

    # 3. Minimal Engine
    solver = core.Engine(
        task=task,
        train_set=bootstrap_ds,
        valid_set=bootstrap_ds,
        test_set=bootstrap_ds,
        optimizer=None,
        scheduler=None,
        early_stopping=None,
        best_model_path=None,
        batch_size=1,
        gpus=gpu_ids,
    )

    # 4. Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    solver.load(checkpoint_path, load_optimizer=False)
    task.eval()

    # 5. Inference
    loader = DataLoader(ds, batch_size=1, collate_fn=graph_collate)
    batch  = next(iter(loader))

    if device.type == "cuda":
        batch = utils.cuda(batch, device=device)

    with torch.no_grad():
        logits = task.predict(batch)
        probs  = torch.softmax(logits, dim=-1)

    # 6. Print results
    pred_idx  = probs.argmax(dim=-1).item()
    pred_conf = probs[0, pred_idx].item()

    print("\n=== PREDICTION ===")
    print(f"PDB file  : {pdb_path}")
    print(f"Prediction: {LABEL_NAMES[pred_idx]}  (class {pred_idx})")
    print(f"Confidence: {pred_conf:.4f}")
    print("\nFull probability distribution:")
    for i, (name, prob) in enumerate(zip(LABEL_NAMES, probs[0].tolist())):
        marker = " <--" if i == pred_idx else ""
        print(f"  {name:20s}: {prob:.4f}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_PDB    = "/work/hdd/bdja/bpokhrel/gearnet_files/pdb_m15_ac/1a0s.pdb"
    CHECKPOINT  = "/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover/production_models_v2/Outer_Fold_1_Inner_Fold_1_best_model.pth"

    MY_TRAINING_CONFIG = {
        "num_classes"       : 9,
        "hidden_dim"        : 512,
        "num_gearnet_layers": 5,
        "mlp_dropout"       : 0.2903210512935248,
        "readout"           : "mean",
        "concat_hidden"     : False,
        "knn_k"             : 25,
        "spatial_radius"    : 12.0,
        "activation"        : "relu",
    }

    if os.path.exists(TEST_PDB) and os.path.exists(CHECKPOINT):
        run_local_inference(TEST_PDB, CHECKPOINT, MY_TRAINING_CONFIG)
    else:
        print("Please update TEST_PDB and CHECKPOINT with real file paths!")