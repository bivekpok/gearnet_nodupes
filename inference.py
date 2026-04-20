"""
inference.py — Portable single-PDB inference with a trained GearNet checkpoint.

Exports:
    DEFAULT_CONFIG         — architecture config matching the provided checkpoint
    DEFAULT_LABEL_NAMES    — human-readable class names (9 localization classes)
    InferenceError         — base exception for inference failures
    UnsupportedFileFormat  — raised when an input file is not a valid PDB
    build_task_and_load_checkpoint(config, checkpoint_path, device=None)
    predict_pdb_file(task, pdb_path, device=None) -> (logits, probs)
    predict(pdb_path, checkpoint_path, config=None, label_names=None) -> dict

CLI:
    python inference.py <pdb_path> <checkpoint_path> [config.json]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Optional

# Make sibling gearnet_modules.py importable regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Torch JIT-extension + matplotlib caches must point to a writable path
# (Hugging Face Spaces / sandboxes often have a read-only $HOME).
_TMP = tempfile.gettempdir()
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(_TMP, "torch_extensions"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_TMP, "matplotlib"))
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Make sure `ninja` (installed as a Python package) is discoverable by
# PyTorch's cpp_extension loader, which shells out to `ninja`.
_python_bin = os.path.dirname(sys.executable)
if _python_bin and _python_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _python_bin + os.pathsep + os.environ.get("PATH", "")

import torch
from torchdrug import data, layers
from torchdrug.data.dataloader import graph_collate

from gearnet_modules import initialize_model, transform


# ---------------------------------------------------------------------------
# Public exceptions — let the UI catch specific failure modes cleanly.
# ---------------------------------------------------------------------------
class InferenceError(RuntimeError):
    """Generic inference-time failure (bad checkpoint, shape mismatch, etc.)."""


class UnsupportedFileFormat(InferenceError):
    """Input file is not a valid / recognised protein structure file."""


# ---------------------------------------------------------------------------
# Defaults (matching the checkpoint used in production)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    "num_classes":        9,
    "hidden_dim":         512,
    "num_gearnet_layers": 5,
    "mlp_dropout":        0.2903210512935248,
    "readout":            "mean",
    "concat_hidden":      False,
    "knn_k":              25,
    "spatial_radius":     12.0,
    "activation":         "relu",
}

DEFAULT_LABEL_NAMES: list[str] = [
    "Archaebac.",
    "Endoplasm. reticulum",
    "Eukaryo. plasma",
    "Gram-neg. inner",
    "Gram-neg. outer",
    "Gram-pos. inner",
    "Mitochon. inner",
    "Mitochon. outer",
    "Thylakoid",
]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device(prefer_cuda: bool = True) -> torch.device:
    """Pick CUDA when available, otherwise fall back to CPU gracefully."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model construction + checkpoint loading
# ---------------------------------------------------------------------------
def _build_mlp_head(task, num_classes: int) -> None:
    """
    Replicates what `task.preprocess()` does for classification, without
    iterating a training dataset. Registers mean/std/weight buffers and
    builds the MLP head so the checkpoint can load cleanly.
    """
    task.register_buffer("mean",   torch.zeros(1))
    task.register_buffer("std",    torch.ones(1))
    task.register_buffer("weight", torch.ones(1))
    task.num_class = [num_classes]

    hidden_dims = [task.model.output_dim] * (task.num_mlp_layer - 1)
    task.mlp = layers.MLP(
        task.model.output_dim,
        hidden_dims + [num_classes],
        batch_norm=task.mlp_batch_norm,
        dropout=task.mlp_dropout,
    )


def build_task_and_load_checkpoint(
    config: dict,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    strict: bool = False,
):
    """Build the GearNet task, attach the MLP head, load weights, return eval-mode task."""
    device = device or get_device()

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        task = initialize_model(
            num_classes        = config["num_classes"],
            class_weights      = None,
            mlp_dropout        = config["mlp_dropout"],
            hidden_dim         = config["hidden_dim"],
            num_gearnet_layers = config["num_gearnet_layers"],
            readout            = config["readout"],
            concat_hidden      = config["concat_hidden"],
            knn_k              = config["knn_k"],
            spatial_radius     = config["spatial_radius"],
            activation         = config["activation"],
        )
    except KeyError as e:
        raise InferenceError(f"Missing required config key: {e}") from e

    _build_mlp_head(task, config["num_classes"])
    task = task.to(device)

    try:
        state = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise InferenceError(f"Failed to load checkpoint '{checkpoint_path}': {e}") from e

    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

    try:
        missing, unexpected = task.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        raise InferenceError(
            f"Checkpoint does not match the model architecture defined by the "
            f"config. Check that num_classes / hidden_dim / num_gearnet_layers / "
            f"etc. match the values used at training time. Underlying error: {e}"
        ) from e

    if missing:
        print(f"[inference] missing {len(missing)} keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[inference] unexpected {len(unexpected)} keys (e.g. {unexpected[:3]})")

    task.eval()
    return task


# ---------------------------------------------------------------------------
# PDB validation + loading
# ---------------------------------------------------------------------------
_PDB_EXTS = (".pdb", ".ent")


def _validate_pdb_file(pdb_path: str) -> None:
    """Cheap sanity-check before handing the file to torchdrug/rdkit."""
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"Structure file not found: {pdb_path}")

    ext = os.path.splitext(pdb_path)[1].lower()
    if ext not in _PDB_EXTS:
        raise UnsupportedFileFormat(
            f"File '{os.path.basename(pdb_path)}' has extension '{ext}', "
            f"but GearNet requires a 3D structure file ({', '.join(_PDB_EXTS)}). "
            "If you have a FASTA sequence, fold it first (e.g. ESMFold / AlphaFold)."
        )

    # Peek at the first non-blank line: a valid PDB starts with one of a small
    # set of record types. This catches accidentally-uploaded FASTA files.
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith(">"):
                    raise UnsupportedFileFormat(
                        "The uploaded file looks like a FASTA sequence, not a "
                        "PDB structure. Fold the sequence first and upload the "
                        "resulting .pdb file."
                    )
                valid_prefixes = (
                    "HEADER", "TITLE", "ATOM", "HETATM", "MODEL", "REMARK",
                    "CRYST1", "SEQRES", "COMPND", "SOURCE", "EXPDTA", "AUTHOR",
                    "NUMMDL", "DBREF", "OBSLTE", "SPLIT", "KEYWDS", "REVDAT",
                )
                if not stripped.startswith(valid_prefixes):
                    raise UnsupportedFileFormat(
                        f"File does not look like a valid PDB — first record "
                        f"starts with '{stripped[:6]}'. Expected one of: "
                        f"{', '.join(valid_prefixes[:5])} …"
                    )
                break
    except UnsupportedFileFormat:
        raise
    except Exception as e:
        raise InferenceError(f"Could not read structure file: {e}") from e


def _sanitize_pdb_text(pdb_text: str, chain: Optional[str] = None) -> str:
    """
    Produce a cleaned PDB string that RDKit's MolFromPDBFile is much more
    likely to accept:

      - drops HETATM, CONECT, ANISOU, LINK, SSBOND, MODRES, SITE, SEQADV,
        REMARK, HELIX, SHEET (RDKit-troublesome records)
      - keeps only the first MODEL block for NMR / multi-model files
      - optionally filters to a single chain (column 22 in ATOM records)

    This matches what GearNet training pipelines typically feed in.
    """
    keep_prefixes = ("ATOM  ", "TER", "END", "HEADER", "TITLE", "CRYST1")
    out: list[str] = []
    model_count = 0
    skip_model = False

    for line in pdb_text.splitlines():
        if line.startswith("MODEL"):
            model_count += 1
            skip_model = model_count > 1
            continue
        if line.startswith("ENDMDL"):
            skip_model = False
            continue
        if skip_model:
            continue
        if not line.startswith(keep_prefixes):
            continue
        if chain and line.startswith("ATOM") and len(line) >= 22 and line[21] != chain:
            continue
        out.append(line)

    if not any(l.startswith("ATOM") for l in out):
        return ""
    return "\n".join(out) + "\n"


def _list_pdb_chains(pdb_text: str) -> list[str]:
    """Return unique chain IDs appearing in ATOM records, in order of first appearance."""
    seen: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and len(line) >= 22:
            c = line[21]
            if c.strip() and c not in seen:
                seen.append(c)
    return seen


def _try_from_pdb(path: str):
    """Call torchdrug's Protein.from_pdb with the features used by GearNet."""
    return data.Protein.from_pdb(
        path,
        atom_feature="position",
        bond_feature="length",
        residue_feature="symbol",
    )


def _load_pdb_sample(pdb_path: str) -> dict:
    _validate_pdb_file(pdb_path)

    # --- 1. Try the file as-is ------------------------------------------------
    try:
        protein = _try_from_pdb(pdb_path)
    except Exception as e:
        raise InferenceError(f"torchdrug failed to parse PDB: {e}") from e

    # --- 2. If RDKit returned None, progressively sanitize ---------------------
    if protein is None:
        try:
            with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
                raw = fh.read()
        except Exception as e:
            raise InferenceError(f"Could not re-read PDB for sanitation: {e}") from e

        tried: list[str] = ["raw"]

        # Try 2a: strip HETATM / non-ATOM records + keep only first MODEL.
        cleaned_all = _sanitize_pdb_text(raw)
        if cleaned_all:
            clean_path = pdb_path + ".clean.pdb"
            with open(clean_path, "w", encoding="utf-8") as fh:
                fh.write(cleaned_all)
            try:
                protein = _try_from_pdb(clean_path)
            except Exception:
                protein = None
            tried.append("cleaned")

        # Try 2b: one chain at a time (trimers like 1a0s parse fine per-chain).
        if protein is None:
            for chain in _list_pdb_chains(raw):
                cleaned_chain = _sanitize_pdb_text(raw, chain=chain)
                if not cleaned_chain:
                    continue
                chain_path = pdb_path + f".clean.{chain}.pdb"
                with open(chain_path, "w", encoding="utf-8") as fh:
                    fh.write(cleaned_chain)
                try:
                    protein = _try_from_pdb(chain_path)
                except Exception:
                    protein = None
                tried.append(f"chain {chain}")
                if protein is not None:
                    print(f"[inference] PDB parsed successfully after isolating chain {chain}.")
                    break

        if protein is None:
            raise InferenceError(
                f"Failed to load a protein from '{os.path.basename(pdb_path)}'. "
                f"Tried: {', '.join(tried)}. RDKit (used by torchdrug) could not "
                "sanitize this PDB — the file likely contains non-standard "
                "residues, broken CONECT records, or ligand HETATMs that break "
                "RDKit's parser. Try pre-cleaning the file (keep only ATOM "
                "records of a single chain) and re-upload."
            )

    item = {"graph": protein, "label": 0, "name": os.path.basename(pdb_path)}
    return transform(item)


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    def _move(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if hasattr(x, "to"):
            try:
                return x.to(device)
            except Exception:
                return x
        return x
    return {k: _move(v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Inference entry points
# ---------------------------------------------------------------------------
def predict_pdb_file(task, pdb_path: str, device: Optional[torch.device] = None):
    """Run inference on a single PDB file. Returns (logits, probs) on CPU."""
    device = device or next(task.parameters()).device
    sample = _load_pdb_sample(pdb_path)
    batch = graph_collate([sample])
    batch = _batch_to_device(batch, device)

    try:
        with torch.no_grad():
            logits = task.predict(batch)
            probs = torch.softmax(logits, dim=-1)
    except Exception as e:
        raise InferenceError(f"Model forward pass failed: {e}") from e

    return logits.detach().cpu(), probs.detach().cpu()


def predict(
    pdb_path: str,
    checkpoint_path: str,
    config: Optional[dict] = None,
    label_names: Optional[list[str]] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    High-level convenience function.

    Returns:
        {
            "device":      "cuda" | "cpu",
            "pred_index":  int,
            "pred_label":  str,
            "confidence":  float,
            "logits":      list[float],
            "probs":       list[float],
            "labels":      list[str],
        }
    """
    config = dict(DEFAULT_CONFIG if config is None else config)
    labels = list(label_names if label_names is not None else DEFAULT_LABEL_NAMES)
    device = device or get_device()

    task = build_task_and_load_checkpoint(config, checkpoint_path, device=device)
    logits, probs = predict_pdb_file(task, pdb_path, device=device)

    p = probs.squeeze(0).tolist()
    pred_idx = int(max(range(len(p)), key=lambda i: p[i]))

    if len(labels) != len(p):
        labels = [f"class_{i}" for i in range(len(p))]

    return {
        "device":     str(device),
        "pred_index": pred_idx,
        "pred_label": labels[pred_idx],
        "confidence": float(p[pred_idx]),
        "logits":     logits.squeeze(0).tolist(),
        "probs":      p,
        "labels":     labels,
    }


def _cli() -> None:
    if len(sys.argv) < 3:
        print("Usage: python inference.py <pdb_path> <checkpoint_path> [config.json]")
        sys.exit(1)

    pdb_path  = sys.argv[1]
    ckpt_path = sys.argv[2]
    cfg = None
    if len(sys.argv) >= 4:
        with open(sys.argv[3]) as fh:
            cfg = json.load(fh)

    try:
        result = predict(pdb_path, ckpt_path, config=cfg)
    except (InferenceError, FileNotFoundError) as e:
        print(f"[inference] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"\n=== PREDICTION ===")
    print(f"Device     : {result['device']}")
    print(f"PDB file   : {pdb_path}")
    print(f"Prediction : {result['pred_label']}  (class {result['pred_index']})")
    print(f"Confidence : {result['confidence']:.4f}")
    print("\nFull probability distribution:")
    for i, (name, prob) in enumerate(zip(result["labels"], result["probs"])):
        marker = " <--" if i == result["pred_index"] else ""
        print(f"  {name:25s}: {prob:.4f}{marker}")


if __name__ == "__main__":
    _cli()
