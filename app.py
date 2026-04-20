"""
app.py — Streamlit UI for GearNet protein localization prediction.

Designed to run on Hugging Face Spaces (Free Tier / CPU) as well as a local
GPU. Users upload a protein 3D structure (PDB) + a trained checkpoint + an
optional JSON config, and get ranked class probabilities with a 3D viewer.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

# Make sibling modules importable no matter where Streamlit is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
import streamlit as st
import torch

# --- Optional 3D viewer imports ---------------------------------------------
# We keep them optional so the app still runs if py3Dmol / stmol aren't
# installed — only the viewer panel is disabled in that case.
_VIEWER_IMPORT_ERROR: Optional[Exception] = None
try:
    import py3Dmol
    from stmol import showmol
    _VIEWER_AVAILABLE = True
except Exception as _e:
    _VIEWER_IMPORT_ERROR = _e
    _VIEWER_AVAILABLE = False

# --- Inference backend import ----------------------------------------------
_INFERENCE_IMPORT_ERROR: Optional[Exception] = None
try:
    from inference import (
        DEFAULT_CONFIG,
        DEFAULT_LABEL_NAMES,
        InferenceError,
        UnsupportedFileFormat,
        build_task_and_load_checkpoint,
        get_device,
        predict_pdb_file,
    )
except Exception as e:
    _INFERENCE_IMPORT_ERROR = e
    DEFAULT_CONFIG = {}
    DEFAULT_LABEL_NAMES = []


# ---------------------------------------------------------------------------
# Demo mode: fixed local paths used when the user flips the sidebar toggle.
# ---------------------------------------------------------------------------
DEMO_PDB_PATH = "/Users/bivekpokhrel/Desktop/bio/correction_gnet/web_demo_test/1a0s.pdb"
DEMO_CKPT_PATH = (
    "/Users/bivekpokhrel/Desktop/bio/correction_gnet/web_demo_test/"
    "Outer_Fold_1_Inner_Fold_1_best_model.pth"
)


# ---------------------------------------------------------------------------
# Unified file representation
# ---------------------------------------------------------------------------
@dataclass
class FilePayload:
    """A single PDB or checkpoint file, regardless of whether it came from
    an `st.file_uploader` or a demo-mode disk path."""
    name: str
    data: bytes

    @property
    def ext(self) -> str:
        return os.path.splitext(self.name)[1].lower().lstrip(".")


def _payload_from_upload(upload) -> Optional[FilePayload]:
    if upload is None:
        return None
    return FilePayload(name=upload.name, data=bytes(upload.getbuffer()))


def _payload_from_path(path: str) -> FilePayload:
    with open(path, "rb") as fh:
        return FilePayload(name=os.path.basename(path), data=fh.read())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SUPPORTED_STRUCTURE_EXTS = ("pdb", "ent")
SEQUENCE_ONLY_EXTS = ("fasta", "fa", "fas", "faa")


def _hash_bytes(buf: bytes) -> str:
    return hashlib.sha1(buf).hexdigest()[:16]


def _friendly_import_error_hint(err: BaseException) -> Optional[str]:
    """Map common ModuleNotFoundError messages to actionable install hints."""
    msg = str(err)
    if "No module named 'esm'" in msg or "fair-esm" in msg:
        return (
            "The optional `fair-esm` package is missing.\n\n"
            "GearNet does not require it, but something imported it. "
            "Install with:\n\n"
            "`pip install -e \"./gearnet_nodupes/torchdrug[esm]\"`"
        )
    if "torch_scatter" in msg or "torch_cluster" in msg:
        return (
            "A PyTorch-Geometric dependency is missing or ABI-incompatible. "
            "See README.md → Installation Step 3 for OS-specific commands."
        )
    if "rdkit" in msg.lower() and "mplCanvas" in msg:
        return (
            "Your `rdkit` version is too new for this fork. "
            "Pin it to `rdkit=2022.03.5` (conda-forge)."
        )
    if "py3Dmol" in msg or "stmol" in msg:
        return "Install the 3D viewer: `pip install py3Dmol stmol`"
    return None


# ---------------------------------------------------------------------------
# 3D viewer
# ---------------------------------------------------------------------------
def _render_3d_viewer(pdb: FilePayload) -> None:
    """Render the PDB structure inside an expander using py3Dmol + stmol."""
    with st.expander("👁️ View 3D Structure", expanded=True):
        if not _VIEWER_AVAILABLE:
            st.warning(
                "The 3D viewer requires `py3Dmol` and `stmol`. "
                "Install them to enable this panel:\n\n"
                "```bash\npip install py3Dmol stmol\n```"
            )
            if _VIEWER_IMPORT_ERROR is not None:
                st.caption(f"Import error: {_VIEWER_IMPORT_ERROR}")
            return

        try:
            pdb_text = pdb.data.decode("utf-8", errors="ignore")
            view = py3Dmol.view(width=800, height=400)
            view.addModel(pdb_text, "pdb")
            view.setStyle({"cartoon": {"color": "spectrum"}})
            view.zoomTo()
            showmol(view, height=400, width=800)
            st.caption(f"Showing `{pdb.name}` — cartoon / spectrum colouring.")
        except Exception as e:
            st.error(f"Could not render 3D structure: {e}")


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _cached_task(ckpt_hash: str, ckpt_path: str, config_json: str, device_str: str):
    config = json.loads(config_json)
    device = torch.device(device_str)
    return build_task_and_load_checkpoint(config, ckpt_path, device=device)


# ---------------------------------------------------------------------------
# File-source resolver: unifies demo mode and uploader mode.
# ---------------------------------------------------------------------------
def _resolve_demo_files() -> Tuple[Optional[FilePayload], Optional[FilePayload], list[str]]:
    """Load demo PDB + checkpoint from disk. Returns (pdb, ckpt, errors)."""
    errors: list[str] = []
    pdb_payload: Optional[FilePayload] = None
    ckpt_payload: Optional[FilePayload] = None

    if not os.path.isfile(DEMO_PDB_PATH):
        errors.append(f"Demo PDB not found at `{DEMO_PDB_PATH}`.")
    else:
        try:
            pdb_payload = _payload_from_path(DEMO_PDB_PATH)
        except Exception as e:
            errors.append(f"Could not read demo PDB: {e}")

    if not os.path.isfile(DEMO_CKPT_PATH):
        errors.append(f"Demo checkpoint not found at `{DEMO_CKPT_PATH}`.")
    else:
        try:
            ckpt_payload = _payload_from_path(DEMO_CKPT_PATH)
        except Exception as e:
            errors.append(f"Could not read demo checkpoint: {e}")

    return pdb_payload, ckpt_payload, errors


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="GPSforTMDs Protein Localization",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧬 **GPSforTMDs** | Protein Localization")
    st.markdown(
        "##### Predict the **subcellular localization** of a protein from its 3D "
        "structure, using a trained graph neural network model."
    )

    if _INFERENCE_IMPORT_ERROR is not None:
        st.error("The backend (`inference.py`) failed to import.")
        hint = _friendly_import_error_hint(_INFERENCE_IMPORT_ERROR)
        if hint:
            st.warning(hint)
        with st.expander("Full traceback"):
            st.code("".join(traceback.format_exception(_INFERENCE_IMPORT_ERROR)))
        st.stop()

    device = get_device()

    # --- Sidebar -----------------------------------------------------------
    with st.sidebar:
        st.header("Runtime")
        col_a, col_b = st.columns(2)
        col_a.metric("Device", device.type.upper())
        col_b.metric("Torch", torch.__version__)
        if device.type == "cuda":
            st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.info(
                "Running on CPU. First prediction may take 30–60 s while the "
                "model loads and kernels are compiled.",
                icon="ℹ️",
            )

        st.divider()
        st.header("Demo mode")
        demo_mode = st.toggle(
            "Load Sample Data",
            value=False,
            help=(
                "Bypass the file uploaders and load a bundled PDB + checkpoint "
                "from disk. Useful for quick local testing."
            ),
        )
        if demo_mode:
            st.caption(f"**PDB:** `{os.path.basename(DEMO_PDB_PATH)}`")
            st.caption(f"**Checkpoint:** `{os.path.basename(DEMO_CKPT_PATH)}`")

        st.divider()
        st.header("Model parameters files")
        ckpt_up = st.file_uploader(
            "Trained checkpoint (.pth / .pt)",
            type=["pth", "pt"],
            help="A state_dict saved from training — must match the config below.",
            disabled=demo_mode,
        )

        use_default_cfg = st.toggle(
            "Use default config",
            value=True,
            help="Untoggle to upload a custom JSON config matching your checkpoint.",
        )
        cfg_up = None
        if not use_default_cfg:
            cfg_up = st.file_uploader("Config (.json)", type=["json"])

        with st.expander("Show default config", expanded=False):
            st.json(DEFAULT_CONFIG)

    # --- Main area: structure source --------------------------------------
    st.header("1. Upload a protein structure", divider=True)

    pdb_up = None
    if not demo_mode:
        pdb_up = st.file_uploader(
            "Protein structure file",
            type=list(SUPPORTED_STRUCTURE_EXTS) + list(SEQUENCE_ONLY_EXTS),
            help=(
                "GearNet requires a 3D structure (.pdb, .ent). "
                "FASTA sequences must be folded first (e.g. with ESMFold or "
                "AlphaFold) before they can be used here."
            ),
        )
    else:
        st.info(
            "**Demo mode is ON.** The file uploaders are disabled — the app "
            "will use the sample PDB and checkpoint shown in the sidebar.",
            icon="🧪",
        )

    # Resolve a unified FilePayload regardless of source
    if demo_mode:
        pdb_payload, ckpt_payload, demo_errors = _resolve_demo_files()
        for msg in demo_errors:
            st.error(msg)
    else:
        pdb_payload = _payload_from_upload(pdb_up)
        ckpt_payload = _payload_from_upload(ckpt_up)

    # FASTA detection (uploader path only — demo file is a known PDB)
    if pdb_payload is not None and pdb_payload.ext in SEQUENCE_ONLY_EXTS:
        st.warning(
            f"`{pdb_payload.name}` looks like a FASTA sequence, not a 3D structure. "
            "GearNet operates on protein **geometry**, so you need to fold "
            "the sequence first (e.g. ESMFold, AlphaFold, ColabFold) and "
            "upload the resulting `.pdb` file.",
            icon="⚠️",
        )

    # --- 3D viewer (shown as soon as we have a valid PDB) -----------------
    if pdb_payload is not None and pdb_payload.ext in SUPPORTED_STRUCTURE_EXTS:
        _render_3d_viewer(pdb_payload)

    # --- Run button --------------------------------------------------------
    st.header("2. Run prediction", divider=True)
    run = st.button(
        "Run prediction",
        type="primary",
        use_container_width=True,
        disabled=(pdb_payload is None or ckpt_payload is None),
        help="Provide a checkpoint and a .pdb structure (or enable Demo mode) to enable.",
    )
    if not run:
        return

    # --- Validation --------------------------------------------------------
    if ckpt_payload is None:
        st.error("No checkpoint available. Upload one or enable Demo mode.")
        return
    if pdb_payload is None:
        st.error("No protein structure available. Upload one or enable Demo mode.")
        return
    if pdb_payload.ext not in SUPPORTED_STRUCTURE_EXTS:
        st.error(
            f"Unsupported file type: `.{pdb_payload.ext}`. "
            "GearNet requires a 3D structure file (.pdb or .ent)."
        )
        return

    # --- Resolve config ----------------------------------------------------
    try:
        if use_default_cfg or cfg_up is None:
            config = dict(DEFAULT_CONFIG)
        else:
            config = json.loads(cfg_up.read().decode("utf-8"))
    except Exception as e:
        st.error(f"Could not parse config JSON: {e}")
        return

    label_names = config.get("label_names", DEFAULT_LABEL_NAMES)

    # --- Inference ---------------------------------------------------------
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pdb_path  = os.path.join(tmp, pdb_payload.name or "input.pdb")
            ckpt_path = os.path.join(tmp, ckpt_payload.name or "model.pth")

            with open(pdb_path, "wb") as fh:
                fh.write(pdb_payload.data)
            with open(ckpt_path, "wb") as fh:
                fh.write(ckpt_payload.data)

            with st.status("Running prediction…", expanded=True) as status:
                st.write("Loading model and weights…")
                task = _cached_task(
                    ckpt_hash   = _hash_bytes(ckpt_payload.data),
                    ckpt_path   = ckpt_path,
                    config_json = json.dumps(config, sort_keys=True),
                    device_str  = str(device),
                )

                st.write("Parsing structure and running inference…")
                _, probs = predict_pdb_file(task, pdb_path, device=device)

                status.update(label="Done", state="complete", expanded=False)

        p = probs.squeeze(0).cpu().numpy()

    except UnsupportedFileFormat as e:
        st.error(str(e))
        return
    except InferenceError as e:
        st.error(f"Inference failed: {e}")
        return
    except (ModuleNotFoundError, ImportError) as e:
        st.error(f"A required Python package is missing: {e}")
        hint = _friendly_import_error_hint(e)
        if hint:
            st.warning(hint)
        return
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return
    except Exception as e:
        st.error("Inference failed with an unexpected error.")
        hint = _friendly_import_error_hint(e)
        if hint:
            st.warning(hint)
        with st.expander("Full traceback"):
            st.code("".join(traceback.format_exception(e)))
        return

    # --- Results -----------------------------------------------------------
    if len(label_names) != len(p):
        label_names = [f"class_{i}" for i in range(len(p))]

    df = (
        pd.DataFrame({"Location": label_names, "Probability": p})
          .sort_values("Probability", ascending=False)
          .reset_index(drop=True)
    )

    top = df.iloc[0]
    runner_up = df.iloc[1] if len(df) > 1 else None

    st.header("3. Results", divider=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Top prediction", str(top["Location"]))
    m2.metric("Confidence", f"{top['Probability']:.1%}")
    if runner_up is not None:
        m3.metric(
            "Runner-up",
            str(runner_up["Location"]),
            delta=f"{(top['Probability'] - runner_up['Probability']):.1%} gap",
            delta_color="off",
        )

    col_chart, col_table = st.columns([1.4, 1])
    with col_chart:
        st.caption("Class probabilities")
        st.bar_chart(df.set_index("Location")["Probability"])
    with col_table:
        st.caption("Ranked probabilities")
        st.dataframe(
            df.style.format({"Probability": "{:.4f}"}),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Raw output (JSON)"):
        st.json({
            "device":     str(device),
            "input_file": pdb_payload.name,
            "mode":       "demo" if demo_mode else "upload",
            "pred_label": str(top["Location"]),
            "confidence": float(top["Probability"]),
            "probs":      {lab: float(pr) for lab, pr in zip(label_names, p)},
        })

    st.download_button(
        "Download results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{os.path.splitext(pdb_payload.name)[0]}_gearnet_prediction.csv",
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
