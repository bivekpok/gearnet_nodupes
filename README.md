---
title: GearNet Protein Localization
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.37.0"
app_file: app.py
pinned: false
license: mit
---

# GearNet Protein Localization (Streamlit Space)

Predict subcellular localization from a PDB structure using a trained GearNet
model on top of a custom [`gearnet_nodupes`](https://github.com/bivekpok/gearnet_nodupes) TorchDrug fork.

## How to use

1. Upload a trained checkpoint (`.pth` / `.pt`).
2. Upload the PDB structure you want to classify (`.pdb` / `.ent`).
3. Optionally upload a JSON config; otherwise the app uses `DEFAULT_CONFIG`
   from `inference.py` (the values that match the provided checkpoint).
4. Click **Run prediction** — the app shows the top class, a bar chart of all
   class probabilities, and a ranked table.

## Files

| File                | Purpose                                                   |
| ------------------- | --------------------------------------------------------- |
| `app.py`            | Streamlit UI (file uploads, device detection, results).   |
| `inference.py`      | Pure-Python prediction API (importable + CLI).            |
| `gearnet_modules.py`| Model definition, matching training code 1-to-1.          |
| `requirements.txt`  | HF Spaces dependencies (incl. the custom `torchdrug`).    |

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

CLI usage (no UI):

```bash
python inference.py path/to/structure.pdb path/to/checkpoint.pth [optional_config.json]
```

## Notes for Hugging Face Spaces

* **Hardware:** the app automatically detects CUDA and falls back to CPU, so
  it runs on the Free Tier (CPU). Expect ~30–60 seconds for the first
  prediction while TorchDrug JIT-compiles its C++ extensions.
* **Writable caches:** `inference.py` points `TORCH_EXTENSIONS_DIR` and
  `MPLCONFIGDIR` to `$TMPDIR`, which is always writable inside the HF
  container.
* **Custom TorchDrug:** installed from
  `git+https://github.com/bivekpok/gearnet_nodupes.git#egg=torchdrug`.
  If that repository's `setup.py` lives in a subdirectory rather than the
  repo root, change the entry in `requirements.txt` to something like:

  ```
  git+https://github.com/bivekpok/gearnet_nodupes.git#egg=torchdrug&subdirectory=torchdrug
  ```

## Default class labels

Defined in `inference.DEFAULT_LABEL_NAMES` and override-able via a `"label_names"`
key in the uploaded config JSON:

1. Archaebac.
2. Endoplasm. reticulum
3. Eukaryo. plasma
4. Gram-neg. inner
5. Gram-neg. outer
6. Gram-pos. inner
7. Mitochon. inner
8. Mitochon. outer
9. Thylakoid
