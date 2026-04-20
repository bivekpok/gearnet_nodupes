import os
import re

from setuptools import find_packages, setup


def read_version():
    """Parse __version__ from torchdrug/__init__.py so it stays the single source of truth."""
    init_path = os.path.join(os.path.dirname(__file__), "torchdrug", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        text = f.read()
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    if not match:
        raise RuntimeError(
            "Could not find __version__ in torchdrug/__init__.py. "
            "Make sure that file defines e.g. __version__ = \"0.2.1\"."
        )
    return match.group(1)


setup(
    name="torchdrug-custom",
    version=read_version(),
    description="Modified TorchDrug fork used by the GPSforTMDs / GearNet project.",
    long_description=(
        "A project-local fork of TorchDrug with modifications required by "
        "GPSforTMDs. Installed as 'torchdrug-custom' on PyPI-style metadata "
        "but still importable as `import torchdrug`."
    ),
    url="https://github.com/bivekpok/gearnet_nodupes",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "decorator",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "tqdm",
        "networkx",
        "scikit-learn",
        "ninja",
        "jinja2",
        "pyyaml",
        "lmdb",
        "easydict",
        "huggingface_hub",
        "biopython",
        # NOTE: torch, torch-scatter, and torch-cluster are intentionally NOT
        # listed here. They must match the PyTorch build that is already
        # installed in the user's environment, and the correct installation
        # procedure differs by OS. See README.md -> "Installation".
    ],
    extras_require={
        # ESM is only needed if you instantiate torchdrug.models.EvolutionaryScaleModeling.
        # GearNet does not require it. Install with:  pip install -e './torchdrug[esm]'
        "esm": ["fair-esm"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
