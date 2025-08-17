from setuptools import setup, find_packages

setup(
    name="gearnet-nodupes",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torchdrug-custom @ file://localhost/${PWD}/torchdrug",  # Local path reference
        # Other Python deps (already in environment.yml)
    ],
)
