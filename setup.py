from setuptools import setup

setup(
    name="tyc_dataset",
    version="0.1",
    url="https://github.com/ChristophReich1996/TYC-Dataset",
    license="CC BY 4.0",
    author="Christoph Reich",
    author_email="ChristophReich@gmx.net",
    description="Code of the TYC Dataset.",
    packages=[
        "tyc_dataset",
        "tyc_dataset.data",
        "tyc_dataset.eval",
        "tyc_dataset.vis",
    ],
    install_requires=[
        "torch>=1.0.0",
        "torchmetrics @ git+https://github.com/Lightning-AI/torchmetrics.git@release/stable"
        "numpy",
        "matplotlib",
        "kornia",
        "opencv-python",
    ],
)
