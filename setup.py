from setuptools import setup, find_packages
from glob import glob
from os.path import basename
from os.path import splitext

setup(
    name="mlm-scoring-transformers",
    version="0.1",
    description="Masked Language Model Scoring by Huggingface transformers",
    author="Ryutaro Asahara",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    install_requires=["transformers", "fugashi<=1.1.2", "ipadic<=1.0.0"],
)

