from setuptools import setup, find_packages

setup(
    name='mlm-scoring-transformers',
    version='0.1',
    description="Masked Language Model Scoring by Huggingface transformers",
    author="Ryutaro Asahara",
    packages=find_packages("mlmt"),
    install_requires=[
        'transformers~=3.3.1',
        'fugashi<=1.1.2',
        'ipadic<=1.0.0'
    ],
)