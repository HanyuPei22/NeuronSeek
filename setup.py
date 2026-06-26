from setuptools import find_packages, setup


setup(
    name="neuronseek",
    version="2.0.0",
    description="Task-driven neuron discovery with differentiable structure search and tensor decomposition.",
    long_description=open("readme.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Hanyu Pei",
    packages=find_packages(include=["neuronseek", "neuronseek.*"]),
    python_requires=">=3.8",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "sympy",
        "torch",
        "tqdm",
    ],
    extras_require={
        "vision": ["torchvision"],
        "experiments": ["func-timeout", "openml", "tabulate"],
    },
)
