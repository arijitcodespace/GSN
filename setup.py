from setuptools import setup, find_packages

setup(
    name="gsn",
    version="0.1.0",
    description="Graph State Networks — temporal link prediction with Mamba-2 SSM",
    packages=find_packages(include=["gsn", "gsn.*"]),
    python_requires=">=3.10",
    install_requires=[
        "tensorflow>=2.15",
        "einops>=0.7",
        "numpy>=1.24",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "rich>=13.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort"],
    },
)
