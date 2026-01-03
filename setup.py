"""
Master-Sim: Physical AI Simulation Platform
"""
from setuptools import setup, find_packages

setup(
    name="master-sim",
    version="0.1.0",
    description="Physical AI simulation platform for precision assembly tasks",
    author="Master-Sim Team",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mujoco>=3.1.2",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=24.1.0",
            "ruff>=0.1.0",
            "mypy>=1.8.0",
            "isort>=5.13.0",
        ],
        "api": [
            "fastapi>=0.110.0",
            "uvicorn>=0.27.0",
            "pydantic>=2.6.0",
        ],
        "cloud": [
            "boto3>=1.34.0",
            "psycopg2-binary>=2.9.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
