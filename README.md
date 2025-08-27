# InversePMMDesign

Library built on top of **ceviche** and **angler** (https://github.com/fancompute) 
tailored for creating plasma metamaterial devices. This repo uses forks of these 
project with portable sparse solver backends so it runs well on Apple Silicon, Intel, 
and AMD.

---

## Quickstart

We use a standard `pyproject.toml` build with `pip`. Create a fresh
environment first.

### 1) Create and activate an environment

Option A: conda
~~~
    conda create -n PMM python=3.10
    conda activate PMM
~~~
Option B: venv
~~~
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
~~~
### 2) Choose a solver backend

- Apple Silicon (macOS arm64): prefer SuiteSparse/UMFPACK.
- Intel/AMD (x86_64): prefer PARDISO via pypardiso.
- Minimal install (SciPy SuperLU) works everywhere but is slower.

Apple Silicon (recommended):
~~~
    # Install SciPy + SuiteSparse/UMFPACK via conda-forge:
    conda install -c conda-forge scipy numpy scikit-umfpack suitesparse
    pip install -e ".[solver-suitesparse]"
~~~
Intel/AMD (Linux/Windows, x86_64, recommended):
~~~
    pip install -e ".[solver-mkl]"
~~~
Minimal install (works everywhere, slower):
~~~
    pip install -e .
~~~
You can force a backend at runtime:
ANGLER_SOLVER=pardiso or CEVICHE_SOLVER=pardiso (x86_64 only)
ANGLER_SOLVER=scipy or CEVICHE_SOLVER=scipy

### 3) Prepare output directories
~~~
    cd scripts
    python OutputDirs.py
~~~
### 4) Run an example
~~~
    # From the scripts/ directory:
    python BentWaveguide10x10.py
~~~
---

## Notes on performance and portability

- On x86_64, pypardiso bundles MKL PARDISO and is typically the
  fastest direct solver for our FDFD matrices.
- On Apple Silicon, MKL is not available natively; use
  SuiteSparse/UMFPACK (solver-suitesparse) for excellent performance.
- On HPC clusters, PETSc (solver-petsc) is supported; install
  petsc4py via conda-forge.

---

## Developer workflow

    # Install dev tools
    pip install -e ".[dev]"

    # Build sdist + wheel
    python -m pip install --upgrade pip build
    python -m build

    # Run tests
    pytest