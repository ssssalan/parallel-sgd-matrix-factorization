# Parallel SGD for Matrix Factorization

C++/OpenMP implementations of parallel stochastic-gradient baselines for large-scale matrix factorization and recommendation-system experiments.

This repository is a cleaned, reproducible portfolio version of research code developed around **parallel SGD for recommendation systems**. It demonstrates C++ engineering, OpenMP parallelism, matrix factorization, reproducible evaluation, and benchmark-oriented experiment design.

> **Research-code note:** the uploaded project also contains a proposed A-ASGD implementation. Its full source is intentionally withheld from this public-ready version until publication/IP clearance is confirmed. See [`docs/PUBLICATION_SAFETY.md`](docs/PUBLICATION_SAFETY.md).

## What is included

- **Sequential SGD** baseline
- **Hogwild!** lock-free OpenMP training
- **Locked SGD** baseline with per-user/per-item locks
- Biased matrix factorization (`global mean + user bias + item bias + latent factors`)
- Deterministic train/validation/test splitting
- RMSE evaluation and CSV metric export
- Portable CMake build
- Synthetic smoke-test data generator

The original research function labelled `DSGD` used per-user/per-item locks, so this cleaned repository calls it **locked-sgd** rather than presenting it as a canonical block-stratified DSGD implementation.

## Project structure

```text
parallel-sgd-matrix-factorization/
├── CMakeLists.txt
├── include/
│   ├── mf.hpp
│   └── optimizers.hpp
├── src/
│   ├── main.cpp
│   ├── mf.cpp
│   ├── sgd.cpp
│   ├── hogwild.cpp
│   └── locked_sgd.cpp
├── scripts/
│   ├── generate_toy_ratings.py
│   └── run_smoke_test.sh
├── data/
│   └── README.md
├── results/
│   ├── README.md
│   └── example_metrics.csv
├── figures/
│   └── README.md
└── docs/
    ├── ORIGINAL_CODE_REVIEW.md
    └── PUBLICATION_SAFETY.md
```

## Build

### Linux / macOS with OpenMP

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Windows

Use a C++ compiler with OpenMP support (for example recent MSVC):

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## Quick smoke test

The synthetic dataset is only for checking that the pipeline builds and runs; it is not a benchmark.

```bash
python3 scripts/generate_toy_ratings.py --output data/toy_ratings.csv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/mf_train \
  --data data/toy_ratings.csv \
  --method hogwild \
  --threads 4 \
  --epochs 3 \
  --output results/smoke_test.csv
```

## Run on MovieLens

Download MovieLens from GroupLens and point the executable to `ratings.csv`:

```bash
./build/mf_train \
  --data /path/to/ml-20m/ratings.csv \
  --method hogwild \
  --threads 8 \
  --rank 32 \
  --epochs 30 \
  --seed 42 \
  --output results/hogwild_t8.csv
```

Available public methods:

```text
sgd
hogwild
locked-sgd
```

The proposed research method is reserved as `aasgd`, but the public build will refuse to run it until the implementation is approved for release.

## Recommended benchmark protocol

For a recruiter-facing repository, report at least:

| Method | Threads | Rank | Validation RMSE | Test RMSE | Train time | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| SGD | 1 | 32 | — | — | — | 1.00× |
| Hogwild! | 2/4/8/16 | 32 | — | — | — | — |
| Locked SGD | 2/4/8/16 | 32 | — | — | — | — |
| A-ASGD* | 2/4/8/16 | 32 | — | — | — | — |

`*` Add A-ASGD only after release approval and after regenerating audited results from the exact paper protocol.

Recommended plots:

- RMSE vs. epoch
- Speedup vs. number of threads
- Training time vs. number of threads

## Research context

The project focuses on the systems problem behind large-scale recommendation training: sparse user-item interactions allow substantial parallelism, but concurrent SGD updates can introduce parameter conflicts, synchronization cost, load imbalance, and poor cache behavior. The research code explores alternative update strategies and data-partitioning schemes using C++ and OpenMP.

## License

MIT License for the cleaned public portfolio code in this repository. Publication/IP restrictions may separately apply to unreleased research components.
