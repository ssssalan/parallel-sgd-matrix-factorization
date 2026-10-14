# Parallel SGD for Matrix Factorization

C++/OpenMP implementations and research artifacts for parallel stochastic gradient descent in large-scale matrix factorization and recommendation systems.

This repository is associated with my first-author IEEE conference paper:

> **An Efficient Hybrid Parallel SGD Algorithm for Matrix Factorization in Recommendation Systems**  
> **Peixin Li**, Qicong Hu  
> 2025 6th International Symposium on Computer Engineering and Intelligent Communications (ISCEIC)  
> IEEE, 2025  
> DOI: [10.1109/ISCEIC67854.2025.11405719](https://doi.org/10.1109/ISCEIC67854.2025.11405719)

The project studies how to improve matrix-factorization training efficiency on large-scale sparse recommendation data using shared-memory parallelism.

The repository contains a cleaned public implementation of parallel SGD baselines and reproducible experiment infrastructure. The complete A-ASGD research implementation used for the published paper is not distributed in this portfolio release.

---

## Research Paper

### An Efficient Hybrid Parallel SGD Algorithm for Matrix Factorization in Recommendation Systems

**Peixin Li**, Qicong Hu  
2025 6th International Symposium on Computer Engineering and Intelligent Communications (ISCEIC)  
IEEE, 2025

**DOI:** [10.1109/ISCEIC67854.2025.11405719](https://doi.org/10.1109/ISCEIC67854.2025.11405719)

### Motivation

Matrix Factorization (MF) is a standard collaborative-filtering technique for modeling latent user-item interactions. On large sparse datasets such as MovieLens-20M, however, conventional SGD training becomes expensive.

Parallel SGD introduces a systems trade-off:

- lock-free methods such as **Hogwild!** can provide high throughput but permit conflicting updates;
- strongly synchronized methods improve update consistency but introduce synchronization and coordination overhead.

The paper proposes **A-ASGD (Asymmetric Asynchronous SGD)** to balance training efficiency and update stability.

### Key Idea

A-ASGD exploits asymmetric access patterns between user-side and item-side parameters.

- **User-side updates are lock-free**, maximizing parallelism.
- **Item-side updates use fine-grained block synchronization**, reducing conflicts on high-contention item regions.
- The interaction space is partitioned into a **2N × 4N block grid**, where `N` is the number of worker threads.
- Workers randomly sample blocks during training.
- Item blocks use a **non-blocking try-lock policy**: if the selected item block is busy, the worker immediately samples another block instead of waiting.

Conceptually:

```text
                     User-Item Interaction Matrix
                                |
                                v
                     2N × 4N Grid Partition
                                |
                                v
                    Randomized Block Scheduler
                                |
                +---------------+---------------+
                |                               |
                v                               v
        User-side Parameters            Item-side Parameters
          M_u and b_u                     N_i and b_i
                |                               |
                v                               v
        Lock-free Updates            Fine-grained Block Lock
                |                               |
                +---------------+---------------+
                                |
                                v
                       Parallel SGD Training
```

### Matrix Factorization Model

For user `u` and item `i`:

```text
r_hat_ui = mu + b_u + b_i + M_u^T N_i
```

where `mu` is the global mean, `b_u` and `b_i` are user/item biases, and `M_u` and `N_i` are latent-factor vectors.

For each observed rating:

```text
error = r_ui - r_hat_ui
```

The user-side parameters are updated without locks, while item-side updates are protected by the selected item-block lock.

---

## Published Experimental Setup

| Setting | Value |
|---|---|
| Dataset | MovieLens-20M |
| Data split | 80% training / 20% testing |
| CPU | 13th Gen Intel Core i7-13700 |
| CPU configuration | 16 cores / 24 threads |
| Memory | 32 GB |
| Implementation | C++ |
| Parallel framework | OpenMP |
| Accuracy metric | RMSE |
| Compared methods | A-ASGD, Hogwild!, DSGD |

---

## Published Results

### Total Training Time — 30 Epochs

The paper reports the following total training times:

| Threads | A-ASGD | DSGD | Hogwild! |
|---:|---:|---:|---:|
| 1 | **27.8 s** | 105.0 s | 49.8 s |
| 2 | **16.3 s** | 51.8 s | 25.7 s |
| 4 | **10.7 s** | 26.6 s | 12.9 s |
| 8 | **6.1 s** | 14.9 s | 7.5 s |
| 16 | **3.5 s** | 10.5 s | 5.0 s |

At 16 threads, the reported total training time is:

```text
A-ASGD      3.5 s
Hogwild!    5.0 s
DSGD       10.5 s
```

The paper attributes A-ASGD's throughput advantage to lower synchronization overhead and improved locality from grid partitioning.

### Relative Speedup vs. DSGD

The paper uses DSGD as a same-thread-count baseline in its relative-speedup comparison.

For clarity, this comparison can be written as:

```text
RelativeSpeedup(method, p) = T_DSGD(p) / T_method(p)
```

Using the published training times above:

| Threads | A-ASGD vs. DSGD | Hogwild! vs. DSGD |
|---:|---:|---:|
| 2 | 3.18× | 2.02× |
| 4 | 2.49× | 2.06× |
| 8 | 2.44× | 1.99× |
| 16 | **3.00×** | **2.10×** |

This is different from conventional per-model parallel speedup `T(1) / T(p)`.

### Convergence

At 16 threads, all three methods converge to stable RMSE values. The paper reports approximately:

```text
DSGD / Hogwild!: ~0.778 RMSE
A-ASGD:          ~0.788 RMSE
```

The approximately 0.01 RMSE difference reflects the throughput-accuracy trade-off introduced by partially lock-free updates.

The central claim is therefore an **efficiency-stability trade-off**, rather than best RMSE alone.

---

## What This Public Repository Includes

- **Sequential SGD** baseline
- **Hogwild!** lock-free OpenMP training
- **Locked SGD** engineering baseline with fine-grained locks
- Biased matrix factorization: global mean + user bias + item bias + latent factors
- Deterministic train / validation / test splitting
- RMSE evaluation
- CSV metric export
- Portable CMake build
- Synthetic smoke-test data generator

### Implementation Note

The public `locked-sgd` implementation is **not** an exact implementation of the DSGD baseline reported in the paper.

The published paper compares:

```text
Hogwild! — lock-free asynchronous SGD
DSGD     — synchronized distributed SGD baseline
A-ASGD   — asymmetric hybrid parallel SGD
```

The current public repository instead provides `locked-sgd` as a compact engineering demonstration of fine-grained synchronization.

See [docs/RESEARCH_IMPLEMENTATION.md](docs/RESEARCH_IMPLEMENTATION.md) for the relationship between the public code and the published research implementation.

---

## Project Structure

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
    └── RESEARCH_IMPLEMENTATION.md
```

---

## Build

### Linux / macOS with OpenMP

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Windows

Use a recent MSVC toolchain with OpenMP support:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

---

## Quick Smoke Test

The synthetic dataset is only for verifying that the pipeline builds and runs. It is **not** a scientific benchmark.

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

---

## Run on MovieLens

After downloading MovieLens ratings data, provide the dataset path through the command line:

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

Available methods in the current public implementation:

```text
sgd
hogwild
locked-sgd
```

---

## Evaluation

The cleaned public implementation reports:

- per-epoch training metrics;
- validation RMSE;
- final test RMSE;
- training time;
- CSV experiment logs.

For experiments produced by this public code, conventional per-model scalability metrics should use:

```text
Speedup:    S(p) = T(1) / T(p)
Efficiency: E(p) = S(p) / p
```

These metrics should be kept distinct from the paper's relative comparison against DSGD.

---

## Research Context

Large-scale recommendation data are sparse, creating opportunities for parallel SGD but also introducing systems challenges:

- concurrent parameter conflicts;
- synchronization overhead;
- load imbalance;
- poor cache and memory locality.

A-ASGD explores a heterogeneous synchronization policy instead of treating all model parameters identically. The central idea is that user-side and item-side parameters can exhibit different contention patterns, so the synchronization strategy can be adapted to the structure of the data.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2025aasgd,
  title     = {An Efficient Hybrid Parallel SGD Algorithm for Matrix Factorization in Recommendation Systems},
  author    = {Li, Peixin and Hu, Qicong},
  booktitle = {2025 6th International Symposium on Computer Engineering and Intelligent Communications (ISCEIC)},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/ISCEIC67854.2025.11405719}
}
```

---

## Author

**Peixin Li**  
M.S. Student, Southwest University

Research interests:

- Recommendation Systems
- Parallel Machine Learning
- Matrix / Tensor Factorization
- Large-scale Optimization

Primary technologies:

`C++` · `Python` · `OpenMP` · `PyTorch` · `CUDA`

---

## License

The cleaned portfolio code included in this repository is released under the MIT License.

The published paper and the complete research implementation used for its experiments are separate from this public portfolio release.
