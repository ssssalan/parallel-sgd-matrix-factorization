# Research implementation and public release

This repository accompanies the first-author IEEE paper:

**An Efficient Hybrid Parallel SGD Algorithm for Matrix Factorization in Recommendation Systems**  
Peixin Li, Qicong Hu  
2025 6th International Symposium on Computer Engineering and Intelligent Communications (ISCEIC), IEEE  
DOI: https://doi.org/10.1109/ISCEIC67854.2025.11405719

## Scope of this public repository

The code in this repository is a cleaned portfolio implementation for demonstrating:

- biased matrix factorization;
- C++ / OpenMP parallel training;
- sequential SGD;
- Hogwild!-style lock-free training;
- a fine-grained locked SGD baseline;
- reproducible train / validation / test evaluation;
- CMake-based build and smoke-test infrastructure.

The complete research implementation used to produce the published A-ASGD results is not distributed in this portfolio repository.

## Relationship to the paper

The paper's A-ASGD method uses an asymmetric update strategy:

- user-side factor and bias updates are lock-free;
- item-side updates use fine-grained block synchronization;
- the interaction space is partitioned into a 2N × 4N block grid;
- workers randomly sample blocks;
- item-block synchronization uses a non-blocking try-lock policy.

The public `locked-sgd` method in this repository is an engineering baseline and should not be interpreted as the DSGD baseline reported in the paper.

## Published benchmark protocol

The paper reports experiments on MovieLens-20M with an 80% / 20% train-test split, implemented in C++ with OpenMP and evaluated on an Intel Core i7-13700 workstation with 32 GB RAM.

Published paper results and results produced by the cleaned public implementation should be treated as separate experiment sources unless the full paper protocol is reproduced exactly.
