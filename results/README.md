# Results

This repository distinguishes between two sources of results.

## Published paper results

The IEEE paper reports MovieLens-20M experiments using C++ / OpenMP on an Intel Core i7-13700 workstation.

The published 30-epoch training times are summarized in the root README.

These values originate from the published research implementation, not from the cleaned public portfolio code.

## Public implementation results

The cleaned public implementation can produce its own CSV experiment logs for:

- sequential SGD;
- Hogwild!;
- locked SGD.

Recommended fields include:

- method;
- thread count;
- latent rank;
- validation RMSE;
- final test RMSE;
- training time;
- per-model speedup `T(1) / T(p)`;
- parallel efficiency `S(p) / p`.

`example_metrics.csv` documents the output schema and should not be treated as a published scientific result.
