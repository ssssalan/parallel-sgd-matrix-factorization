# Figures

This directory contains recruiter-facing visualizations re-created from the numeric results reported in the published IEEE paper.

## Included

### `published_training_time.svg`

Total training time for 30 epochs across 1, 2, 4, 8, and 16 threads for:

- A-ASGD
- Hogwild!
- DSGD

### `relative_speedup_vs_dsgd.svg`

Relative speedup using DSGD at the same thread count as the baseline:

```text
RelativeSpeedup(method, p) = T_DSGD(p) / T_method(p)
```

This metric should not be confused with conventional per-model parallel speedup `T(1) / T(p)`.

## Convergence figure policy

The paper includes an RMSE-convergence figure at 16 threads, but the exact per-epoch numeric values are not tabulated in the published text.

To avoid digitizing or fabricating intermediate values, this repository does not currently re-create that convergence curve. It reports only the approximate converged RMSE values stated in the paper text.

The SVG plots in this folder are newly generated visualizations and do not reproduce IEEE page artwork.
