# Figures

The recruiter-facing figures for this repository should follow the definitions used in the published paper.

Recommended figures:

1. **Total training time vs. thread count** — A-ASGD, DSGD, and Hogwild!, using the published 30-epoch results.
2. **RMSE convergence at 16 threads** — corresponding to the paper's convergence comparison.
3. **Relative speedup vs. DSGD** — explicitly defined as `T_DSGD(p) / T_method(p)`.

Do not mix these paper figures with conventional per-model speedup `T(1) / T(p)` unless the metric is clearly labeled.

The current repository does not embed copyrighted IEEE page images. Re-created plots should be generated from documented numeric results.
