# Results

This repository separates **published paper results** from results produced by the cleaned public implementation.

## Published paper results

The IEEE paper reports MovieLens-20M experiments using C++ / OpenMP on an Intel Core i7-13700 workstation.

The transcribed numeric values used by the recruiter-facing plots are stored in:

- [`published_paper_results.csv`](published_paper_results.csv)

Fields:

- `threads`
- `total_time_aasgd_s`
- `total_time_dsgd_s`
- `total_time_hogwild_s`
- `relative_aasgd_vs_dsgd`
- `relative_hogwild_vs_dsgd`

The relative comparison is defined as:

```text
T_DSGD(p) / T_method(p)
```

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

`example_metrics.csv` documents the public output schema and should not be treated as a published scientific result.
