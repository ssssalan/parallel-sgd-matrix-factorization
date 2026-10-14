# Data

This repository does **not** redistribute MovieLens or other recommendation datasets.

For MovieLens 20M/25M, download the dataset from the official GroupLens website and pass the path to `ratings.csv` at runtime:

```bash
./build/mf_train --data /path/to/ratings.csv --method hogwild --threads 8
```

The expected CSV schema is MovieLens-compatible:

```text
userId,movieId,rating,timestamp
```

For a local smoke test, generate a small synthetic file:

```bash
python3 scripts/generate_toy_ratings.py
```
