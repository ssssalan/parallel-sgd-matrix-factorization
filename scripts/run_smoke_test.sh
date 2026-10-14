#!/usr/bin/env bash
set -euo pipefail
python3 scripts/generate_toy_ratings.py --output data/toy_ratings.csv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/mf_train --data data/toy_ratings.csv --method hogwild --threads 4 --epochs 3 --output results/smoke_test.csv
