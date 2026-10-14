#!/usr/bin/env python3
"""Generate a small MovieLens-shaped CSV for build/smoke testing only."""

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/toy_ratings.csv")
    p.add_argument("--users", type=int, default=100)
    p.add_argument("--items", type=int, default=200)
    p.add_argument("--ratings", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["userId", "movieId", "rating", "timestamp"])
        for t in range(args.ratings):
            user = rng.randint(1, args.users)
            item = rng.randint(1, args.items)
            rating = rng.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            writer.writerow([user, item, rating, t])

    print(output)


if __name__ == "__main__":
    main()
