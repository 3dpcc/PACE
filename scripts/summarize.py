import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import argparse

import pandas as pd


def summarize(path):
    df = pd.read_csv(path)

    levels = sorted({m.group(1) for c in df.columns if (m := re.match(r"(d\d+)_bits$", c))})

    print(f"num_rows: {len(df)}")
    for level in levels:
        total_bits = df[f"{level}_bits"].sum()
        total_points = df[f"{level}_points"].sum()
        bpp = total_bits / total_points
        print(f"{level}: bpp={bpp:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="results_ford_fp.csv")
    args = parser.parse_args()

    summarize(args.path)
