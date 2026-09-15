import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor


def number_in_line(line):
    wordlist = line.split(" ")
    for _, item in enumerate(wordlist):
        try:
            number = float(item)
        except ValueError:
            continue

    return number


def pc_error(infile1, infile2, resolution, normal=False, show=False, details=False):
    headers = ["mseF      (p2point)", "mseF,PSNR (p2point)"]
    if details:
        headers += [
            "mse1      (p2point)",
            "mse1,PSNR (p2point)",
            "mse2      (p2point)",
            "mse2,PSNR (p2point)",
        ]

    command = str(
        "./extension/pc_error_d"
        + " -a "
        + infile1
        + " -b "
        + infile2
        + " --dropdups=2"
        + " --neighborsProc=1"
        + " --resolution="
        + str(resolution)
    )

    if normal:
        headers += ["mseF      (p2plane)", "mseF,PSNR (p2plane)"]
        command = str(command + " -n " + infile1)
    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    results = {}
    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding="utf-8")  # python3.
        if show:
            print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value
        c = subp.stdout.readline()

    return results

if __name__ == "__main__":
    import argparse

    import pandas as pd
    from tqdm import tqdm

    from utils.dataset import create_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="kitti", choices=["kitti", "nuscenes", "ford", "qnx"])
    parser.add_argument("--multi_level", type=int, default=0, choices=[0, 1])
    parser.add_argument("--depths", type=int, nargs="+", default=[16, 15, 14, 13, 12, 11])
    args = parser.parse_args()

    resolution = 59.70 if args.dataset in ["kitti", "nuscenes"] else 30000
    tag = "ml" if args.multi_level else "sl"

    out_path = f"./psnr/{args.dataset}"
    log_path = f"psnr_results_{args.dataset}_{tag}.csv"

    if args.dataset in ["kitti", "nuscenes"]:
        ori_paths = sorted(glob.glob(f"{out_path}/ori/*_ori.ply"))
    else:
        ori_paths = create_dataset(args.dataset).get_test_paths()

    for idx, depth in enumerate(args.depths):
        new_row = {"depth": depth}

        rec_paths = sorted(glob.glob(f"{out_path}/rec/*_rec_{tag}_{depth}.ply"))

        assert len(ori_paths) == len(rec_paths)

        with ThreadPoolExecutor(max_workers=16) as executor:
            results_list = list(tqdm(executor.map(
                lambda p: pc_error(p[0], p[1], resolution=resolution, normal=True, show=False, details=False),
                zip(ori_paths, rec_paths)
            ), total=len(ori_paths)))

        d1_psnr_list = [r["mseF,PSNR (p2point)"] for r in results_list]
        d2_psnr_list = [r["mseF,PSNR (p2plane)"] for r in results_list]

        average_d1_psnr = sum(d1_psnr_list) / len(d1_psnr_list)
        average_d2_psnr = sum(d2_psnr_list) / len(d2_psnr_list)

        print(f"Depth: {depth}, Average D1 PSNR: {average_d1_psnr:.4f}, Average D2 PSNR: {average_d2_psnr:.4f}")

        new_row["d1_psnr"] = average_d1_psnr
        new_row["d2_psnr"] = average_d2_psnr

        results = pd.DataFrame([new_row])

        if idx == 0:
            results.to_csv(log_path, index=False)
        else:
            results.to_csv(log_path, mode="a", header=False, index=False)
