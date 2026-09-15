import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.rw import read_coords, write_ply_o3d
from utils.dataset import create_dataset

rules = [(0.00, 0.25), (0.25, 0.50), (0.50, 1.00)]

def make_part(dataset, coords, depth, part_idx):
    pc, rho_max = dataset.quantize(coords, depth)

    lo, hi = rules[part_idx]
    level = int(np.ceil(np.log2(pc.max())))
    rho = pc[:, 0] / 2 ** level

    return pc[(rho >= lo) & (rho < hi)], rho_max


def process_point_clouds(dataset, point_path, ori_path, rec_path, depth, multi_level, ori_save=False):
    point_cloud = read_coords(point_path, dataset.name, dtype="float32")

    base_name = os.path.splitext(os.path.basename(point_path))[0]

    if ori_save:
        ori_filename = os.path.join(ori_path, f"{base_name}_ori.ply")
        write_ply_o3d(ori_filename, point_cloud, dtype="float32", normal=True, knn=20)
        return

    if multi_level:
        pc_0, rho_max_0 = make_part(dataset, point_cloud, depth + 0, part_idx=0)
        pc_1, rho_max_1 = make_part(dataset, point_cloud, depth + 1, part_idx=1)
        pc_2, rho_max_2 = make_part(dataset, point_cloud, depth + 2, part_idx=2)

        pc_0 = dataset.dequantize(pc_0, depth + 0, rho_max_0)
        pc_1 = dataset.dequantize(pc_1, depth + 1, rho_max_1)
        pc_2 = dataset.dequantize(pc_2, depth + 2, rho_max_2)

        coords_rec = np.concatenate([pc_0, pc_1, pc_2], axis=0)
    else:
        pc, rho_max = dataset.quantize(point_cloud, depth)
        coords_rec = dataset.dequantize(pc, depth, rho_max)

    tag = "ml" if multi_level else "sl"
    rec_filename = os.path.join(rec_path, f"{base_name}_rec_{tag}_{depth}.ply")
    write_ply_o3d(rec_filename, coords_rec, dtype="float32", normal=False)

    return


if __name__ == "__main__":
    import argparse
    import joblib

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="kitti", choices=["kitti", "nuscenes", "ford", "qnx"])
    parser.add_argument("--multi_level", type=int, default=0, choices=[0, 1])
    parser.add_argument("--depths", type=int, nargs="+", default=[16, 15, 14, 13, 12, 11])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset = create_dataset(args.dataset)
    point_paths = dataset.get_test_paths()

    if args.limit is not None:
        point_paths = point_paths[:args.limit]

    to_save_path = f"./psnr/{args.dataset}"

    ori_path = os.path.join(to_save_path, "ori")
    rec_path = os.path.join(to_save_path, "rec")

    os.makedirs(ori_path, exist_ok=True)
    os.makedirs(rec_path, exist_ok=True)

    if args.dataset in ["kitti", "nuscenes"]:
        joblib.Parallel(n_jobs=16, verbose=10, pre_dispatch="all")([
            joblib.delayed(process_point_clouds)(dataset, point_path, ori_path, rec_path, None, args.multi_level, ori_save=True)
            for point_path in point_paths
        ])

    joblib.Parallel(n_jobs=16, verbose=10, pre_dispatch="all")([
        joblib.delayed(process_point_clouds)(dataset, point_path, ori_path, rec_path, depth, args.multi_level)
        for point_path in point_paths
        for depth in args.depths
    ])

