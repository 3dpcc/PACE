import os

import numpy as np
import open3d as o3d

def read_ply_o3d(path, dtype=np.float32):
    pcd = o3d.io.read_point_cloud(path)
    coords = np.asarray(pcd.points).astype(dtype)

    return coords

def read_ply_ascii(path, dtype=np.float32):
    files = open(path, "r")
    data = []

    for _, line in enumerate(files):
        wordslist = line.split(" ")
        try:
            line_values = []
            for _, v in enumerate(wordslist):
                if v == "\n":
                    continue
                line_values.append(float(v))
        except ValueError:
            continue
        data.append(line_values)

    data = np.array(data)
    coords = data[:, 0:3].astype(dtype)

    return coords


def read_bin_kitti(path, dtype=np.float32):
    data = np.fromfile(path, dtype).reshape(-1, 4)
    coords = data[:, :3]

    return coords

def read_bin_nuscenes(path, dtype=np.float32):
    data = np.fromfile(path, dtype).reshape(-1, 5)
    coords = data[:, :3]

    return coords

def write_ply_ascii(path, coords, dtype="float32"):
    if os.path.exists(path):
        os.remove(path)

    with open(path, "a+") as f:
        f.writelines(["ply\n", "format ascii 1.0\n"])
        f.write(f"element vertex {coords.shape[0]}\n")
        f.writelines(["property float x\n", "property float y\n", "property float z\n"])
        f.write("end_header\n")

        coords = coords.astype(dtype)
        for p in coords:
            f.writelines(f"{p[0]} {p[1]} {p[2]}\n")
    return

def write_ply_o3d(path, coords, dtype="int32", normal=False, knn=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(dtype))
    if normal:
        assert knn is not None
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    o3d.io.write_point_cloud(path, pcd, write_ascii=True)
    f = open(path)
    lines = f.readlines()
    lines[4] = "property float x\n"
    lines[5] = "property float y\n"
    lines[6] = "property float z\n"
    if normal:
        lines[7] = "property float nx\n"
        lines[8] = "property float ny\n"
        lines[9] = "property float nz\n"
    fo = open(path, "w")
    fo.writelines(lines)

    return

def read_coords(path, dataset, dtype=np.float32):
    dataset = dataset.lower()

    match dataset:
        case "kitti":
            coords = read_bin_kitti(path, dtype=dtype)
        case "nuscenes":
            coords = read_bin_nuscenes(path, dtype=dtype)
        case "ford" | "qnx":
            coords = read_ply_o3d(path, dtype=dtype)
        case _:
            raise NotImplementedError(f"Dataset {dataset} not supported yet.")

    return coords
