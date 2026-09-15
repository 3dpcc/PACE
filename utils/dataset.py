import glob

import numpy as np
import yaml


class IntrinsicAwareQuantizer:
    def __init__(self, cfg):
        self.lasers_num = cfg["lasers_num"]
        self.lasers_theta = np.array(cfg["lasers_theta"])
        self.lasers_z = np.array(cfg["lasers_z"])

    def convert_coords(self, coords):
        rho = np.hypot(coords[:, 0:1], coords[:, 1:2])
        azi = np.arctan2(coords[:, 1:2], coords[:, 0:1]) + np.pi
        azi = azi % (2 * np.pi)
        z = coords[:, 2:3]

        ele = (z + self.lasers_z[np.newaxis, :]) / rho
        dist = np.abs(ele - self.lasers_theta[np.newaxis, :])
        lid = np.argmin(dist, axis=1)[:, np.newaxis]

        return rho, lid, azi

    def quantize(self, coords, level, max_v):
        rho, lid, azi = self.convert_coords(coords)

        rho_step = 2 ** (18 - level)
        bin_num = max_v // (rho_step)
        azi_step = np.pi * 2 / bin_num

        q_rho = np.floor(rho / rho_step)
        q_azi = np.floor(azi / azi_step)
        q_lid = lid * (bin_num // self.lasers_num)

        q_coords = np.concatenate((q_rho, q_azi, q_lid), axis=-1).astype(np.int32)
        q_coords = np.unique(q_coords, axis=0)

        return q_coords, max_v

    def dequantize(self, geom, level, max_v):
        q_rho, q_azi, q_lid = geom[:, 0:1], geom[:, 1:2], geom[:, 2:3]

        rho_step = 2 ** (18 - level)
        bin_num = max_v // (rho_step)
        azi_step = np.pi * 2 / bin_num

        r_rho = q_rho * rho_step + rho_step / 2
        r_azi = q_azi * azi_step - np.pi + azi_step / 2
        r_lid = q_lid / (bin_num // self.lasers_num)

        r_lid = np.round(r_lid).astype(np.int32)
        r_ele = self.lasers_theta[r_lid]

        x = r_rho * np.cos(r_azi)
        y = r_rho * np.sin(r_azi)
        z = r_rho * r_ele - self.lasers_z[r_lid]

        r_coords = np.concatenate([x, y, z], axis=-1)

        return r_coords


class SphericalQuantizer:
    def __init__(self, qs):
        self.qs = qs

    def car2shp(self, coords):
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]

        hxy = np.hypot(x, y)
        rho = np.hypot(hxy, z)
        ele = np.arctan2(z, hxy)
        azi = np.arctan2(y, x)

        return rho, ele, azi

    def shp2car(self, rho, ele, azi):
        rcos_theta = rho * np.cos(ele)

        x = rcos_theta * np.cos(azi)
        y = rcos_theta * np.sin(azi)
        z = rho * np.sin(ele)

        return np.concatenate([x, y, z], axis=-1)

    def quantize(self, coords, level, max_v):
        rho, ele, azi = self.car2shp(coords)

        rho_max = rho.max()

        rho_step = self.qs / (2 ** level - 1)
        ele_step = np.pi * 1 * rho_step / rho_max
        azi_step = np.pi * 2 * rho_step / rho_max

        q_rho = np.floor(rho / rho_step)
        q_ele = np.floor((ele + np.pi / 2) / ele_step)  # shift ele by pi/2 to make it positive [0, 1pi]
        q_azi = np.floor((azi + np.pi / 1) / azi_step)  # shift azi by pi/1 to make it positive [0, 2pi]

        q_coords = np.concatenate((q_rho, q_ele, q_azi), axis=-1).astype(np.int32)
        q_coords = np.unique(q_coords, axis=0)

        return q_coords, rho_max

    def dequantize(self, coords, level, max_v):
        q_rho, q_ele, q_azi = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]

        rho_step = self.qs / (2 ** level - 1)
        ele_step = np.pi * 1 * rho_step / max_v
        azi_step = np.pi * 2 * rho_step / max_v

        rho = (q_rho + 1 / 2) * rho_step
        ele = (q_ele + 1 / 2) * ele_step - np.pi / 2
        azi = (q_azi + 1 / 2) * azi_step - np.pi / 1

        r_coords = self.shp2car(rho, ele, azi)

        return r_coords


class BaseDataset:
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = cls.__name__.replace("Dataset", "").lower()
        cls._registry[cls.name] = cls


class KITTIDataset(BaseDataset):
    def __init__(self, root_dir="/zhu/data/SemanticKITTI"):
        self.root_dir = root_dir
        self.quantizer = SphericalQuantizer(qs=400.0)
        self.name = "kitti"

    def get_train_paths(self, pattern="*.bin"):
        seqs = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

        paths = []
        for seq in seqs:
            seq_paths = glob.glob(f"{self.root_dir}/train/dataset/sequences/{seq}/velodyne/{pattern}", recursive=True)
            seq_paths = sorted(seq_paths)
            paths.extend(seq_paths)

        return paths

    def get_test_paths(self, pattern="*.bin"):
        seqs = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]

        paths = []
        for seq in seqs:
            seq_paths = glob.glob(f"{self.root_dir}/test/dataset/sequences/{seq}/velodyne/{pattern}", recursive=True)
            seq_paths = sorted(seq_paths)
            paths.extend(seq_paths)

        return paths

    def quantize(self, coords, level):
        return self.quantizer.quantize(coords, level, None)

    def dequantize(self, coords, level, rho_max):
        return self.quantizer.dequantize(coords, level, rho_max)


class nuScenesDataset(BaseDataset):
    def __init__(self, root_dir="/zhu/data/nuScenes"):
        self.root_dir = root_dir
        self.quantizer = SphericalQuantizer(qs=450.0)

    def get_train_paths(self, pattern="*.bin"):
        paths = []
        for idx in range(0, 5):
            for seq in range(0, 12):
                seq_paths = glob.glob(f"{self.root_dir}/train/{idx:02d}/{seq:02d}/{pattern}", recursive=True)
                seq_paths = sorted(seq_paths)
                paths.extend(seq_paths)

        return paths

    def get_test_paths(self, pattern="0[0-8][0-9].bin"):
        paths = []
        for idx in range(5, 10):
            seq_paths = glob.glob(f"{self.root_dir}/test/{idx:02d}/00/{pattern}", recursive=True)
            seq_paths = sorted(seq_paths)
            paths.extend(seq_paths)

        return paths

    def quantize(self, coords, level):
        return self.quantizer.quantize(coords, level, None)

    def dequantize(self, coords, level, rho_max):
        return self.quantizer.dequantize(coords, level, rho_max)


class FordDataset(BaseDataset):
    def __init__(self, root_dir="/zhu/data/Ford", config_path="utils/cfg/ford.yaml"):
        self.root_dir = root_dir
        self.quantizer = IntrinsicAwareQuantizer(yaml.safe_load(open(config_path, "r")))

    def get_train_paths(self, pattern="*.ply"):
        paths = glob.glob(f"{self.root_dir}/train/Ford_01_q_1mm/{pattern}", recursive=True)
        paths = sorted(paths)
        return paths

    def get_test_paths(self, pattern="*.ply"):
        paths = []

        seqs = ["Ford_02_q_1mm", "Ford_03_q_1mm"]
        for seq in seqs:
            seq_paths = glob.glob(f"{self.root_dir}/test/{seq}/{pattern}", recursive=True)
            seq_paths = sorted(seq_paths)
            paths.extend(seq_paths)

        return paths

    def quantize(self, coords, level):
        return self.quantizer.quantize(coords, level, max_v=122880)

    def dequantize(self, coords, level, max_v):
        return self.quantizer.dequantize(coords, level, max_v)


class QNXDataset(BaseDataset):
    def __init__(self, root_dir="/zhu/data/QNX", config_path="utils/cfg/qnx.yaml"):
        self.root_dir = root_dir
        self.quantizer = IntrinsicAwareQuantizer(yaml.safe_load(open(config_path, "r")))

    def get_train_paths(self, pattern="*.ply"):
        paths = []

        seqs = ["qnxadas-motorway-join", "qnxadas-navigating-bends"]
        for seq in seqs:
            seq_paths = glob.glob(f"{self.root_dir}/train/{seq}/{pattern}", recursive=True)
            seq_paths = sorted(seq_paths)
            paths.extend(seq_paths)

        return paths

    def get_test_paths(self, pattern="*.ply"):
        paths = []

        seqs = ["qnxadas-junction-approach", "qnxadas-junction-exit"]
        for seq in seqs:
            seq_paths = glob.glob(f"{self.root_dir}/test/{seq}/{pattern}", recursive=True)
            seq_paths = sorted(seq_paths)
            paths.extend(seq_paths)

        return paths

    def quantize(self, coords, level):
        return self.quantizer.quantize(coords, level, max_v=129024)

    def dequantize(self, coords, level, max_v):
        return self.quantizer.dequantize(coords, level, max_v)


def create_dataset(name):
    if name not in BaseDataset._registry:
        raise ValueError(f"Unknown dataset: {name}, available datasets: {list(BaseDataset._registry.keys())}")
    return BaseDataset._registry[name]()
