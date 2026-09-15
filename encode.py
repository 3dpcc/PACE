import os
import time

import numpy as np
import octcode
import pandas as pd
import torch

from mamba_ssm.utils.generation import InferenceParams
from torch.nn.functional import softmax
from tqdm import tqdm

from model.network import Network
from utils.dataset import create_dataset
from utils.helper import OctRansEncoder, attn_cache_batches
from utils.rw import read_coords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_code_type(code_type):
    match code_type:
        case "fp":
            return 1
        case "ar":
            return -1
        case _ if code_type.endswith("s") and code_type[:-1].isdigit():
            return int(code_type[:-1])
        case _:
            raise ValueError(f"Unknown code_type: {code_type}")


def partition_octree(coords, multi_level):
    if not multi_level:
        level = int(np.ceil(np.log2(coords.max())))
        nodes = octcode.build_octree(coords, level)
        return [(nodes, level)]

    rules = [(4, 0.00, 0.25), (2, 0.25, 0.50), (1, 0.50, 1.00)]

    parts = []
    for stride, lo, hi in rules:
        pts = np.unique(coords // stride, axis=0) if stride > 1 else coords
        level = int(np.ceil(np.log2(pts.max())))
        rho = pts[:, 0] / 2 ** level
        pts = pts[(rho >= lo) & (rho < hi)]
        nodes = octcode.build_octree(pts, level)
        parts.append((nodes, level))

    return parts


def prepare_octree_nodes(nodes):
    nodes = torch.from_numpy(nodes).to(torch.float32).to(device)

    nodes[:, :, 0] = nodes[:, :, 0] - 1

    levels = nodes[:, -1, 1]
    max_level = levels.max().item() + 1
    nodes[:, -1, 3:6] = (nodes[:, -1, 3:6] / (2 ** max_level)) * 2.0 - 1.0

    return nodes

# NOTE: parallel mode is numerically close but not bit-exact vs the
# sequential step path used at decode time; bitstreams from this fast
# path may fail to decode. Use only for quick verification.
def encode_level_ar_parallel(model, ctx):
    length = ctx.shape[0]

    occupys = ctx[:, -1, 0].long()
    probs = torch.zeros((length, model.cfg["nout"]), dtype=torch.float32, device=device)

    for cache, s in attn_cache_batches(model, ctx, model.cfg["ctx_win"], attn_bs=4, pred_bs=4):
        b, l = cache.shape[0], cache.shape[1]
        occu_emb = model.occu_emb(occupys[s : s + b * l].reshape(b, l))
        output = model._pred_shift(cache, occu_emb)
        probs[s : s + b * l] = softmax(output, dim=-1).reshape(-1, model.cfg["nout"])

    return probs, occupys

def encode_level_ar(model, ctx):
    occupys = ctx[:, -1, 0].long()

    prob_chunks = []
    sym_chunks = []
    for cache, s in attn_cache_batches(model, ctx, model.cfg["ctx_win"], attn_bs=4, pred_bs=256):
        b, l = cache.shape[0], cache.shape[1]
        infer_params = [InferenceParams(max_seqlen=l, max_batch_size=b)]
        win_bias = torch.arange(b, device=device) * l

        for i in range(l):
            tgts = occupys[s + (i - 1) + win_bias].unsqueeze(1) if i > 0 else None

            logit = model.pred_ar(cache[:, i : i + 1], tgts, infer_params)

            prob_chunks.append(softmax(logit, dim=-1)[:, 0, :])
            sym_chunks.append(occupys[s + i + win_bias])

    return torch.cat(prob_chunks, dim=0), torch.cat(sym_chunks, dim=0)


def encode_level_ms(model, ctx, stages):
    length = ctx.shape[0]

    occupys = ctx[:, -1, 0].long()
    probs = torch.zeros((length, model.cfg["nout"]), dtype=torch.float32, device=device)

    for cache, s in attn_cache_batches(model, ctx, model.cfg["ctx_win"], attn_bs=4, pred_bs=4):
        b, l = cache.shape[0], cache.shape[1]
        occu_emb = model.occu_emb(occupys[s : s + b * l].reshape(b, l))
        output = model.pred_ms_enc(cache, occu_emb, stages)
        probs[s : s + b * l] = output.reshape(-1, model.cfg["nout"])

    phase = torch.arange(length, device=device).remainder(model.cfg["ctx_win"]).remainder(stages)
    enc_idx = torch.cat([torch.where(phase == r)[0] for r in range(stages)], dim=0)

    return probs[enc_idx], occupys[enc_idx]


def encode_part(model, nodes, target_level, stages, encoder):
    nodes = prepare_octree_nodes(nodes)

    levels = nodes[:, -1, 1]
    _, counts = torch.unique_consecutive(levels, return_counts=True)

    offset = 0
    root_bytes = None

    for level in range(target_level):
        count = counts[level].item()
        level_nodes = nodes[offset : offset + count]

        if level == 0:
            root_bytes = level_nodes[:, -1, 0].cpu().numpy().astype(np.uint8).tobytes()
        else:
            result = encode_level_ar(model, level_nodes) if stages == -1 else encode_level_ms(model, level_nodes, stages)
            encoder.add_chunk(*result)

        offset += count

    return root_bytes


@torch.inference_mode()
def encode(model, dataset, path, save_path, qlevel, stages, multi_level):
    coords = read_coords(path, dataset.name, dtype=np.float32)
    points = coords.shape[0]

    torch.cuda.synchronize()
    start_time = time.time()

    q_coords, bin_num = dataset.quantize(coords, qlevel + 2 if multi_level else qlevel)

    parts = partition_octree(q_coords, multi_level)

    encoder = OctRansEncoder()
    part_meta = []

    for nodes, level in parts:
        root_bytes = encode_part(model, nodes, level, stages, encoder)
        part_meta.append((level, root_bytes))

    rans_bytes = encoder.finalize()
    total_bits = len(rans_bytes) * 8

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    with open(save_path, "wb") as f:
        f.write(np.array(stages, dtype=np.int32).tobytes())
        f.write(np.array(bin_num, dtype=np.float32).tobytes())
        f.write(np.array(len(part_meta), dtype=np.int32).tobytes())
        for level, root_bytes in part_meta:
            f.write(np.array(level, dtype=np.int32).tobytes())
            f.write(root_bytes)
        f.write(rans_bytes)

    return {
        "total_bits": total_bits,
        "total_points": points,
        "enc_time": elapsed,
    }


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ford", choices=["kitti", "nuscenes", "ford", "qnx"])
    parser.add_argument("--code_type", type=str, default="fp", help="fp, 2s, 3s, 4s, ... , ar")
    parser.add_argument("--level_range", type=int, nargs="+", default=[11, 12, 13, 14, 15, 16, 17])
    parser.add_argument("--multi_level", type=int, default=1, choices=[0, 1])
    parser.add_argument("--paths", type=str, nargs="+", default=None)
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--log_dir", type=str, default="./results")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(f"configs/{args.dataset}.yaml", "r"))
    cfg["code_type"] = args.code_type
    stages = parse_code_type(args.code_type)

    model = Network(cfg)
    model.load_state_dict(torch.load(f"ckpts/{args.dataset}.ckpt", weights_only=True))
    model = model.cuda().eval()

    dataset = create_dataset(args.dataset)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    log_path = os.path.join(args.log_dir, f"results_{args.dataset}_{args.code_type}.csv")

    paths = args.paths if args.paths is not None else dataset.get_test_paths()
    print(f"Encoding {len(paths)} files from {args.dataset} dataset")

    # warm up
    for i in range(3):
        encode(model, dataset, paths[0], os.devnull, args.level_range[0], stages, args.multi_level)

    for idx, path in enumerate(tqdm(paths)):
        row_data = {"path": path}

        for level in args.level_range:
            out_path = os.path.join(args.out_dir, f"{os.path.basename(path)[:-4]}_{level}.bin")
            result = encode(model, dataset, path, out_path, level, stages, args.multi_level)

            user_bpp = result["total_bits"] / result["total_points"]
            wall_bpp = os.stat(out_path).st_size * 8 / result["total_points"]
            memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            row_data[f"d{level}_user_bpp"] = user_bpp
            row_data[f"d{level}_wall_bpp"] = wall_bpp
            row_data[f"d{level}_bits"] = result["total_bits"]
            row_data[f"d{level}_points"] = result["total_points"]
            row_data[f"d{level}_time"] = result["enc_time"]
            row_data[f"d{level}_memory_gb"] = memory_gb

            torch.cuda.reset_peak_memory_stats()

        df = pd.DataFrame([row_data])
        if idx == 0:
            df.to_csv(log_path, index=False)
        else:
            df.to_csv(log_path, mode="a", header=False, index=False)
