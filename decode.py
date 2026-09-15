import time

import numpy as np
import octcode
import torch
from mamba_ssm.utils.generation import InferenceParams
from torch.nn.functional import softmax

from model.network import Network
from utils.dataset import create_dataset
from utils.helper import OctRansDecoder, attn_cache_batches
from utils.rw import write_ply_ascii

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_level_ar(model, decoder, ctx):
    symbols = torch.full((ctx.shape[0],), model.cfg["nout"], dtype=torch.long, device=device)

    for cache, s in attn_cache_batches(model, ctx, model.cfg["ctx_win"], attn_bs=4, pred_bs=256):
        b, l = cache.shape[0], cache.shape[1]
        infer_params = [InferenceParams(max_seqlen=l, max_batch_size=b)]
        win_bias = torch.arange(b, device=device) * l

        for i in range(l):
            tgts = symbols[s + (i - 1) + win_bias].unsqueeze(1) if i > 0 else None

            logit = model.pred_ar(cache[:, i : i + 1], tgts, infer_params)

            syms_chunk = decoder.decode_chunk(softmax(logit, dim=-1)[:, 0, :])
            symbols[s + i + win_bias] = syms_chunk.to(device=device, dtype=torch.long)

    return symbols.cpu().numpy() + 1


def decode_level_ms(model, decoder, ctx, stages):
    length = ctx.shape[0]

    symbols = torch.full((length,), model.cfg["nout"], dtype=torch.long, device=device)
    batches = list(attn_cache_batches(model, ctx, model.cfg["ctx_win"], attn_bs=4, pred_bs=4))

    phase = torch.arange(length, device=device).remainder(model.cfg["ctx_win"]).remainder(stages)

    for r in range(stages):
        tgt_idx = torch.where(phase == r)[0]
        if tgt_idx.numel() == 0:
            continue

        probs = torch.zeros((length, model.cfg["nout"]), dtype=torch.float32, device=device)
        for cache, s in batches:
            b, l = cache.shape[0], cache.shape[1]
            tgt_chunk = symbols[s : s + b * l].reshape(b, l)
            logits = model.pred_ms_dec(cache, tgt_chunk)
            probs[s : s + b * l] = softmax(logits, dim=-1).reshape(-1, model.cfg["nout"])

        syms_r = decoder.decode_chunk(probs[tgt_idx])
        symbols[tgt_idx] = syms_r.to(device=device, dtype=torch.long)

    return symbols.cpu().numpy() + 1


def decode_part(model, dataset, decoder, target_level, root_bytes, stages, bin_num):
    octree = octcode.OctreeDec(target_level)

    for level in range(target_level):
        octree.calc_context(level)

        if level == 0:
            symbols = np.frombuffer(root_bytes, dtype=np.uint8).astype(np.int32) + 1
        else:
            ctx = torch.from_numpy(octree.get_context(level)).to(torch.float32).to(device)
            ctx[:, :-1, 0] = ctx[:, :-1, 0] - 1
            ctx[:, -1, 3:6] = (ctx[:, -1, 3:6] / (2 ** target_level)) * 2.0 - 1.0

            symbols = decode_level_ar(model, decoder, ctx) if stages == -1 else decode_level_ms(model, decoder, ctx, stages)

        octree.calc_children(symbols, level)

    coords = octree.get_coords(target_level)
    del octree

    coords = dataset.dequantize(coords, target_level + 2 if dataset.name in ["kitti", "nuscenes"] else target_level + 1, bin_num)

    return coords


@torch.inference_mode()
def decode(model, dataset, bin_path):
    with open(bin_path, "rb") as f:
        stages = np.frombuffer(f.read(4), dtype=np.int32)[0]
        bin_num = np.frombuffer(f.read(4), dtype=np.float32)[0]
        num_parts = np.frombuffer(f.read(4), dtype=np.int32)[0]

        part_meta = []
        for _ in range(num_parts):
            target_level = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            root_bytes = f.read(1)
            part_meta.append((target_level, root_bytes))

        rans_bytes = f.read()

    decoder = OctRansDecoder(rans_bytes)

    torch.cuda.synchronize()
    start_time = time.time()

    coords_ls = [
        decode_part(model, dataset, decoder, target_level, root_bytes, stages, bin_num)
        for target_level, root_bytes in part_meta
    ]
    coords = np.concatenate(coords_ls, axis=0) if len(coords_ls) > 1 else coords_ls[0]

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    return coords, elapsed


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ford", choices=["kitti", "nuscenes", "ford", "qnx"])
    parser.add_argument("--bin_path", type=str, default="./output/Ford_02_vox1mm-0100_14.bin")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(f"configs/{args.dataset}.yaml", "r"))

    model = Network(cfg)
    model.load_state_dict(torch.load(f"ckpts/{args.dataset}.ckpt", weights_only=True))
    model = model.cuda().eval()

    dataset = create_dataset(args.dataset)

    # warm_up
    for i in range(3):
        _ = decode(model, dataset, args.bin_path)

    coords, elapsed = decode(model, dataset, args.bin_path)
    print("Decoded coords shape:", coords.shape)
    print("Decoding time (s):", elapsed)

    write_ply_ascii(args.bin_path.replace(".bin", ".ply"), coords)
