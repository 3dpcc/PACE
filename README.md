# PACE: Post-Causal Entropy Modeling for Learned LiDAR Point Cloud Compression

[![arXiv](https://img.shields.io/badge/Arxiv-2605.01320-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2605.01320)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Introduction

This repository is the official PyTorch implementation of our paper *PACE: Post-Causal Entropy Modeling for Learned LiDAR Point Cloud Compression*.

**Abstract:** LiDAR point cloud compression is vital for autonomous systems to handle massive data from high-resolution sensors. While learned entropy modeling built upon octree structures yields high compression gains, it faces two critical bottlenecks: 1) prohibitive latency, particularly during decoding, caused by causal, multi-stage context modeling; and 2) a rigid performance-latency trade-off, preventing a single model from adapting to varying constraints. To address this, we propose PACE, a framework that reformulates ancestral context aggregation as a non-causal backbone and confines causality to a lightweight, stage-scalable predictor, eliminating repetitive backbone executions and reducing computational overhead. The predictor supports an arbitrary number of prediction stages, enabling seamless adaptation across diverse performance-latency trade-offs without reloading parameters. Experiments demonstrate that PACE sets a new state-of-the-art in compression efficiency, achieving notable BD-BR savings and reducing decoding latency by over 90% in autoregressive mode.

## Roadmap
- [ ] Release training code
- [x] ~~Release inference code~~
- [x] ~~Release checkpoint~~

## ️Environment Setup

```bash
conda create -n pace python=3.12
conda activate pace

# Install pytorch
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install torch-geometric and torch-cluster
pip install torch-geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# Install mamba (a C++/CUDA toolchain is required)
pip install mamba-ssm

# Install the octree and rANS extensions
cd ./extension/octcode && pip install --no-build-isolation . && cd -
cd ./extension/octrans && pip install --no-build-isolation . && cd -

# Install other dependencies
pip install numpy pandas pyyaml tqdm joblib open3d
```
Tested with `torch 2.8.0+cu128`, `mamba_ssm 2.2.5`.

## Data Preparation

We support SemanticKITTI, Ford, nuScenes and QNX. Set `root_dir` of the corresponding dataset class in `utils/dataset.py` to your own data root, or pass explicit file paths via `--paths` to `encode.py`.

A few example point clouds are already included under `data/`.

**Note:** For QNX, we removed duplicate points from the official point clouds before testing.

## Usage

Download the pre-trained weights from [this link](https://box.nju.edu.cn/d/67512ab819ea4a6e8771/) and place them under `./ckpts/`.

### Encoding

```bash
python encode.py --dataset kitti --code_type fp --multi_level 1 --paths ./data/kitti/11_000000.bin --level_range 14
```

`--code_type` selects the coding mode: `fp` (fully parallel, one-stage), `ar` (autoregressive), or `Ns` (N-stage, e.g. `2s`, `3s`, `4s`...).


To reproduce the rate-distortion curves over the full test split, omit `--paths` and sweep the depths of each dataset:

```bash
python encode.py --dataset kitti    --code_type fp --multi_level 1 --level_range 11 12 13 14 15 16
python encode.py --dataset nuscenes --code_type fp --multi_level 1 --level_range 11 12 13 14 15 16
python encode.py --dataset ford     --code_type fp --multi_level 1 --level_range 11 12 13 14 15 16 17
python encode.py --dataset qnx      --code_type fp --multi_level 0 --level_range 11 12 13 14 15 16 17
```

Replace `--code_type` with `ar` or `Ns` to trade decoding latency against compression ratio using the same checkpoint.

### Decoding

```bash
python decode.py --dataset kitti --bin_path xxxxxx.bin
```

### Evaluation

**Bitrate (bpp):** aggregate the log written by `encode.py`.

```bash
python scripts/summarize.py --path ./results/results_kitti_fp.csv
```

**Distortion (PSNR):**
```bash
chmod +x ./extension/pc_error_d
python scripts/prepare_test.py --dataset kitti --multi_level 1 --depths 11 12 13 14 15 16
python scripts/eval.py --dataset kitti --multi_level 1 --depths 11 12 13 14 15 16
```

**Note:** `--multi_level` must be identical across `encode.py`, `prepare_test.py` and `eval.py`.

## Citation

If you find this work useful, we would appreciate a citation:

```bibtex
@inproceedings{zhu2026pace,
  title={{PACE}: Post-Causal Entropy Modeling for Learned {LiDAR} Point Cloud Compression},
  author={Jiahao Zhu and Kang You and Dandan Ding and Zhan Ma},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```
