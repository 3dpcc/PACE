import math

import torch
import torch.nn as nn
from torch_cluster import knn_graph
from torch_geometric.nn import MessagePassing

from model.mamba import MambaBlock


class SwiGLU(nn.Module):
    def __init__(self, ninp):
        super().__init__()

        self.w1 = nn.Linear(ninp, 688)
        self.w2 = nn.Linear(ninp, 688)
        self.w3 = nn.Linear(688, ninp)

        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.w3(self.w1(x) * self.silu(self.w2(x)))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, ninp, nhead):
        super().__init__()

        assert ninp % nhead == 0

        self.ninp = ninp
        self.nhead = nhead
        self.d_k = ninp // nhead

        self.qkv_proj = nn.Linear(ninp, ninp * 3)
        self.out_proj = nn.Linear(ninp, ninp)

        self.scale = 1 / math.sqrt(self.d_k)

    def forward(self, x, mask):
        b, l, c = x.size()

        qkv = self.qkv_proj(x)
        qkv = qkv.view(b, l, self.nhead, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(b, l, c)

        out = self.out_proj(attn_output)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(self, ninp, nhead):
        super().__init__()

        self.attn = MultiHeadSelfAttention(ninp, nhead)

        self.ffn = SwiGLU(ninp)

        self.ln_0 = nn.LayerNorm(ninp)
        self.ln_1 = nn.LayerNorm(ninp)

    def forward(self, src, mask):
        out = self.attn(src, mask)
        src = self.ln_0(src + out)

        out = self.ffn(src)
        src = self.ln_1(src + out)

        return src


class SelfAttentionBlock(nn.Module):
    def __init__(self, ninp, nhead, nlayers):
        super().__init__()

        self.layers = nn.ModuleList([
            SelfAttentionLayer(ninp, nhead)
            for _ in range(nlayers)
        ])

    def forward(self, src, mask):
        for layer in self.layers:
            src = layer(src, mask)

        return src


class Predictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.mamba_block = MambaBlock(cfg["ninp"], 1)

        self.pred = nn.Sequential(
            nn.Linear(cfg["ninp"], cfg["ninp"]),
            nn.GELU(),
            nn.Linear(cfg["ninp"], cfg["nout"]),
        )

    @torch._dynamo.disable
    def forward(self, feats):
        feats = self.mamba_block(feats)
        return self.pred(feats)

    def step(self, feats, infer_params):
        feat = self.mamba_block.step(feats, infer_params)
        return self.pred(feat)


class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln(self.embedding(x))
        return x


class GatedEdgeConv(MessagePassing):
    def __init__(self, ninp, nout, aggr="max"):
        super().__init__(aggr=aggr)

        self.in_proj = nn.Sequential(
            nn.Linear(ninp * 2, nout),
            nn.GELU(),
            nn.Linear(nout, nout),
        )

        self.gate = nn.Sequential(
            nn.Linear(nout, nout),
            nn.SiLU(),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(nout, nout),
            nn.GELU(),
        )

    def forward(self, x, edge_index):
        # x: (N, ninp), pos: (N, pos_dim)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        msg = torch.cat([x_i, x_j - x_i], dim=-1)

        msg = self.in_proj(msg)
        g = self.gate(msg)
        msg = self.out_proj(msg * g)

        return msg


class IRNBlock(nn.Module):
    def __init__(self, ninp):
        super().__init__()

        self.branch_1 = nn.Sequential(
            nn.Conv1d(ninp // 1, ninp // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(ninp // 4, ninp // 4, kernel_size=3, padding=1, groups=ninp // 4),
            nn.SiLU(),
            nn.Conv1d(ninp // 4, ninp // 2, kernel_size=1),
        )

        self.branch_2 = nn.Sequential(
            nn.Conv1d(ninp // 1, ninp // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(ninp // 4, ninp // 2, kernel_size=1),
        )

    def forward(self, x):
        # x: (b, l, c)
        x = x.permute(0, 2, 1)  # (b, c, l)

        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        x = torch.cat([x1, x2], dim=1) + x  # (b, c, l)

        x = x.permute(0, 2, 1)  # (b, l, c)

        return x


class NodeEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # 1d embeddings
        self.occ_emb = EmbeddingLayer(cfg["ntoken"], 80, padding_idx=cfg["ntoken"] - 1)
        self.lvl_emb = EmbeddingLayer(cfg["nlevel"], 8)
        self.oct_emb = EmbeddingLayer(cfg["noctant"], 8)

        # 3d embeddings
        self.pos_emb = nn.Sequential(
            nn.Linear(3, 256, bias=False),
            nn.LayerNorm(256),
        )

        self.conv = IRNBlock(256)

        self.edge_conv = GatedEdgeConv(cfg["ninp"], cfg["ninp"])

        self.ln_1 = nn.LayerNorm(cfg["ninp"])
        self.ln_2 = nn.LayerNorm(cfg["ninp"])

    @torch._dynamo.disable
    def build_sparse_graph(self, pos: torch.Tensor, num_win=1, k=16):
        # pos: (b, l, 3)
        b, l, _ = pos.size()

        pos = pos.reshape(b * l, -1).contiguous()
        batch_idx = torch.arange(b * num_win, device=pos.device).repeat_interleave(l // num_win)
        edge_idx = knn_graph(pos, k=k, batch=batch_idx, loop=False)

        return edge_idx

    def forward(self, source: torch.Tensor):
        # source: (b, l, k, nfeat)
        b, l, _, _ = source.size()

        occ = source[:, :, :-1, 0].long()
        lvl = source[:, :, -1, 1].long()
        oct = source[:, :, -1, 2].long()
        pos = source[:, :, -1, 3:6].float()

        lvl = torch.clamp(lvl, max=self.cfg["nlevel"] - 1)

        occ_emb = self.occ_emb(occ).reshape(b, l, -1)
        lvl_emb = self.lvl_emb(lvl).reshape(b, l, -1)
        oct_emb = self.oct_emb(oct).reshape(b, l, -1)

        feats_1d = torch.cat([occ_emb, lvl_emb, oct_emb], dim=-1)
        feats_3d = self.pos_emb(pos)

        out_1 = self.ln_1(self.conv(feats_1d))

        edge_idx = self.build_sparse_graph(pos)
        out_2 = self.edge_conv(
            (out_1 + feats_3d).reshape(b * l, -1),
            edge_idx,
        ).reshape(b, l, -1)

        out = self.ln_2(out_2 + out_1)

        return out
