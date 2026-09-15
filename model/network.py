import torch
import torch.nn as nn

from model.nn import NodeEmbedding, SelfAttentionBlock, Predictor


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.node_emb = NodeEmbedding(self.cfg)
        self.occu_emb = nn.Embedding(self.cfg["ntoken"], self.cfg["ninp"], padding_idx=self.cfg["ntoken"] - 1)

        self.attn_block = SelfAttentionBlock(self.cfg["ninp"], self.cfg["nhead"], self.cfg["nlayer"])

        self.pred = Predictor(self.cfg)

    def get_attn_cache(self, ctx):
        return self.attn_block(self.node_emb(ctx), mask=None)

    def _pred_shift(self, feats, occu_emb):
        occu_emb = torch.roll(occu_emb, shifts=1, dims=1)
        occu_emb[:, 0].zero_()
        return self.pred(feats + occu_emb)

    def pred_ar(self, feats, tgts, infer_params):
        if tgts is not None:
            feats = feats + self.occu_emb(tgts.long())
        return self.pred.step(feats, infer_params)

    def pred_ms_enc(self, feats, occu_emb, k):
        b, l, _ = feats.size()
        phase = torch.arange(l, device=feats.device).remainder(k)

        outs = [
            self._pred_shift(feats, occu_emb * (phase < r).reshape(1, l, 1))
            for r in range(min(k, l))
        ]

        output = torch.zeros([b, l, self.cfg["nout"]], dtype=outs[0].dtype, device=outs[0].device)
        for r, out_r in enumerate(outs):
            if self.training:
                output[:, r::k] = out_r[:, r::k]
            else:
                output[:, r::k] = torch.nn.functional.softmax(out_r[:, r::k], dim=-1)

        return output

    def pred_ms_dec(self, feats, tgts):
        return self._pred_shift(feats, self.occu_emb(tgts.long()))
