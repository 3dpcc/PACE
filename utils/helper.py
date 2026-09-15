import octrans
import torch


def to_pinned_cpu(x):
    out = torch.empty(x.size(), dtype=x.dtype, device="cpu", pin_memory=True)
    out.copy_(x)
    return out


def quantize_pmf(prob):
    prob = to_pinned_cpu(prob)
    s = prob.shape[-1]
    cdf = torch.floor(prob * (65536 - s) + 1.0).cumsum(-1)
    cdf = cdf.clamp(max=65535)
    cdf[:, -1] = 65535
    return cdf.to(torch.uint16).contiguous()


class OctRansEncoder:
    def __init__(self):
        self._encoder = octrans.RansEncoder()
        self._chunks = []

    def add_chunk(self, prob, symbol):
        cdf = quantize_pmf(prob)
        symbol = to_pinned_cpu(symbol.to(torch.int64)).to(torch.uint16).contiguous()
        self._chunks.append((cdf, symbol))

    def finalize(self):
        for cdf, symbol in reversed(self._chunks):
            self._encoder.encode(cdf, symbol)
        return self._encoder.flush()


class OctRansDecoder:
    def __init__(self, byte_stream):
        self._decoder = octrans.RansDecoder()
        self._decoder.flush(byte_stream)

    def decode_chunk(self, prob):
        cdf = quantize_pmf(prob)
        out = torch.empty((cdf.shape[0],), dtype=torch.uint16)
        self._decoder.decode(cdf, out)
        return out.to(torch.int64)


def attn_cache_batches(model, ctx, ctx_win, attn_bs, pred_bs):
    length = ctx.shape[0]
    win_q, win_r = divmod(length, ctx_win)

    if win_q > 0:
        win_ctx = ctx[: win_q * ctx_win].reshape(win_q, ctx_win, 4, 6)
        full_cache = torch.cat([
            model.get_attn_cache(win_ctx[l : l + attn_bs])
            for l in range(0, win_q, attn_bs)
        ], dim=0)

        for l in range(0, win_q, pred_bs):
            yield full_cache[l : l + pred_bs], l * ctx_win

    if win_r > 0:
        rest_ctx = ctx[win_q * ctx_win :].unsqueeze(0)  # [1, win_r, 4, 6]
        yield model.get_attn_cache(rest_ctx), win_q * ctx_win
