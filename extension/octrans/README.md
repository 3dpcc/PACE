# OctRans

Fast **rANS** (range Asymmetric Numeral Systems) coder for octree point cloud compression, packaged as a PyTorch C++ extension.

## Installation

```bash
pip install .
```

## Encode / decode order

rANS is a **stack**: encode in *reverse* order, decode in *forward* order.

```python
enc = octrans.RansEncoder()

# two levels of octree — encode innermost (last) first
enc.encode(cdf_level1, sym_level1)   # level 1 first (encoded last = outermost)
enc.encode(cdf_level0, sym_level0)   # level 0 last  (encoded first = innermost)
bitstream = enc.flush()

dec = octrans.RansDecoder()
dec.flush(bitstream)
dec.decode(cdf_level0, sym_level0_out)   # decode forward
dec.decode(cdf_level1, sym_level1_out)
```

## Quick example

```python
import octrans
import torch

n, s = 10_000, 255

# --- build a CDF from random logits ---
pmfs = torch.randn(n, s).double().softmax(-1)
cdf = torch.floor(pmfs * (65536 - s) + 1.0).cumsum(-1)
cdf[:, -1] = 65535
cdf = cdf.to(torch.uint16).contiguous()

symbols = torch.randint(0, s, (n,), dtype=torch.uint16)

# --- encode ---
enc = octrans.RansEncoder()
enc.encode(cdf, symbols)
bitstream = enc.flush()
print(f"compressed size: {len(bitstream)} bytes")

# --- decode ---
dec = octrans.RansDecoder()
dec.flush(bitstream)
decoded = torch.empty((n,), dtype=torch.uint16)
dec.decode(cdf, decoded)

assert torch.equal(symbols, decoded), "round-trip mismatch!"
print("round-trip OK")
```

