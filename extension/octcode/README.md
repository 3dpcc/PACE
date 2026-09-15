# octcode

A Morton-code octree builder that turns point clouds into occupancy symbols for the context model and back into coordinates on decode.

## Installation

```bash
pip install pybind11 numpy
pip install .
```

## Quick example

```python
import numpy as np
import octcode

rng = np.random.default_rng(0)
depth = 6
voxels = rng.integers(0, 2 ** depth, size=(200_000, 3), dtype=np.int32)
pc = np.unique(voxels, axis=0)

# --- encode side: occupancy symbol per level ---
nodes = octcode.build_octree(pc, depth)               # [M, 4, 6]
levels = nodes[:, -1, 1]
_, counts = np.unique(levels, return_counts=True)

occupancies = []
off = 0
for n in counts:
    occupancies.append(nodes[off : off + n, -1, 0])   # occupancy per level
    off += n

# --- decode side: replay symbols and recover coords ---
octree = octcode.OctreeDec(depth)
for d, occ in enumerate(occupancies):
    octree.calc_context(d)
    octree.calc_children(occ, d)

recovered = octree.get_coords(depth)                  # [K, 3] int32 voxel coords
assert recovered.shape[0] == pc.shape[0]
assert set(map(tuple, recovered)) == set(map(tuple, pc))
print(f"round-trip OK: {len(recovered)} points")
```
