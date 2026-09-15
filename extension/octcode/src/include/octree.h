#pragma once

#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


struct OctreeLevel {
    std::vector<uint64_t> morton;       // node Morton prefix (3*depth bits, lower bits = leaf-ward)
    std::vector<uint8_t>  occupancy;    // 8-bit occupancy; bit (7 - octant) set if child octant is present
    std::vector<int32_t>  parent_idx;   // index of this node's parent in level[d-1]; unused for d == 0
};


class OctreeEnc {
public:
    explicit OctreeEnc(int32_t max_depth);

    void build(const pybind11::array_t<int32_t>& pc);
    pybind11::array_t<int32_t> traverse();

private:
    int32_t max_depth_;
    std::vector<OctreeLevel> levels_;   // levels_[d] holds internal nodes at depth d (d in [0, max_depth-1])
};


class OctreeDec {
public:
    explicit OctreeDec(int32_t max_depth);

    void calc_context(int32_t depth);
    void calc_children(const pybind11::array_t<int32_t>& syms, int32_t depth);
    pybind11::array_t<int32_t> get_context(int32_t depth);
    pybind11::array_t<int32_t> get_coords(int32_t depth);

private:
    int32_t max_depth_;

    struct DecLevel : public OctreeLevel {
        std::vector<int32_t> ctx;       // flattened [N, 4, 6] context buffer
    };

    std::vector<DecLevel> levels_;      // levels_[d] for d in [0, max_depth]
};
