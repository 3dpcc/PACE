#include <algorithm>
#include <bit>
#include <cstring>
#include <cstdint>

#include "include/octree.h"
#include "include/morton.h"


namespace {

constexpr int32_t kParallelThreshold = 1 << 14;
constexpr int kInfoWidth = 6;
constexpr int kCtxWidth = 4 * kInfoWidth;

struct NodeInfo {
    int32_t occupancy, depth, octant, xc, yc, zc;
};

inline NodeInfo make_info(const OctreeLevel& lvl, int32_t i, int32_t d, int32_t max_depth) noexcept {
    const uint64_t m = lvl.morton[i];
    uint32_t mx, my, mz;
    mortonkit::decode(m, mx, my, mz);
    const int32_t  shift = max_depth - d;
    const uint32_t half  = (shift > 0) ? (1u << (shift - 1)) : 0u;
    return NodeInfo{
        static_cast<int32_t>(lvl.occupancy[i]),
        d,
        static_cast<int32_t>(m & 7u),
        static_cast<int32_t>((mx << shift) + half),
        static_cast<int32_t>((my << shift) + half),
        static_cast<int32_t>((mz << shift) + half),
    };
}

inline void write_info(int32_t* dst, const NodeInfo& info) noexcept {
    dst[0] = info.occupancy;
    dst[1] = info.depth;
    dst[2] = info.octant;
    dst[3] = info.xc;
    dst[4] = info.yc;
    dst[5] = info.zc;
}

}  // namespace


// =================== OctreeEnc ===================

OctreeEnc::OctreeEnc(int32_t max_depth) : max_depth_(max_depth) {}

void OctreeEnc::build(const pybind11::array_t<int32_t>& pc) {
    auto view = pc.unchecked<2>();
    const int32_t N = static_cast<int32_t>(pc.shape(0));

    std::vector<uint64_t> codes(static_cast<size_t>(N));
    #pragma omp parallel for schedule(static) if(N > kParallelThreshold)
    for (int32_t i = 0; i < N; i++) {
        codes[i] = mortonkit::encode(
            static_cast<uint32_t>(view(i, 0)),
            static_cast<uint32_t>(view(i, 1)),
            static_cast<uint32_t>(view(i, 2))
        );
    }

    std::sort(codes.begin(), codes.end());
    codes.erase(std::unique(codes.begin(), codes.end()), codes.end());

    levels_.clear();
    levels_.resize(std::max(max_depth_, 1));

    if (max_depth_ == 0 || codes.empty()) {
        levels_[0].morton.push_back(0);
        levels_[0].occupancy.push_back(0);
        return;
    }

    {
        auto& deepest = levels_[max_depth_ - 1];
        const int32_t M = static_cast<int32_t>(codes.size());
        deepest.morton.reserve(M);
        deepest.occupancy.reserve(M);
        int32_t start = 0;
        while (start < M) {
            const uint64_t pm = codes[start] >> 3;
            uint8_t occ = 0;
            int32_t end = start;
            while (end < M && (codes[end] >> 3) == pm) {
                occ |= static_cast<uint8_t>(1u << (7u - (codes[end] & 7u)));
                ++end;
            }
            deepest.morton.push_back(pm);
            deepest.occupancy.push_back(occ);
            start = end;
        }
    }

    for (int d = max_depth_ - 2; d >= 0; --d) {
        auto& child = levels_[d + 1];
        auto& cur   = levels_[d];
        const int32_t M = static_cast<int32_t>(child.morton.size());
        child.parent_idx.assign(M, 0);
        cur.morton.reserve(M);
        cur.occupancy.reserve(M);
        int32_t start = 0;
        while (start < M) {
            const uint64_t pm = child.morton[start] >> 3;
            uint8_t occ = 0;
            const int32_t pidx = static_cast<int32_t>(cur.morton.size());
            int32_t end = start;
            while (end < M && (child.morton[end] >> 3) == pm) {
                occ |= static_cast<uint8_t>(1u << (7u - (child.morton[end] & 7u)));
                child.parent_idx[end] = pidx;
                ++end;
            }
            cur.morton.push_back(pm);
            cur.occupancy.push_back(occ);
            start = end;
        }
    }
}

pybind11::array_t<int32_t> OctreeEnc::traverse() {
    namespace py = pybind11;

    const int D = max_depth_;
    std::vector<size_t> level_offset(static_cast<size_t>(D) + 1, 0);
    for (int d = 0; d < D; ++d) {
        level_offset[d + 1] = level_offset[d] + levels_[d].morton.size();
    }
    const size_t total = level_offset[D];

    py::array_t<int32_t> out({total, size_t(4), size_t(6)});
    if (total == 0) return out;
    int32_t* data = out.mutable_data();

    {
        const NodeInfo info = make_info(levels_[0], 0, 0, D);
        int32_t* row = data;
        write_info(row + 0 * kInfoWidth, info);
        write_info(row + 1 * kInfoWidth, info);
        write_info(row + 2 * kInfoWidth, info);
        write_info(row + 3 * kInfoWidth, info);
    }

    for (int d = 1; d < D; ++d) {
        auto& lvl = levels_[d];
        if (lvl.morton.empty()) continue;
        const int32_t N = static_cast<int32_t>(lvl.morton.size());
        const size_t base   = level_offset[d];
        const size_t p_base = level_offset[d - 1];
        #pragma omp parallel for schedule(static) if(N > kParallelThreshold)
        for (int32_t i = 0; i < N; ++i) {
            const int32_t pi = lvl.parent_idx[i];
            const int32_t* p_row = data + (p_base + pi) * kCtxWidth;
            int32_t* row = data + (base + i) * kCtxWidth;

            const NodeInfo self = make_info(lvl, i, d, D);

            if (d == 1) {
                std::memcpy(row + 0 * kInfoWidth, p_row + 3 * kInfoWidth, kInfoWidth * sizeof(int32_t));
                std::memcpy(row + 1 * kInfoWidth, p_row + 3 * kInfoWidth, kInfoWidth * sizeof(int32_t));
                std::memcpy(row + 2 * kInfoWidth, p_row + 3 * kInfoWidth, kInfoWidth * sizeof(int32_t));
            } else if (d == 2) {
                std::memcpy(row + 0 * kInfoWidth, p_row + 2 * kInfoWidth, kInfoWidth * sizeof(int32_t));
                std::memcpy(row + 1 * kInfoWidth, p_row + 2 * kInfoWidth, kInfoWidth * sizeof(int32_t));
                std::memcpy(row + 2 * kInfoWidth, p_row + 3 * kInfoWidth, kInfoWidth * sizeof(int32_t));
            } else {
                std::memcpy(row + 0 * kInfoWidth, p_row + 1 * kInfoWidth, kInfoWidth * sizeof(int32_t));
                std::memcpy(row + 1 * kInfoWidth, p_row + 2 * kInfoWidth, kInfoWidth * sizeof(int32_t));
                std::memcpy(row + 2 * kInfoWidth, p_row + 3 * kInfoWidth, kInfoWidth * sizeof(int32_t));
            }
            write_info(row + 3 * kInfoWidth, self);
        }
    }

    return out;
}


// =================== OctreeDec ===================

OctreeDec::OctreeDec(int32_t max_depth) : max_depth_(max_depth), levels_(static_cast<size_t>(max_depth) + 1) {
    auto& root = levels_[0];
    root.morton.push_back(0);
    root.occupancy.push_back(255);
    root.parent_idx.push_back(0);
}

void OctreeDec::calc_context(int32_t depth) {
    auto& lvl = levels_[depth];
    const int32_t N = static_cast<int32_t>(lvl.morton.size());
    lvl.ctx.assign(static_cast<size_t>(N) * kCtxWidth, 0);
    if (N == 0) return;

    if (depth == 0) {
        #pragma omp parallel for schedule(static) if(N > kParallelThreshold)
        for (int32_t i = 0; i < N; ++i) {
            const NodeInfo info = make_info(lvl, i, depth, max_depth_);
            int32_t* base = lvl.ctx.data() + i * kCtxWidth;
            write_info(base + 0 * kInfoWidth, info);
            write_info(base + 1 * kInfoWidth, info);
            write_info(base + 2 * kInfoWidth, info);
            write_info(base + 3 * kInfoWidth, info);
        }
        return;
    }

    auto& p_lvl = levels_[depth - 1];
    #pragma omp parallel for schedule(static) if(N > kParallelThreshold)
    for (int32_t i = 0; i < N; ++i) {
        const int32_t pi = lvl.parent_idx[i];
        const int32_t* p_ctx = p_lvl.ctx.data() + pi * kCtxWidth;
        int32_t* base = lvl.ctx.data() + i * kCtxWidth;

        const NodeInfo self = make_info(lvl, i, depth, max_depth_);

        if (depth == 1) {
            std::memcpy(base + 0 * kInfoWidth, p_ctx + 3 * kInfoWidth, kInfoWidth * sizeof(int32_t));
            std::memcpy(base + 1 * kInfoWidth, p_ctx + 3 * kInfoWidth, kInfoWidth * sizeof(int32_t));
            std::memcpy(base + 2 * kInfoWidth, p_ctx + 3 * kInfoWidth, kInfoWidth * sizeof(int32_t));
        } else if (depth == 2) {
            std::memcpy(base + 0 * kInfoWidth, p_ctx + 2 * kInfoWidth, kInfoWidth * sizeof(int32_t));
            std::memcpy(base + 1 * kInfoWidth, p_ctx + 2 * kInfoWidth, kInfoWidth * sizeof(int32_t));
            std::memcpy(base + 2 * kInfoWidth, p_ctx + 3 * kInfoWidth, kInfoWidth * sizeof(int32_t));
        } else {
            std::memcpy(base + 0 * kInfoWidth, p_ctx + 1 * kInfoWidth, kInfoWidth * sizeof(int32_t));
            std::memcpy(base + 1 * kInfoWidth, p_ctx + 2 * kInfoWidth, kInfoWidth * sizeof(int32_t));
            std::memcpy(base + 2 * kInfoWidth, p_ctx + 3 * kInfoWidth, kInfoWidth * sizeof(int32_t));
        }
        write_info(base + 3 * kInfoWidth, self);
    }
}

pybind11::array_t<int32_t> OctreeDec::get_context(int32_t depth) {
    namespace py = pybind11;
    auto& lvl = levels_[depth];
    const int32_t N = static_cast<int32_t>(lvl.morton.size());
    py::array_t<int32_t> out({static_cast<size_t>(N), size_t(4), size_t(6)});
    if (N > 0) {
        std::memcpy(out.mutable_data(), lvl.ctx.data(), static_cast<size_t>(N) * kCtxWidth * sizeof(int32_t));
    }
    return out;
}

void OctreeDec::calc_children(const pybind11::array_t<int32_t>& syms, int32_t depth) {
    auto& lvl   = levels_[depth];
    auto& child = levels_[depth + 1];
    auto sv = syms.unchecked<1>();
    const int32_t N = static_cast<int32_t>(lvl.morton.size());

    #pragma omp parallel for schedule(static) if(N > kParallelThreshold)
    for (int32_t i = 0; i < N; ++i) {
        const uint8_t s = static_cast<uint8_t>(sv(i));
        lvl.occupancy[i] = s;
        lvl.ctx[i * kCtxWidth + 3 * kInfoWidth] = static_cast<int32_t>(s);
    }

    std::vector<int32_t> offsets(static_cast<size_t>(N) + 1, 0);
    for (int32_t i = 0; i < N; ++i) {
        offsets[i + 1] = offsets[i] + std::popcount(static_cast<uint32_t>(lvl.occupancy[i]));
    }
    const int32_t total = offsets[N];

    child.morton.assign(total, 0);
    child.occupancy.assign(total, 255);
    child.parent_idx.assign(total, 0);

    #pragma omp parallel for schedule(static) if(N > kParallelThreshold)
    for (int32_t i = 0; i < N; ++i) {
        uint8_t occ = lvl.occupancy[i];
        const uint64_t pm_shifted = lvl.morton[i] << 3;
        int32_t off = offsets[i];
        for (int j = 0; j < 8; ++j) {
            if (occ & (1u << (7u - j))) {
                child.morton[off]     = pm_shifted | static_cast<uint64_t>(j);
                child.parent_idx[off] = i;
                ++off;
            }
        }
    }
}

pybind11::array_t<int32_t> OctreeDec::get_coords(int32_t depth) {
    namespace py = pybind11;
    auto& lvl = levels_[depth];
    const int32_t N = static_cast<int32_t>(lvl.morton.size());
    py::array_t<int32_t> out({static_cast<size_t>(N), size_t(3)});
    if (N == 0) return out;
    auto v = out.mutable_unchecked<2>();
    const int32_t shift = max_depth_ - depth;
    #pragma omp parallel for schedule(static) if(N > kParallelThreshold)
    for (int32_t i = 0; i < N; ++i) {
        uint32_t x, y, z;
        mortonkit::decode(lvl.morton[i], x, y, z);
        v(i, 0) = static_cast<int32_t>(x << shift);
        v(i, 1) = static_cast<int32_t>(y << shift);
        v(i, 2) = static_cast<int32_t>(z << shift);
    }
    return out;
}
