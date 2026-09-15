#pragma once

#include <cstdint>

#if defined(__BMI2__)
#include <immintrin.h>
#endif

namespace mortonkit {

constexpr uint64_t MORTON_MASK_X = 0x1249249249249249ULL;

inline uint64_t expand_bits(uint32_t v) noexcept {
#if defined(__BMI2__)
    return _pdep_u64(static_cast<uint64_t>(v), MORTON_MASK_X);
#else
    uint64_t x = v & 0x00000000001FFFFFULL;
    x = (x | (x << 32)) & 0x001F00000000FFFFULL;
    x = (x | (x << 16)) & 0x001F0000FF0000FFULL;
    x = (x | (x << 8))  & 0x100F00F00F00F00FULL;
    x = (x | (x << 4))  & 0x10C30C30C30C30C3ULL;
    x = (x | (x << 2))  & MORTON_MASK_X;
    return x;
#endif
}

inline uint32_t compact_bits(uint64_t v) noexcept {
#if defined(__BMI2__)
    return static_cast<uint32_t>(_pext_u64(v, MORTON_MASK_X));
#else
    uint64_t x = v & MORTON_MASK_X;
    x = (x | (x >> 2))  & 0x10C30C30C30C30C3ULL;
    x = (x | (x >> 4))  & 0x100F00F00F00F00FULL;
    x = (x | (x >> 8))  & 0x001F0000FF0000FFULL;
    x = (x | (x >> 16)) & 0x001F00000000FFFFULL;
    x = (x | (x >> 32)) & 0x00000000001FFFFFULL;
    return static_cast<uint32_t>(x);
#endif
}

inline uint64_t encode(uint32_t x, uint32_t y, uint32_t z) noexcept {
    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2);
}

inline void decode(uint64_t m, uint32_t& x, uint32_t& y, uint32_t& z) noexcept {
    x = compact_bits(m);
    y = compact_bits(m >> 1);
    z = compact_bits(m >> 2);
}

}  // namespace mortonkit
