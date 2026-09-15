#ifndef OCTRANS_H_
#define OCTRANS_H_

#include <torch/extension.h>
#include "rans_byte.h"

namespace py = pybind11;

#define DEFAULT_ENC_BUF_SIZE (32 * 1024 * 1024)
#define PRECISION (16u)
#define PROB_SCALE (1u << PRECISION)


/**
 * RansEncoder
 *
 * Encodes symbols using rANS (range Asymmetric Numeral Systems).
 * CDFs and symbols are CPU contiguous torch tensors.
 *
 * CDF format (cdf_arr shape [N, S], dtype torch.uint16):
 *   cdf_arr[i, j] = P(X <= j) scaled by PROB_SCALE.
 *   Symbol j owns [cdf_arr[i, j-1], cdf_arr[i, j]); the lower boundary of
 *   symbol 0 is implicitly 0 and the upper boundary of symbol S-1 is PROB_SCALE.
 *
 *   Example for a 5-symbol alphabet:
 *     cdf_arr[i] = [c0, c1, c2, c3, 65535]   where 0 < c0 <= c1 <= c2 <= c3 < 65535.
 *
 * A single-row CDF ([1, S]) broadcasts over all N symbols.
 *
 * Encoding is done in *reverse* order (rANS stack semantics).
 * Call encode() one or more times, then flush() to get the bitstream.
 */
class RansEncoder
{
public:
    explicit RansEncoder(size_t enc_buf_size = DEFAULT_ENC_BUF_SIZE);
    ~RansEncoder();

    // cdf_arr: (N, S) or (1, S) uint16; symbol_arr: (N,) uint16.
    uint64_t encode(const torch::Tensor &cdf_arr, const torch::Tensor &symbol_arr);

    // cdf_arr: (N,) or (N, 1) uint16 giving P(X=0); symbol_arr: (N,) bool.
    uint64_t encode_bin(const torch::Tensor &cdf_arr, const torch::Tensor &symbol_arr);

    // Flush accumulated bits to a Python bytes object. Resets the encoder state.
    py::bytes flush();

private:
    RansState rans;
    uint8_t * enc_buf_ptr;
    size_t    enc_buf_size;
    uint8_t * enc_ptr;
};


/**
 * RansDecoder
 *
 * Decodes symbols encoded by RansEncoder.
 * Call flush(encoded) to load a bitstream, then call decode() in the
 * *same* order the corresponding encode() calls were made.
 */
class RansDecoder
{
public:
    RansDecoder();

    // Load a compressed bitstream. Must be called before any decode().
    int flush(const py::bytes &encoded);

    int decode(const torch::Tensor &cdf_arr, torch::Tensor &symbol_arr);
    int decode_bin(const torch::Tensor &cdf_arr, torch::Tensor &symbol_arr);

private:
    RansState rans;
    py::bytes buf;   // holds a reference so ptr stays valid across decode() calls
    uint8_t * ptr;
};

#endif  // OCTRANS_H_
