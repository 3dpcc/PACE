#include <torch/extension.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include "octrans.h"

namespace py = pybind11;

static uint16_t * _u16_ptr(const torch::Tensor &t)
{
    if (!t.device().is_cpu() || t.scalar_type() != torch::kUInt16 || !t.is_contiguous())
        throw py::type_error("expected CPU contiguous torch.uint16 tensor");
    return t.data_ptr<uint16_t>();
}

static bool * _bool_ptr(const torch::Tensor &t)
{
    if (!t.device().is_cpu() || t.scalar_type() != torch::kBool || !t.is_contiguous())
        throw py::type_error("expected CPU contiguous torch.bool tensor");
    return t.data_ptr<bool>();
}


// ============================================================
//  RansEncoder
// ============================================================

RansEncoder::RansEncoder(size_t enc_buf_size)
    : enc_buf_size(enc_buf_size)
{
    assert(enc_buf_size > 0);
    enc_buf_ptr = new uint8_t[enc_buf_size];
    enc_ptr = enc_buf_ptr + enc_buf_size;
    RansEncInit(&rans);
}

RansEncoder::~RansEncoder()
{
    delete[] enc_buf_ptr;
}

uint64_t RansEncoder::encode(
    const torch::Tensor &cdf_arr, const torch::Tensor &symbol_arr)
{
    const uint16_t * const cdf_ptr = _u16_ptr(cdf_arr);
    const uint16_t * const sym_ptr = _u16_ptr(symbol_arr);
    const size_t ncdf = (size_t)cdf_arr.size(0);
    const size_t nsym = (size_t)cdf_arr.size(1);
    const size_t n    = (size_t)symbol_arr.size(0);
    assert(nsym <= PROB_SCALE);
    assert(n == ncdf || ncdf == 1);

    const size_t stride = (ncdf == 1) ? 0 : nsym;   // 0 broadcasts one row over all n

    for (size_t i = n; i-- > 0; )
    {
        const uint16_t * const cdf = cdf_ptr + i * stride;
        const size_t sym = sym_ptr[i];
        const uint32_t curr = (sym == 0)        ? 0          : (uint32_t)cdf[sym - 1];
        const uint32_t next = (sym == nsym - 1) ? PROB_SCALE : (uint32_t)cdf[sym];
        RansEncPut(&rans, &enc_ptr, curr, next - curr, PRECISION);
    }

    return (uint64_t)(enc_buf_ptr + enc_buf_size - enc_ptr);
}

uint64_t RansEncoder::encode_bin(
    const torch::Tensor &cdf_arr, const torch::Tensor &symbol_arr)
{
    const uint16_t * const cdf_ptr = _u16_ptr(cdf_arr);
    const bool * const sym_ptr = _bool_ptr(symbol_arr);
    const size_t ncdf = (size_t)cdf_arr.size(0);
    const size_t n    = (size_t)symbol_arr.size(0);
    assert(cdf_arr.dim() == 1 || cdf_arr.size(1) == 1);
    assert(n == ncdf || ncdf == 1);

    const size_t stride = (ncdf == 1) ? 0 : 1;

    for (size_t i = n; i-- > 0; )
    {
        const uint32_t p0 = (uint32_t)cdf_ptr[i * stride];
        const uint32_t curr = sym_ptr[i] ? p0         : 0u;
        const uint32_t next = sym_ptr[i] ? PROB_SCALE : p0;
        RansEncPut(&rans, &enc_ptr, curr, next - curr, PRECISION);
    }

    return (uint64_t)(enc_buf_ptr + enc_buf_size - enc_ptr);
}

py::bytes RansEncoder::flush()
{
    RansEncFlush(&rans, &enc_ptr);
    assert(enc_buf_ptr <= enc_ptr);
    py::bytes encoded(
        reinterpret_cast<const char *>(enc_ptr),
        (py::size_t)(enc_buf_ptr + enc_buf_size - enc_ptr));
    enc_ptr = enc_buf_ptr + enc_buf_size;
    RansEncInit(&rans);
    return encoded;
}


// ============================================================
//  RansDecoder
// ============================================================

RansDecoder::RansDecoder()
    : ptr(nullptr)
{}

int RansDecoder::flush(const py::bytes &encoded)
{
    buf = encoded;
    ptr = reinterpret_cast<uint8_t *>(PyBytes_AS_STRING(buf.ptr()));
    RansDecInit(&rans, &ptr);
    return 0;
}

int RansDecoder::decode(const torch::Tensor &cdf_arr, torch::Tensor &symbol_arr)
{
    const uint16_t * const cdf_ptr = _u16_ptr(cdf_arr);
    uint16_t * const sym_ptr = _u16_ptr(symbol_arr);
    const size_t ncdf = (size_t)cdf_arr.size(0);
    const size_t nsym = (size_t)cdf_arr.size(1);
    const size_t n    = (size_t)symbol_arr.size(0);
    assert(nsym <= PROB_SCALE);
    assert(n == ncdf || ncdf == 1);

    const size_t stride = (ncdf == 1) ? 0 : nsym;

    for (size_t i = 0; i < n; ++i)
    {
        const uint16_t * const cdf = cdf_ptr + i * stride;
        const uint32_t cf = RansDecGet(&rans, PRECISION);
        size_t sym = (size_t)(std::upper_bound(cdf, cdf + nsym, (uint16_t)cf) - cdf);
        if (sym > nsym - 1) sym = nsym - 1;

        const uint32_t curr = (sym == 0)        ? 0          : (uint32_t)cdf[sym - 1];
        const uint32_t next = (sym == nsym - 1) ? PROB_SCALE : (uint32_t)cdf[sym];
        RansDecAdvance(&rans, &ptr, curr, next - curr, PRECISION);
        sym_ptr[i] = (uint16_t)sym;
    }
    return 0;
}

int RansDecoder::decode_bin(const torch::Tensor &cdf_arr, torch::Tensor &symbol_arr)
{
    const uint16_t * const cdf_ptr = _u16_ptr(cdf_arr);
    bool * const sym_ptr = _bool_ptr(symbol_arr);
    const size_t ncdf = (size_t)cdf_arr.size(0);
    const size_t n    = (size_t)symbol_arr.size(0);
    assert(cdf_arr.dim() == 1 || cdf_arr.size(1) == 1);
    assert(n == ncdf || ncdf == 1);

    const size_t stride = (ncdf == 1) ? 0 : 1;

    for (size_t i = 0; i < n; ++i)
    {
        const uint32_t p0 = (uint32_t)cdf_ptr[i * stride];
        const uint32_t cf = RansDecGet(&rans, PRECISION);
        const bool sym = cf >= p0;
        const uint32_t curr = sym ? p0         : 0u;
        const uint32_t next = sym ? PROB_SCALE : p0;
        RansDecAdvance(&rans, &ptr, curr, next - curr, PRECISION);
        sym_ptr[i] = sym;
    }
    return 0;
}


// ============================================================
//  Python bindings
// ============================================================

PYBIND11_MODULE(octrans, m)
{
    m.doc() = R"pbdoc(
        octrans — fast rANS coder for octree point cloud compression.

        Typical workflow
        ----------------
        Encoding (must encode in *reverse* order due to rANS stack semantics):

            enc = octrans.RansEncoder()          # default 32 MB buffer
            enc.encode(cdf_last,  sym_last)      # encode last block first
            enc.encode(cdf_first, sym_first)     # encode first block last
            bitstream = enc.flush()              # returns Python bytes

        Decoding (decode in *forward* order, matching encode order):

            dec = octrans.RansDecoder()
            dec.flush(bitstream)                 # load the bitstream
            dec.decode(cdf_first, sym_first)     # decode first block first
            dec.decode(cdf_last,  sym_last)      # decode last block last

        Tensor formats
        --------------
        cdf_arr : torch.Tensor, dtype=torch.uint16, device=cpu, shape (N, S)
            Quantized cumulative distribution function.
            S is the alphabet size. Each row satisfies:
              0 < cdf_arr[i,0] <= ... <= cdf_arr[i,S-2] < 65535
            and cdf_arr[i,S-1] = 65535, with an implicit 0 before the first
            entry. A single-row tensor (shape [1, S]) broadcasts over all N
            symbols.

        symbol_arr : torch.Tensor, dtype=torch.uint16, device=cpu, shape (N,)
            Symbol indices in [0, S).

        Tensors must be contiguous.
    )pbdoc";

    py::class_<RansEncoder>(m, "RansEncoder", "rANS encoder.  Buffer size can be set at construction time.")
        .def(py::init<size_t>(),
             py::arg("enc_buf_size") = (size_t)DEFAULT_ENC_BUF_SIZE,
             "Allocate an encoder with *enc_buf_size* bytes of internal buffer.")
        .def("encode", &RansEncoder::encode,
             py::arg("cdf_arr"), py::arg("symbol_arr"),
             "Encode symbols from a CDF tensor (shape [N,S] or [1,S]).")
        .def("encode_bin", &RansEncoder::encode_bin,
             py::arg("cdf_arr"), py::arg("symbol_arr"),
             "Encode binary (bool) symbols.  cdf_arr gives P(X=0) per symbol.")
        .def("flush", &RansEncoder::flush,
             "Flush the encoder state to bytes.  Resets the encoder for reuse.");

    py::class_<RansDecoder>(m, "RansDecoder",
        "rANS decoder.  Call flush(bitstream) before any decode().")
        .def(py::init<>())
        .def("flush", &RansDecoder::flush,
             py::arg("encoded"),
             "Load a compressed bitstream produced by RansEncoder.flush().")
        .def("decode", &RansDecoder::decode,
             py::arg("cdf_arr"), py::arg("symbol_arr"),
             "Decode into *symbol_arr* from a CDF tensor (shape [N,S] or [1,S]).")
        .def("decode_bin", &RansDecoder::decode_bin,
             py::arg("cdf_arr"), py::arg("symbol_arr"),
             "Decode binary (bool) symbols into *symbol_arr*.");
}
