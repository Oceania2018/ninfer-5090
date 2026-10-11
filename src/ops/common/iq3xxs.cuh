#pragma once

// IQ3_XXS weight decoding shared by the iq3_xxs linear kernels.
//
// Physical layout (written by tools/artifact, repacked from llama.cpp block_iq3_xxs):
//   codes  [rows, blocks * 96] uint8, per 256-value block: 64 grid-index bytes, then eight
//          uint32 sub-block words (bits 0..27 = four 7-bit sign indices, bits 28..31 = scale ls)
//   scales [rows, blocks] binary16 block scale d
// Value j (0..31) of sub-block s: magnitude = byte (j & 3) of grid[idx[s * 8 + j / 4]],
// sign = bit (j & 7) of sign8(word[s] >> (7 * (j / 8))), scale = d * (2 * ls + 1) / 4.
// Magnitudes are integers <= 62, so signed magnitudes are exact in BF16 and the sub-block scale
// is applied to the FP32 partial sums. Grid table from llama.cpp ggml-common.h (MIT).

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

constexpr int kIq3XxsBlockK         = 256;
constexpr int kIq3XxsSubBlockK      = 32;
constexpr int kIq3XxsBlockCodeBytes = 96;
constexpr int kIq3XxsIndexBytes     = 64;

__device__ __constant__ inline std::uint32_t kIq3XxsGrid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

// The 7-bit stored sign index plus an even-parity eighth bit (llama.cpp ksigns_iq2xs).
__device__ __forceinline__ unsigned iq3xxs_sign8(unsigned index7) {
    return index7 | ((__popc(index7) & 1u) << 7);
}

__device__ __forceinline__ float iq3xxs_sub_scale(float d, std::uint32_t word) {
    return d * (0.25f + 0.5f * static_cast<float>(word >> 28));
}

// Copies the grid into shared memory; divergent constant-cache lookups serialize.
__device__ __forceinline__ void iq3xxs_load_grid(std::uint32_t* grid_sh) {
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += static_cast<int>(blockDim.x)) {
        grid_sh[i] = kIq3XxsGrid[i];
    }
}

// Grid entries as BF16x2 words: .x = magnitudes (0, 1), .y = magnitudes (2, 3). Integers <= 62
// convert exactly, so a BF16 pair is the upper half of each FP32 word.
__device__ __forceinline__ void iq3xxs_load_grid_bf16(uint2* table) {
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += static_cast<int>(blockDim.x)) {
        const std::uint32_t g = kIq3XxsGrid[i];
        unsigned b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            b[j] = __float_as_uint(static_cast<float>((g >> (8 * j)) & 0xffu)) >> 16;
        }
        table[i] = make_uint2(b[0] | (b[1] << 16), b[2] | (b[3] << 16));
    }
}

__device__ __forceinline__ void iq3xxs_load_grid_f32(float4* table) {
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += static_cast<int>(blockDim.x)) {
        const std::uint32_t g = kIq3XxsGrid[i];
        table[i] = make_float4(static_cast<float>(g & 0xffu), static_cast<float>((g >> 8) & 0xffu),
                               static_cast<float>((g >> 16) & 0xffu), static_cast<float>(g >> 24));
    }
}

// BF16 sign masks for the two values selected by `bits` (bit 0 -> low half, bit 1 -> high half).
__device__ __forceinline__ unsigned iq3xxs_bf16x2_sign_mask(unsigned bits) {
    return ((bits & 1u) << 15) | ((bits & 2u) << 30);
}

// Signed integer magnitudes of values j and j + 1 (j even) of one sub-block, packed as BF16x2.
__device__ __forceinline__ unsigned iq3xxs_pair_bf16(const std::uint32_t* grid_sh,
                                                     const std::uint8_t* sub_index,
                                                     std::uint32_t word, int j) {
    const std::uint32_t g = grid_sh[sub_index[j >> 2]];
    const unsigned signs  = iq3xxs_sign8((word >> (7 * (j >> 3))) & 127u);
    const int shift       = 8 * (j & 3);
    float v0              = static_cast<float>((g >> shift) & 0xffu);
    float v1              = static_cast<float>((g >> (shift + 8)) & 0xffu);
    if (signs & (1u << (j & 7))) v0 = -v0;
    if (signs & (2u << (j & 7))) v1 = -v1;
    union {
        __nv_bfloat162 value;
        unsigned bits;
    } pair;
    pair.value = __floats2bfloat162_rn(v0, v1);
    return pair.bits;
}

} // namespace ninfer::ops::detail
