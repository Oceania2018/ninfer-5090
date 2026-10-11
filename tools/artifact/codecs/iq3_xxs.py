"""IQ3_XXS codec for ``iq3_xxs_planar_v1``.

Codes are produced by llama.cpp (``llama-quantize`` with an importance matrix); this module only
re-orders llama.cpp ``block_iq3_xxs`` records (binary16 ``d`` followed by 96 code bytes) into
NInfer's layout and provides the floating-point reference decode. Grid table from llama.cpp
``ggml-common.h`` (MIT).

Payload: a code plane of ``[N, K/256, 96]`` bytes, then the ``[N, K/256]`` binary16 ``d`` plane
(256-byte aligned). Inside each 96-byte block the bytes are ordered for the four lanes ``q`` of an
MMA quad, which each own the grid entries ``2q`` and ``2q+1`` of every 32-value sub-block:

* bytes ``[16q, 16q+16)``: index bytes ``(8s + 2q, 8s + 2q + 1)`` for sub-blocks ``s = 0..7``;
* bytes ``[64 + 8q, 72 + 8q)``: a little-endian uint64 whose bits ``[7s, 7s+7)`` are the stored
  7-bit sign index ``q`` of sub-block ``s`` and whose bits ``[56, 64)`` are the 4-bit scales of
  sub-blocks ``2q`` (low nibble) and ``2q+1``.

The byte count equals llama.cpp's, and the transform is a permutation of the stored bits.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..layouts import iq3_xxs_geometry

GGUF_BLOCK_BYTES = 98

IQ3_XXS_GRID = np.array(
    [
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
    ],
    dtype=np.uint32,
)


def repack_gguf_codes(qs: np.ndarray) -> np.ndarray:
    """Re-order llama.cpp ``qs`` bytes ``[..., 96]`` into the quad-lane block order."""

    qs = np.ascontiguousarray(qs, dtype=np.uint8)
    lead = qs.shape[:-1]
    index = qs[..., :64].reshape(*lead, 8, 4, 2)  # [s, q, e]
    out = np.empty_like(qs)
    out[..., :64] = np.swapaxes(index, -3, -2).reshape(*lead, 64)  # [q, s, e]
    words = np.ascontiguousarray(qs[..., 64:]).view("<u4").reshape(*lead, 8).astype(np.uint64)
    lanes = np.zeros((*lead, 4), dtype=np.uint64)
    for q in range(4):
        for s in range(8):
            field = (words[..., s] >> np.uint64(7 * q)) & np.uint64(127)
            lanes[..., q] |= field << np.uint64(7 * s)
        lanes[..., q] |= ((words[..., 2 * q] >> np.uint64(28)) & np.uint64(15)) << np.uint64(56)
        lanes[..., q] |= ((words[..., 2 * q + 1] >> np.uint64(28)) & np.uint64(15)) << np.uint64(60)
    out[..., 64:] = lanes.astype("<u8").view(np.uint8).reshape(*lead, 32)
    return out


def encode_iq3_xxs_from_gguf_blocks(blocks: np.ndarray, shape: Sequence[int]) -> bytes:
    """Encode uint8 ``[N, K/256, 98]`` llama.cpp blocks as an ``iq3_xxs_planar_v1`` payload."""

    geometry = iq3_xxs_geometry("iq3_xxs", shape)
    records = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(
        geometry.n, geometry.blocks_per_row, GGUF_BLOCK_BYTES
    )
    payload = np.zeros(geometry.payload_bytes, dtype=np.uint8)
    payload[: geometry.code_plane_bytes] = repack_gguf_codes(records[:, :, 2:]).reshape(-1)
    scale_end = geometry.scale_plane_offset + geometry.scale_plane_bytes
    payload[geometry.scale_plane_offset : scale_end] = records[:, :, :2].reshape(-1)
    return payload.tobytes()


def _decode(magnitude_index, sign7, ls, d):
    """index [n, b, s, entry], sign7 [n, b, s, q], ls [n, b, s], d [n, b] -> values [n, b*256]."""

    n, nb = d.shape
    parity = np.unpackbits(sign7.astype(np.uint8)[..., None], axis=-1).sum(-1) & 1
    sign8 = sign7.astype(np.uint32) | (parity.astype(np.uint32) << 7)
    bits = (sign8[..., None] >> np.arange(8, dtype=np.uint32)) & 1
    sign = 1.0 - 2.0 * bits.reshape(n, nb, 8, 32).astype(np.float32)
    grid_bytes = IQ3_XXS_GRID.view(np.uint8).reshape(256, 4)
    magnitude = grid_bytes[magnitude_index].reshape(n, nb, 8, 32).astype(np.float32)
    sub_scale = d.astype(np.float32)[..., None] * (0.25 + 0.5 * ls.astype(np.float32))
    return (magnitude * sign * sub_scale[..., None]).reshape(n, nb * 256)


def dequantize_iq3_xxs(payload: bytes | np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Decode an ``iq3_xxs_planar_v1`` payload to float32 ``[N, K]``."""

    geometry = iq3_xxs_geometry("iq3_xxs", shape)
    data = np.frombuffer(payload, dtype=np.uint8) if isinstance(payload, bytes) else payload
    if data.size != geometry.payload_bytes:
        raise ValueError("IQ3_XXS payload size does not match its shape")
    n, nb = geometry.n, geometry.blocks_per_row
    codes = data[: geometry.code_plane_bytes].reshape(n, nb, 96)
    scale_end = geometry.scale_plane_offset + geometry.scale_plane_bytes
    d = data[geometry.scale_plane_offset : scale_end].view("<f2").reshape(n, nb)
    index = np.swapaxes(codes[:, :, :64].reshape(n, nb, 4, 8, 2), 2, 3).reshape(n, nb, 8, 8)
    lanes = np.ascontiguousarray(codes[:, :, 64:]).view("<u8").reshape(n, nb, 4)
    shifts = np.arange(8, dtype=np.uint64) * np.uint64(7)
    sign7 = np.swapaxes((lanes[..., None] >> shifts) & np.uint64(127), 2, 3)  # [s, q]
    nibbles = (lanes >> np.uint64(56)).astype(np.uint8)
    ls = np.stack([nibbles & 15, nibbles >> 4], axis=-1).reshape(n, nb, 8)
    return _decode(index, sign7, ls, d)


def dequantize_gguf_blocks(blocks: np.ndarray, k: int) -> np.ndarray:
    """Reference llama.cpp ``dequantize_row_iq3_xxs`` on uint8 ``[N, K/256, 98]`` records."""

    nb = k // 256
    records = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1, nb, GGUF_BLOCK_BYTES)
    n = records.shape[0]
    d = np.ascontiguousarray(records[:, :, :2]).view("<f2").reshape(n, nb)
    qs = records[:, :, 2:]
    index = qs[:, :, :64].reshape(n, nb, 8, 8)
    words = np.ascontiguousarray(qs[:, :, 64:]).view("<u4").reshape(n, nb, 8)
    sign7 = (words[..., None] >> (np.arange(4, dtype=np.uint32) * 7)) & 127
    return _decode(index, sign7, words >> 28, d)


__all__ = [
    "GGUF_BLOCK_BYTES",
    "IQ3_XXS_GRID",
    "dequantize_gguf_blocks",
    "dequantize_iq3_xxs",
    "encode_iq3_xxs_from_gguf_blocks",
    "repack_gguf_codes",
]
