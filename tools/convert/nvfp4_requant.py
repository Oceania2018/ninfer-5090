"""Re-encode selected weights of a NInfer artifact to NVFP4 (A16-only) to save device memory.

Every object bound by a binding whose name matches one of the given regular expressions is
decoded to FP32 and re-encoded as ``nvfp4`` / ``block_scale_k16_m128x4_v1``; all other objects,
bindings and metadata are copied unchanged. Converted weights carry no activation divisor, so
every use of a converted parameter is set to ``A16Only``.

Quantization: one FP32 weight divisor per object maps the tensor's absolute maximum to the top of
the E2M1 x E4M3 range; each 16-element block then takes the E4M3 scale, out of the rounded ideal
scale and its two E4M3 neighbours, with the smallest squared reconstruction error.

usage: python -m tools.convert.nvfp4_requant SRC.ninfer DST.ninfer BINDING_REGEX [...]

example (16 GB DFlash2 package):
    python -m tools.convert.nvfp4_requant in.ninfer out.ninfer \\
        '^text/output_head$' '^dflash2/layers/\\d+/mlp/(gate|up)$'
"""

from __future__ import annotations

import copy
import re
import sys
import time

import torch

from tools.artifact.codecs.direct import decode_direct
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
from tools.artifact.codecs.row_split import dequantize_row_split, split_row_planes
from tools.artifact.layouts import row_split_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceObject, ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter

NVFP4_LAYOUT = "block_scale_k16_m128x4_v1"
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Rounding boundaries between consecutive E2M1 magnitudes.
E2M1_EDGES = (E2M1_LEVELS[1:] + E2M1_LEVELS[:-1]) / 2
ROWS_PER_CHUNK = 2048


def bound_objects(directory) -> dict[str, list[str]]:
    """Object id -> binding names that reference it."""

    def objects(binding):
        if isinstance(binding, dict):
            if isinstance(binding.get("object"), str):
                yield binding["object"]
            for value in binding.values():
                yield from objects(value)
        elif isinstance(binding, list):
            for value in binding:
                yield from objects(value)

    out: dict[str, list[str]] = {}
    for name, binding in directory.bindings.items():
        for object_id in set(objects(binding)):
            out.setdefault(object_id, []).append(name)
    return out


def decode_fp32(art: Artifact, obj) -> torch.Tensor:
    payload = art.read_object(obj.id)
    if obj.format in ("bf16", "fp32"):
        return decode_direct(payload, obj.format, tuple(obj.shape)).to(torch.float32)
    n, k = obj.shape
    geometry = row_split_geometry(obj.format, (n, k))
    out = torch.empty((n, k), dtype=torch.float32)
    # Row chunks keep decode intermediates small; the head alone is 5 GiB in FP32.
    for begin in range(0, n, ROWS_PER_CHUNK):
        rows = min(ROWS_PER_CHUNK, n - begin)
        planes = split_row_planes(payload, geometry, begin, rows)
        out[begin : begin + rows] = dequantize_row_split(
            planes, obj.format, (rows, k), dtype=torch.float32)
    return out


def e4m3_bits(values: torch.Tensor) -> torch.Tensor:
    """Round nonnegative FP32 values to E4M3FN words (saturating at 448)."""
    return values.clamp(0.0, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_values(bits: torch.Tensor) -> torch.Tensor:
    return bits.view(torch.float8_e4m3fn).to(torch.float32)


def quantize_blocks(scaled: torch.Tensor, scale: torch.Tensor):
    """E2M1 magnitude indices and signs for ``scaled / scale`` ([rows, blocks, 16])."""
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    ratio = (scaled / safe.unsqueeze(-1)).clamp(-E2M1_MAX, E2M1_MAX)
    index = torch.bucketize(ratio.abs(), E2M1_EDGES)
    index = torch.where(scale.unsqueeze(-1) > 0, index, torch.zeros_like(index))
    return index, ratio < 0


def quantize_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed E2M1 codes [N,K/2], natural E4M3 scales [N,K/16] and the FP32 divisor."""
    n, k = weight.shape
    amax = float(weight.abs().max())
    divisor = torch.tensor(E2M1_MAX * E4M3_MAX / amax if amax > 0 else 1.0, dtype=torch.float32)
    codes = torch.empty((n, k // 2), dtype=torch.uint8)
    scales = torch.empty((n, k // 16), dtype=torch.uint8)
    for begin in range(0, n, ROWS_PER_CHUNK):
        end = min(n, begin + ROWS_PER_CHUNK)
        scaled = weight[begin:end].reshape(end - begin, k // 16, 16) * divisor
        ideal = scaled.abs().amax(dim=-1) / E2M1_MAX
        center = e4m3_bits(ideal).to(torch.int16)
        best_bits = best_error = best_index = best_sign = None
        for step in (-1, 0, 1):
            # 0x7E is the largest finite E4M3FN word; 0 keeps all-zero blocks exact.
            bits = (center + step).clamp(0, 0x7E).to(torch.uint8)
            scale = e4m3_values(bits)
            index, sign = quantize_blocks(scaled, scale)
            magnitude = E2M1_LEVELS[index] * scale.unsqueeze(-1)
            error = (torch.where(sign, -magnitude, magnitude) - scaled).square().sum(dim=-1)
            if best_error is None:
                best_bits, best_error, best_index, best_sign = bits, error, index, sign
                continue
            better = error < best_error
            best_bits = torch.where(better, bits, best_bits)
            best_error = torch.where(better, error, best_error)
            best_index = torch.where(better.unsqueeze(-1), index, best_index)
            best_sign = torch.where(better.unsqueeze(-1), sign, best_sign)
        nibble = (best_index | (best_sign.to(best_index.dtype) << 3)).to(torch.uint8)
        nibble = nibble.reshape(end - begin, k)
        # Low nibble holds the even element (cvt.rn.satfinite.e2m1x2 packing).
        codes[begin:end] = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
        scales[begin:end] = best_bits
    return codes, scales, divisor


def dequantize_nvfp4(codes, scales, divisor, shape) -> torch.Tensor:
    n, k = shape
    nibble = torch.stack((codes & 0xF, codes >> 4), dim=-1).reshape(n, k).to(torch.int64)
    magnitude = E2M1_LEVELS[nibble & 0x7]
    value = torch.where((nibble & 0x8) != 0, -magnitude, magnitude)
    block_scale = e4m3_values(scales).repeat_interleave(16, dim=1)
    return value * block_scale / divisor


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    src, dst, *patterns = sys.argv[1:]
    selected = [re.compile(p) for p in patterns]
    art = Artifact.open(src)
    d = art.directory
    owners = bound_objects(d)

    targets: set[str] = set()
    for obj in d.objects:
        names = owners.get(obj.id, [])
        matched = [n for n in names if any(p.search(n) for p in selected)]
        if not matched:
            continue
        if len(matched) != len(names):
            raise SystemExit(f"{obj.id}: shared by selected {matched} and unselected "
                             f"{sorted(set(names) - set(matched))}")
        if isinstance(obj, ResourceObject) or len(obj.shape) != 2:
            raise SystemExit(f"{obj.id} ({matched[0]}): only 2-D tensors can become NVFP4")
        n, k = obj.shape
        if n % 128 or k % 64:
            raise SystemExit(f"{obj.id} ({matched[0]}): NVFP4 needs N%128=0 and K%64=0, "
                             f"got {list(obj.shape)}")
        if obj.format == "nvfp4":
            continue
        targets.add(obj.id)
    if not targets:
        raise SystemExit("no objects matched")
    converted_params = {name for oid in targets for name in owners[oid]}

    uses, relaxed = [], 0
    for use in d.uses:
        use = copy.deepcopy(use)
        if use["parameter"] in converted_params and use.get("activation_policy") != "A16Only":
            use["activation_policy"] = "A16Only"
            use.pop("auxiliaries", None)
            relaxed += 1
        uses.append(use)

    specs = []
    for obj in d.objects:
        if isinstance(obj, ResourceObject):
            specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
        elif obj.id in targets:
            specs.append(TensorSpec(obj.id, tuple(obj.shape), "nvfp4", NVFP4_LAYOUT))
        else:
            specs.append(TensorSpec(obj.id, tuple(obj.shape), obj.format, obj.layout))

    writer = ArtifactWriter(dst, specs, components=d.components, bindings=d.bindings, uses=uses,
                            metadata=d.metadata, provenance=d.provenance)
    t0 = time.time()
    old_bytes = new_bytes = 0
    with writer:
        for obj in d.objects:
            if obj.id not in targets:
                writer.write_object(obj.id, art.iter_object(obj.id))
                continue
            weight = decode_fp32(art, obj)
            codes, scales, divisor = quantize_nvfp4(weight)
            payload = encode_nvfp4(codes, scales, divisor, obj.shape)
            # Round-trip through the registered decoder to check the stored layout, then measure
            # the reconstruction error in row chunks (a full FP32 restore of the head is 5 GiB).
            r_codes, r_scales, r_divisor = decode_nvfp4_words(payload, obj.shape)
            if not (torch.equal(r_codes, codes) and torch.equal(r_scales, scales)
                    and float(r_divisor) == float(divisor)):
                raise SystemExit(f"{obj.id}: NVFP4 payload does not round-trip")
            error = norm = 0.0
            for begin in range(0, obj.shape[0], ROWS_PER_CHUNK):
                end = min(obj.shape[0], begin + ROWS_PER_CHUNK)
                part = dequantize_nvfp4(codes[begin:end], scales[begin:end], divisor,
                                        (end - begin, obj.shape[1]))
                error += float((part - weight[begin:end]).square().sum())
                norm += float(weight[begin:end].square().sum())
            rel = (error / norm) ** 0.5 if norm > 0 else 0.0
            writer.write_object(obj.id, payload)
            old_bytes += obj.bytes
            new_bytes += len(payload)
            print(f"  {obj.id} {owners[obj.id][0]} {list(obj.shape)} {obj.format} "
                  f"{obj.bytes / 2**20:.0f} MiB -> nvfp4 {len(payload) / 2**20:.0f} MiB, "
                  f"rel RMS error {rel:.4f}", flush=True)
            del weight, codes, scales, r_codes, r_scales
        writer.finish()
    print(f"{len(targets)} objects {old_bytes / 2**30:.2f} GiB -> {new_bytes / 2**30:.2f} GiB, "
          f"{relaxed} uses -> A16Only, {time.time() - t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    main()
