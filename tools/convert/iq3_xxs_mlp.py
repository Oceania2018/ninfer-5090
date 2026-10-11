"""Replace the dense MLP of a Qwen3.8-27B NInfer artifact with llama.cpp IQ3_XXS codes.

The IQ3_XXS codes come from a GGUF quantized by ``llama-quantize`` with an importance matrix,
for example::

    llama-quantize --allow-requantize --imatrix imatrix.gguf \\
        --tensor-type ffn_gate=iq3_xxs --tensor-type ffn_up=iq3_xxs --tensor-type ffn_down=iq3_xxs \\
        Qwen3.8-27B-Q8_0.gguf Qwen3.8-27B-mlp-iq3xxs.gguf Q8_0

Every ``text/layers/N/mlp/{gate,up,down}`` object of the source artifact is rewritten as
``iq3_xxs`` / ``iq3_xxs_planar_v1``; all other objects, bindings, uses and metadata are copied
unchanged. The grouped gate/up object keeps its row order (gate rows, then up rows), so the
binding ranges stay valid.

usage: python -m tools.convert.iq3_xxs_mlp SRC.ninfer MLP.gguf DST.ninfer
"""

from __future__ import annotations

import re
import struct
import sys
import time

import numpy as np

from tools.artifact.codecs.iq3_xxs import GGUF_BLOCK_BYTES, encode_iq3_xxs_from_gguf_blocks
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceObject, ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter

GGML_TYPE_IQ3_XXS = 18
_MLP_BINDING = re.compile(r"^text/layers/(\d+)/mlp/(gate|up|down)$")


class GgufTensors:
    """Minimal GGUF v3 reader: tensor directory plus a memory map of the data section."""

    _SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q",
               12: "d"}

    def __init__(self, path: str):
        with open(path, "rb") as f:
            head = f.read(64 << 20)
        self._buf, self._pos = head, 0
        if head[:4] != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        self._pos = 4
        version, n_tensors, n_kv = self._u("I"), self._u("Q"), self._u("Q")
        if version < 2:
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        alignment = 32
        for _ in range(n_kv):
            key = self._str()
            value = self._value(self._u("I"))
            if key == "general.alignment":
                alignment = int(value)
        self.tensors: dict[str, tuple[int, tuple[int, ...], int]] = {}
        for _ in range(n_tensors):
            name = self._str()
            dims = tuple(self._u("Q") for _ in range(self._u("I")))
            ggml_type, offset = self._u("I"), self._u("Q")
            self.tensors[name] = (ggml_type, dims, offset)
        data_start = (self._pos + alignment - 1) // alignment * alignment
        self._map = np.memmap(path, dtype=np.uint8, mode="r")
        self._data_start = data_start

    def _u(self, fmt: str):
        value = struct.unpack_from("<" + fmt, self._buf, self._pos)[0]
        self._pos += struct.calcsize("<" + fmt)
        return value

    def _str(self) -> str:
        n = self._u("Q")
        value = self._buf[self._pos : self._pos + n].decode("utf-8")
        self._pos += n
        return value

    def _value(self, t: int):
        if t in self._SCALAR:
            return self._u(self._SCALAR[t])
        if t == 8:
            return self._str()
        if t == 9:
            item, n = self._u("I"), self._u("Q")
            return [self._value(item) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {t}")

    def iq3_xxs_blocks(self, name: str, rows: int, k: int) -> np.ndarray:
        ggml_type, dims, offset = self.tensors[name]
        if ggml_type != GGML_TYPE_IQ3_XXS:
            raise SystemExit(f"{name}: expected IQ3_XXS (18), got GGML type {ggml_type}")
        if dims != (k, rows):
            raise SystemExit(f"{name}: expected GGUF dims ({k}, {rows}), got {dims}")
        nbytes = rows * (k // 256) * GGUF_BLOCK_BYTES
        start = self._data_start + offset
        return self._map[start : start + nbytes].reshape(rows, k // 256, GGUF_BLOCK_BYTES)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    src, gguf_path, dst = sys.argv[1:]
    art = Artifact.open(src)
    d = art.directory
    gguf = GgufTensors(gguf_path)

    # object id -> [(layer, role, first_row, rows)]
    sources: dict[str, list[tuple[int, str, int, int]]] = {}
    shapes = {obj.id: tuple(obj.shape) for obj in d.objects if not isinstance(obj, ResourceObject)}
    for name, binding in d.bindings.items():
        match = _MLP_BINDING.match(name)
        if not match:
            continue
        layer, role = int(match.group(1)), match.group(2)
        parts = binding.get("parts") or [binding]
        if len(parts) != 1:
            raise SystemExit(f"{name}: expected one object part")
        part = parts[0]
        obj_id = part["object"]
        k = shapes[obj_id][1]
        begin, end = part.get("range", [0, shapes[obj_id][0] * k])
        if begin % k or end % k:
            raise SystemExit(f"{name}: binding range is not row aligned")
        sources.setdefault(obj_id, []).append((layer, role, begin // k, (end - begin) // k))
    if not sources:
        raise SystemExit("no text/layers/*/mlp bindings found")

    specs = []
    for obj in d.objects:
        if isinstance(obj, ResourceObject):
            specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
        elif obj.id in sources:
            specs.append(TensorSpec(obj.id, tuple(obj.shape), "iq3_xxs", "iq3_xxs_planar_v1"))
        else:
            specs.append(TensorSpec(obj.id, tuple(obj.shape), obj.format, obj.layout))

    writer = ArtifactWriter(
        dst, specs, components=d.components, bindings=d.bindings, uses=d.uses,
        metadata=d.metadata, provenance=d.provenance,
    )
    t0 = time.time()
    old_bytes = new_bytes = 0
    with writer:
        for obj in d.objects:
            if obj.id not in sources:
                writer.write_object(obj.id, art.iter_object(obj.id))
                continue
            n, k = obj.shape
            blocks = np.empty((n, k // 256, GGUF_BLOCK_BYTES), dtype=np.uint8)
            covered = 0
            for layer, role, first_row, rows in sorted(sources[obj.id], key=lambda s: s[2]):
                blocks[first_row : first_row + rows] = gguf.iq3_xxs_blocks(
                    f"blk.{layer}.ffn_{role}.weight", rows, k)
                covered += rows
            if covered != n:
                raise SystemExit(f"{obj.id}: bindings cover {covered} of {n} rows")
            payload = encode_iq3_xxs_from_gguf_blocks(blocks, obj.shape)
            writer.write_object(obj.id, payload)
            old_bytes += obj.bytes
            new_bytes += len(payload)
            print(f"  {obj.id} {list(obj.shape)} {obj.format} {obj.bytes / 2**20:.0f} MiB -> "
                  f"iq3_xxs {len(payload) / 2**20:.0f} MiB", flush=True)
        writer.finish()
    print(f"MLP {old_bytes / 2**30:.2f} GiB -> {new_bytes / 2**30:.2f} GiB in "
          f"{time.time() - t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    main()
