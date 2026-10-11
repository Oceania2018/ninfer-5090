"""Copy a NInfer artifact, taking each weight use's activation policy from a reference artifact.

Objects, bindings, metadata and every other use field are copied unchanged; only
``activation_policy`` changes, for uses whose (parameter, input) pair the reference also has.
Typical use: give a re-quantized package the A8/A4 permissions validated for the production
artifact of the same model, so int8-activation prefill kernels may run.

usage: python -m tools.convert.copy_activation_policies SRC.ninfer REF.ninfer DST.ninfer
"""

from __future__ import annotations

import copy
import sys
import time
from collections import Counter

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceObject, ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    src, ref, dst = sys.argv[1:]
    art = Artifact.open(src)
    d = art.directory
    reference = {(u["parameter"], u["input"]): u.get("activation_policy")
                 for u in Artifact.open(ref).directory.uses}

    uses, changed = [], Counter()
    for use in d.uses:
        use = copy.deepcopy(use)
        policy = reference.get((use["parameter"], use["input"]))
        if policy is not None and policy != use.get("activation_policy"):
            changed[(use.get("activation_policy"), policy)] += 1
            use["activation_policy"] = policy
        uses.append(use)
    for (old, new), n in sorted(changed.items(), key=str):
        print(f"  {n:4d} uses {old} -> {new}")

    specs = [ResourceSpec(o.id, o.bytes, o.encoding) if isinstance(o, ResourceObject)
             else TensorSpec(o.id, tuple(o.shape), o.format, o.layout) for o in d.objects]
    writer = ArtifactWriter(dst, specs, components=d.components, bindings=d.bindings, uses=uses,
                            metadata=d.metadata, provenance=d.provenance)
    t0 = time.time()
    with writer:
        for obj in d.objects:
            writer.write_object(obj.id, art.iter_object(obj.id))
        writer.finish()
    print(f"copied {len(d.objects)} objects in {time.time() - t0:.0f}s -> {dst}")


if __name__ == "__main__":
    main()
