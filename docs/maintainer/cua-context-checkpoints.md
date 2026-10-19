# Context checkpoints for CUA workloads

This page covers the two server options for computer-use agent (CUA) traffic:

- `--ctx-checkpoint-boundaries turns` decides where an agent-loop request saves checkpoints when the
  client marks none ([§3](#boundaries)).
- `--ctx-checkpoints N` decides which saved checkpoints stay. It covers why CUA traffic lost its
  prefix cache under the general retention rules of
  [Resource scheduling and context cache](resource-scheduling-and-context-cache.md#retention), what
  the chain rules change, and how they were evaluated.

## 1. The CUA request pattern

A CUA loop sends one request per step:

```text
[tools + system] [task] [assistant/tool history ...] [user: current screenshot]
```

The next step drops the screenshot turn, appends this step's assistant and tool turns, and adds a
new screenshot. The prompt is append-only except for its last turn. Checkpoints belong at the end
of the system message, the end of the task, and the end of the last history turn before the
screenshot; the next step resumes from that last one. Either the client (or a proxy in front of
NInfer) marks them with `prompt_cache_breakpoint`, or the server places them
([§3](#boundaries)). Qwen3.5/3.8 are hybrid (GDN + attention), so a request can only resume from a
position whose state was saved; there is no partial reuse between markers.

A graph walk runs many such loops back to back, one per node, sharing the system prompt. Each step
of the CUA agent is also accompanied by one-off requests on the same model (grounding and
verification calls on image crops), which publish their own points and are never resumed.

## 2. Failure under the general rules

On a clean engine every step resumes from the previous step's history marker and prefills only the
new screenshot turn (about 1.3–1.4k tokens at 1344×768). Once the Host pool fills, which takes one
walk of about 90 steps with a 20 GiB pool, it stops working:

- A point adopted once is promoted to the reuse tier and is never demoted when a deeper point of
  the same chain supersedes it. Dead history points accumulate as protected entries.
- The new chain head has not been adopted yet. Its recovery loss relative to the surviving point
  behind it is one step, which makes it the cheapest victim for the one-off requests between steps.
- The next history capture has no repeat demand of its own, so its admission budget is one step of
  added coverage, which the full pool rarely satisfies. The capture is skipped silently.
- Each step also leaves a dead private endpoint (E) and input point after the screenshot, and an
  automatic shared marker at the end of the screenshot turn.

The visible symptom is that cached tokens stop advancing inside a node. Every later step falls back
to the task marker, and prefill grows by one step's history per step.

<a id="boundaries"></a>
## 3. Checkpoint boundaries

Without markers, an OpenAI request saves one automatic point at the end of its final turn. In a CUA
loop that is the screenshot turn the next step replaces, so the next step can only resume from the
shared system prefix, if at all.

`--ctx-checkpoint-boundaries turns` is the default; `off` keeps the single automatic point. It
applies to Chat Completions and Responses requests that mark no boundary anywhere (tools or
messages):

| Point | Position |
|---|---|
| System | End of the first message |
| Task | End of the first user turn |
| History | End of the last non-empty turn after the first user turn and before the final turn: after its tool calls for an assistant turn that has them, otherwise after its last content part |

- The points are explicit boundaries, so they count against the limit of four per request and are
  published as shared prefixes exactly as client markers would be.
- When a history point is placed, the automatic end-of-prompt point is dropped. It covers the turn
  the next step replaces, and the request's private endpoint already covers an exact retry.
- A first step (no history yet) and a single-message request keep the automatic point.
- A request with any client marker is left as the client marked it.

This is the rule the CUA proxy applied before (§6.1); moving it into the server makes it available
to every client and keeps placement and retention in one place. A proxy that still marks
boundaries is unaffected: its markers take precedence.

Implementation: `place_turn_boundaries` in
[`openai_common.cpp`](../../src/serve/openai_common.cpp); tests
`test_ctx_checkpoint_turn_boundaries` in
[`test_openai_schema.cpp`](../../tests/test_openai_schema.cpp).

## 4. Chain rules

`--ctx-checkpoints N` (default `4`, `0` restores the general rules) maintains request chains over
public entries in `ResourceManager`:

| Rule | Effect |
|---|---|
| Chain membership | A public point published deeper than the public point P its request resumed from joins P's chain and records P as parent. |
| Roots | Points published by a request that did not resume from public history (system prompt, task) are roots. Chains never supersede or retire them. Resuming from a root starts a new chain, so concurrent runs sharing a system prompt stay separate. |
| Head protection | The new chain head inherits P's demand. One-off requests cannot evict it as the cheapest victim. |
| Supersession | When the head is published and P is a non-root point adopted exactly once, P is demoted to ordinary and counted as superseded. Each chain keeps the newest N superseded points for retries and rewinds; older ones are retired outright. A point adopted by a second request is a branch and is kept under the general rules. |
| Dead points | Unadopted points of the same chain that are deeper than where this request resumed, published by an earlier step, are retired (for example the previous screenshot-end marker). Shallower ones are kept: a session recap ahead of the task can still serve the next task. |
| Private endpoints | When a request resumes from P, the private points of P's publisher beyond the divergence (E, input point) are retired. |
| Admission | A non-input capture that extends a non-root adopted point inherits that point's demand instead of the one-step coverage budget. The input capture at the end of the prompt, the turn the next step replaces, does not. |

Implementation: `note_lineage`, `supersede`, `retire_dead_siblings` and `retire_shared` in
[`resource_manager.h`](../../src/runtime/engine/context_cache/resource_manager.h); unit tests
`test_lineage_*` in
[`test_resource_manager.cpp`](../../tests/runtime/test_resource_manager.cpp).

<a id="lrfu"></a>
### 4.1 LRFU ranking

`--ctx-lrfu-lambda F` (default `0`, LRU) ranks retained points by LRFU (Lee et al., 2001) instead
of their last reference alone, and changes what counts as a reference:

| Rule | Effect |
|---|---|
| Score | An entry's value is `CRF(t) = sum 2^(-F (t - t_i))` over its references `t_i`, with `t` counted in cache events (about ten per CUA step, side requests included). Ranking by CRF at a common time equals ranking by `t_last + log2(CRF(t_last)) / F`, which is stored as the entry's demand, so every existing demand comparison (victim order, reuse-tier balance, capture admission) ranks by LRFU unchanged. A single reference scores exactly as under LRU. |
| Ancestor references | A resume of N tokens also references every shallower public point on its path that is not superseded (system prompt, task, recap). Under LRU those age out while each run resumes from its own deeper history, and the next run then prefills its whole prefix cold (§7). |
| Bound | The frequency bonus is at most about `log2(1 / (F ln 2)) / F` events. With `F = 0.01` a prefix referenced on every step keeps roughly 40 steps of grace after its last run; one referenced every event, at most about 700. |

Superseded chain points stay demoted, and the reuse tier's 3/4 cap still demotes the lowest
scores, so frequency cannot pin more than that share of a pool.

Implementation: `referenced` and `reference_ancestors` in
[`resource_manager.h`](../../src/runtime/engine/context_cache/resource_manager.h); unit tests
`test_lrfu_*` in
[`test_resource_manager.cpp`](../../tests/runtime/test_resource_manager.cpp). The whole suite also
passes with LRFU forced on at `F` = 0.01, 0.05 and 0.2.

## 5. Observability

| Signal | Meaning |
|---|---|
| `ninfer_captures_skipped_total` | Optional captures dropped for lack of admissible space; previously invisible |
| `ninfer_ctx_checkpoint_retired_public_total` | Superseded or dead public chain points retired |
| `ninfer_ctx_checkpoint_retired_private_total` | Private endpoints and input points retired after the chain diverged |
| `ninfer_ctx_checkpoint_boosted_capture_admissions_total` | Capture admissions that inherited the extended point's demand |
| `ninfer_ctx_lrfu_ancestor_references_total` | LRFU references credited to shallower public points of a resumed prefix (§4.1) |
| `NINFER_CACHE_DIAG=1` | Prints one stderr line per capture decision (`ok-direct`, `ok-host-victims`, `SKIP`) with the shortage, free Host bytes and admission demand |

In CUA traffic, most skipped captures come from one-off grounding requests; that is expected.
The useful check is the per-request `cache N` in the request log: inside a node it should advance
every step.

## 6. Evaluation (CUA)

### 6.1 Setup

| Item | Value |
|---|---|
| GPU | RTX 5090, 32 GB |
| Model | Qwen3.8-27B `nvfp4` |
| Flags | `--max-context 32768 --kv-capacity auto --max-concurrency 4 --kv-dtype fp8 --spec dflash2 --draft-tokens 7 --vision --no-thinking --host-context-mib 20480` |
| Proxy | Screenshots downscaled to about 1 MPx; three explicit breakpoints (end of system, end of task, last history turn before the screenshot) |
| Baseline | `81c8ce09` (parent of this change) |

### 6.2 Replay A/B

A recorded production CUA graph walk (88 planner steps across 20 nodes, real screenshots) was
replayed through the proxy:
- Each planner request generated its recorded completion length, so endpoints were frozen as in
  production.
- Each step was followed by two grounding-style image requests on the same model.
- Every configuration started from a freshly created container and ran the walk twice in a row, so
  the second walk runs on a pool that is already full.

| Build | Walk | Prefill tokens | Prefill time | Within-node prefill / step (avg) |
|---|---|---|---|---|
| Baseline | 1 | 286,146 | 43.1 s | 1,727 |
| Baseline | 2 | 301,715 | 46.3 s | 1,953 |
| Chain bounding without head protection (ablation) | 1 | 283,094 | — | 1,683 |
| Chain bounding without head protection (ablation) | 2 | 299,716 | — | 1,924 |
| This change (`a3759c4`) | 1 | 273,679 | 42.0 s | 1,546 |
| This change (`a3759c4`) | 2 | 271,444 | 40.3 s | 1,514 |

- The ideal within-node step on a clean pool, which prefills only the screenshot turn, is about
  1,350 tokens.
- The baseline degrades on the second walk, when the pool is already full; this change does not.
- Without head protection, chain bounding alone changed almost nothing. Capture traces showed the
  one-off grounding requests evicting each freshly captured head between steps; protecting the head
  is what makes the change effective.
- A rerun of the same logic with the diagnostic trace enabled reproduced the result (274,117 and
  273,293 tokens; 1,553 and 1,541 per step).
- Node first steps (58,646 tokens per walk) and cold misses (108,345) are identical in all builds.
  They are caused by the client (see §7), not by retention.

### 6.3 Production comparison

Two serving nodes took the same CUA traffic through the same proxy during overlapping windows:
- one ran `5bce700` (this change plus the recap rule);
- the other ran an unpatched build.

Their most frequent cached-prefix length was identical (12,705 tokens), which confirms they served
the same task mix. The CUA rows count planner requests with at least 10k prompt tokens.

| Metric | Unpatched (50 min, 1,460 requests) | This change (73 min, 1,184 requests) | This change after warm-up (1,050 requests) |
|---|---|---|---|
| Hit rate (cached / prompt tokens) | 81.8% | 84.5% | 84.9% |
| Requests with no hit | 1.2% | 2.7% | 2.1% |
| Recomputed tokens per request, p50 / p90 | 2.8k / 4.6k | 1.54k / 4.46k | 1.54k / 4.46k |
| Server TTFT p50 / p90 | 588 / 1,100 ms | 357 / 655 ms | 354 / 641 ms |

During the window the patched node held the Host pool steady at about 8.8 GB of Host KV and 63
StateImages, with no preemptions and no engine errors.

These comparisons were not strictly controlled:
- The windows only partly overlap.
- The proxy routes walks to nodes by session affinity, so each node served different walks.
- The patched node started with an empty cache, which explains most of its no-hit requests in the
  first ten minutes.

## 7. Limits and open items

- **Node first steps and cold misses.** They need client changes, not retention changes:
  - **Tools are rendered before the system text,** so a node whose tool schema differs (for example
    a node-specific output schema on the finishing tool) invalidates the system marker. The node
    then prefills the whole prefix cold (about 17k tokens).
  - **Shared context ahead of the task** (a session recap repeated by every node) is only reusable
    if it ends at its own marker. That means a separate content part with a breakpoint. The dead-point
    rule above already keeps such a shallower point alive for the next node.
- **Requests with no hit after warm-up** were 2.1% on the patched node and 1.2% on the unpatched one.
  On the patched node, some system prefixes of rarely used agents were evicted after a few idle
  minutes and computed cold again. Whether protected chain heads of finished walks crowd these roots
  out has not been established. If they do, a bound on the number of protected heads (for example
  twice the maximum concurrency) is the candidate fix. LRFU ranking with ancestor references
  (§4.1) addresses the same symptom from the other side: every step of every run now refreshes
  the system prefix it passes through.
- **The replay A/B measured `a3759c4`.** `5bce700` only adds the recap rule and is covered by
  `test_lineage_keeps_a_shallower_recap_point_for_the_next_task`.

## 8. Reproduction

- **Unit tests:** `test_resource_manager.cpp` depends only on headers, so it can be built without
  CUDA:

  ```bash
  g++ -std=c++20 -O1 -Iinclude -Isrc -Ithird_party tests/runtime/test_resource_manager.cpp \
      src/runtime/engine/context_cache/context_cost.cpp \
      src/runtime/engine/context_cache/context_cost_defaults.cpp -o test_resource_manager
  ./test_resource_manager
  ```

- **Replay traces:** the traces used in §6.2 contain customer data and are not published. The method
  needs only a recorded sequence of chat-completion bodies with their screenshots. Replay them in
  order, set each request's generation length to the recorded one, and read `timings.cache_n` per
  request.
