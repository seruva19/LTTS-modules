# LTTS-modules — extension modules registry for LTTS

This repository contains analysis modules for
[LTTS](https://github.com/seruva19/LTTS) and the `registry.json` index. LTTS can
load it from a sibling checkout.

## Using with LTTS

Clone it next to LTTS:

```
projects/
├── LTTS/
└── LTTS-modules/   ← this repo
```

Optionally point LTTS at it explicitly via env vars:

```bash
# where modules are loaded from (default: ../LTTS-modules)
LTTS_MODULES_DIR=../LTTS-modules

# registry index for update checks / remote install (default: ../LTTS-modules/registry.json)
LTTS_MODULES_REGISTRY=https://raw.githubusercontent.com/seruva19/LTTS-modules/main/registry.json
```

LTTS shows registry, installation, and update counts in the status bar.

## Module catalogue

| module | version | what it does |
|---|---|---|
| activation_patching | 0.1.0 | Activation patching (causal tracing): cache hidden states from a source run and patch them into a target run |
| activation_trajectory | 0.1.0 | Track residual-stream movement through depth: per-layer cosine similarity to the previous layer |
| attention_entropy_profiler | 0.1.0 | Per-head attention entropy (normalized by log(seq_len)) profiled across attention layers |
| attention_head_grid | 0.1.0 | Grid of per-head attention heatmaps for a layer — head specialization at a glance |
| attention_rollout | 0.1.0 | Attention rollout (Abnar & Zuidema 2020): cumulative token-to-token influence across attention layers |
| attn_visualizer | 0.0.1 | Attention visualization: heatmaps, head analysis, pattern detection, similarity matrices |
| contrastive_logit_difference | 0.1.0 | Track the logit difference between two candidate tokens through decoder depth |
| direct_logit_attribution | 0.1.0 | Per-block residual update contribution to the current top-1 token's logit |
| embedding_projector | 0.1.0 | Project per-token hidden states of a layer to 2D (PCA / t-SNE) and plot the token cloud |
| gradient_importance | 3.0.0 | Target-score gradients and activation × gradient attribution with an explicit backward pass |
| induction_head_detector | 0.1.0 | Score every attention head for induction behavior |
| layer_inspector | 1.0.0 | Inspect layer shapes, dtypes, devices, activations, and gradients |
| layer_statistics | 1.0.0 | Collect per-layer activation statistics |
| logit_lens | 0.1.0 | Logit lens: top-k token predictions decoded from each layer's hidden state |
| logit_lens_grid | 0.1.0 | Layers × positions logit-lens grid of top-1 predictions |
| memory_monitor | 0.2.0 | Track CPU and GPU memory |
| model_control_vector | 2.1.0 | Control vectors: training interface, visualization, repeng integration |
| model_introspector | 1.0.0 | Report model structure and parameter state |
| neuron_ablation | 0.1.0 | Ablate chosen hidden-state channels (zero/mean) — causal knockout experiments |
| neuron_activation_map | 0.1.0 | Token × neuron activation heatmap for a layer |
| neuron_tracker | 1.0.0 | Track and analyze neuron activation patterns across layers |
| next_token_predictions | 0.1.0 | Top-k next-token distribution at the attached layer |
| occlusion_attribution | 0.1.0 | Perturbation-based token attribution: occlude each token, measure the top-1 logit drop |
| representation_similarity | 0.1.0 | Layer-by-layer representation similarity (linear CKA), rendered as a heatmap |
| residual_stream_norm | 0.1.0 | Track L2/RMS norms and norm growth through decoder depth |
| sae_features | 0.1.0 | Sparse autoencoder feature inspection in the residual stream (needs `sae_lens`) |
| sample_module | 0.3.1 | Minimal scalar emitter — reference example for module authors |
| token_contributor | 2.0.0 | Analyze token contributions using ablation/occlusion methods |

## Installing a module from the LTTS UI

Modules Browser → Remote Registry → Install. Note: module installation
executes third-party code and is disabled by default; start LTTS with
`LTTS_ALLOW_MODULE_INSTALL=1` to enable it.

## Adding or updating a module

1. Create/edit a module directory (`<name>/init.py` with a `METADATA` dict,
   optional `requirements.txt`).
2. Regenerate the index:

   ```bash
   python generate_registry.py
   ```

3. Commit both the module and the updated `registry.json`.
