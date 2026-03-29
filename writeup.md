# W4A8 Fake-Quantization Simulation

## Changes

**`nunchaku/models/linear.py`** — Added `SVDQW4A8Linear` and a `replace_with_fake_quant` swap function.

`testing.md` has more detail on my debugging process. 

`SVDQW4A8Linear` is a drop-in replacement for `SVDQW4A4Linear` that runs entirely in PyTorch. On init, it dequantizes the packed INT4 weights to BF16 by reversing the MMA tile layout and applying per-group scales, unpacks the low-rank projections with the existing `unpack_lowrank_weight`, and unpacks the smooth factor from its packed wscale format. All of this is cached since weights are static. The MMA reversing was definitely the most difficult part of all this. Maybe we should make a function to unpack weights and scales. 

### Smooth factor unpacking

The `smooth_factor` tensor in the safetensors file is stored in the same packed MMA-tile format as weight scales (`packed_wscale_t`). The CUDA kernel reads it correctly through `load_wscale`/`broadcast_wscale`, but our PyTorch code was doing a plain element-wise `x / smooth_factor`, which applied the wrong smooth value to most channels due to the permutation. This was the cause of the ~0.89 cosine similarity between the kernel and our implementation. `_unpack_smooth` inverts the `pack_wscales` permutation (see `smooth_unpack.md` for the full derivation). After fixing this, per-layer cosine similarity went from ~0.89 to ~0.997.

### Forward pass

1. Computes the low-rank branch on the raw input (`x @ proj_down @ proj_up.T`)
2. Applies the unpacked smooth_factor, then fake-quantizes activations to INT8 (signed symmetric, per-group scale, `[-127, 127]`)
3. Does the main matmul in BF16 against the cached dequantized weights
4. Sums the two branches (+ bias if present)

`replace_with_fake_quant(model)` recursively walks the module tree and swaps every `SVDQW4A4Linear` for a `SVDQW4A8Linear` built from its weights. This needs to be run after the weights are loaded into the model. Uses the from_svdq_linear method of the class. 

**`nunchaku/models/transformers/transformer_flux_v2.py`** and **`nunchaku/models/attention.py`** — Both files gate the fused GELU MLP path (a CUDA op) on the linear layers being `SVDQW4A4Linear`. Without this, `fused_gelu_mlp` would still be called after swapping in `SVDQW4A8Linear` layers, bypassing the PyTorch forward pass entirely.

**`test_w4a8.py`** — Loads FLUX.1-schnell via the V2 backend, generates a baseline image through the normal CUDA path, swaps in fake-quant layers, and then generates a second image with the same seed for comparison.
