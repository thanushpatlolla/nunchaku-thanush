# W4A8 Fake-Quantization Simulation

## Changes

**`nunchaku/models/linear.py`** — Added `SVDQW4A8Linear` and a `replace_with_fake_quant` swap function.

`SVDQW4A8Linear` is a drop-in replacement for `SVDQW4A4Linear` that runs entirely in PyTorch. On init, it dequantizes the packed INT4 weights to BF16 by reversing the MMA tile layout and applying per-group scales, and unpacks the low-rank projections with the existing `unpack_lowrank_weight`. All of this is cached since weights are static. The MMA reversing was definitely the most difficult part of all this. Maybe we should make a function to unpack weights and scales. 

The forward pass:
1. Computes the low-rank branch on the raw input (`x @ proj_down @ proj_up.T`)
2. Applies smooth_factor, then fake-quantizes activations to INT8 (signed symmetric, per-tensor scale)
3. Does the main matmul in BF16 against the cached dequantized weights
4. Sums the two branches (+ bias if present)

`replace_with_fake_quant(model)` recursively walks the module tree and swaps every `SVDQW4A4Linear` for a `SVDQW4A8Linear` built from its weights. This needs to be run after the weights are loaded into the model. Uses the from_svdq_linear method of the class. 

**`test_w4a8.py`** — Loads FLUX.1-schnell via the V2 backend, generates a baseline image through the normal CUDA path, swaps in fake-quant layers, and then generates a second image with the same seed for comparison.
