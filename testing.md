# Tests

Not all the test files are committed, just the attention only one for now. Let me know if I should!

## Attention processor logic

Replaced the fused attention kernel with `FakeQuantFluxAttnProcessor` but kept `SVDQW4A4Linear` for the linear layers. Image generation still works correctly.

`_apply_rotary` and `_unpack_rotemb` produce correct results.

## Weight/scale packing roundtrip (`test_pack_roundtrip.py`)

Creates random INT4 weights and BF16 scales, packs them with `NunchakuWeightPacker`, unpacks with `_dequantize_weights`, then repacks. All assertions pass — pack/unpack is lossless.

## Low-rank packing roundtrip (`test_lowrank_roundtrip.py`)

Creates random BF16 `proj_down` and `proj_up` matrices, packs with `pack_lowrank_weight`, unpacks with both `packer.unpack_lowrank_weight` and `nunchaku_converter.unpack_lowrank_weight`. Both unpackers return values identical to the originals.

## Single-layer forward comparison (`test_layer_compare.py`)

Compares `SVDQW4A4Linear` (kernel) vs `SVDQW4A8Linear` (PyTorch) on `transformer_blocks.0.attn.to_qkv` with random input.

- **With low-rank branch**: cosine similarity ~0.91 (before smooth fix), ~0.997 (after smooth fix)
- **Without low-rank branch** (zeroed `proj_down`/`proj_up` on both sides): cosine similarity decreased, confirming the main weight matmul path was the issue, not lora.

## Attention-only replacement end-to-end (`test_attn_only_replace.py`)

Replaces only attention linear layers with `SVDQW4A8Linear` (feedforward layers keep `SVDQW4A4Linear` and fused GELU MLP). Generated image was recognizable but heavily degraded — more than noise but not a real image. This was before the smooth factor fix.

## Full replacement end-to-end (`test_w4a8.py`)

Replaces all `SVDQW4A4Linear` layers with `SVDQW4A8Linear` via `replace_with_fake_quant`. Generates a baseline image with the kernel path and a comparison image with the PyTorch path, same seed.

## Hypothesis tests (`test_hypotheses.py`)

Systematic investigation of the cosine similarity gap on one layer (no lora zeroed):

- **H1 — Activation scale formula**: Our `amax / 7` scales have cosine similarity of only 0.674 with the kernel's ascales. The ratio kernel/ours has mean=0.53 and ranges from 0 to 2.25. Root cause turned out to be the smooth factor packing mismatch (see smooth factor tests below).
- **H3 — Factored vs baked-in scales**: `sum(w_int * x_int) * wscale * ascale` (factored) vs `sum((w_int * wscale) * (x_int * ascale))` (baked-in) give identical cosine similarity (~0.88) vs kernel. Baking scales in is not the issue. Using the kernel's own ascales in the factored matmul drops cosine to 0.63, meaning the kernel's integer activations differ from ours.
- **H5 — Per-group scale application**: Subsumed by H3.

## Lora path isolation (`test_lora_path.py`)

Compared `x @ proj_down` (ours) vs `lora_act_out` from `w4a4_layer.quantize()` (kernel).

- lora_down cosine similarity: 0.005 (essentially uncorrelated)
- Tried smoothed input, `down=False` unpack, raw packed data — all uncorrelated
- Full lora cosine similarity (through `proj_up`): 0.196

This looked like a smoking gun but turned out to be a red herring — the kernel's `quantize()` returns `lora_act_out` computed on the smoothed input internally. The mismatch was actually caused by the smooth factor being in packed MMA-tile format, which we were reading as plain per-channel values.

## Lora contribution via subtraction (`test_lora_isolated.py`)

Computed lora contribution as `full_output - no_lora_output` for both kernel and PyTorch.

- Full forward cosine: ~0.91 (before smooth fix)
- No-lora forward cosine: lower than full (before smooth fix)
- Lora contribution cosine: measured the isolated lora signal

Confirmed that the main weight matmul path was the primary source of error, not the lora path.

## Deep no-lora investigation (`test_nolora_deep.py`)

With lora zeroed on both sides, compared multiple activation quantization strategies:

- No quantization (just smoothed matmul) vs kernel
- Our INT4 fake-quant (`amax / 7`, clamp `[-7, 7]`) vs kernel
- Kernel's own ascales used for fake-quant vs kernel
- `amax / 8` with clamp `[-8, 7]` vs kernel

All gave similar cosine similarities (~0.88), confirming the gap wasn't from the quantization formula itself but from the smooth factor mismatch.

## Smooth factor debugging (`test_weight_debug2.py`)

Single-group and single-element probes with bias subtracted, lora zeroed:

- **Single-element probes**: When activating one element at a time (`x[k] = 7 * smooth[k]`), kernel vs PyTorch cosine similarity was ~1.0 for each element. Individual weight columns are correctly unpacked.
- **Single-group probes**: When activating an entire group (`x[start:end] = 7 * smooth[start:end]`), cosine similarity dropped. The kernel's `ascale` for a constant-per-smooth-channel input was not 1.0 (e.g., 4.375), meaning the kernel was seeing different values per channel — because it reads smooth from the packed wscale format, not plain order.
- **Column permutation check**: Confirmed column 0 maps correctly (no column-level permutation in weights).

This test identified the root cause: `smooth_factor` is stored in packed `packed_wscale_t` MMA-tile format. Our code was reading it as a plain 1D tensor, applying the wrong smooth value to most channels.

## Smooth factor fix verification

After adding `_unpack_smooth` to correctly unpack the smooth factor from the MMA-tile format:

- Per-layer cosine similarity: ~0.89 → ~0.997
- The activation scale mismatch (H1) and lora path mismatch resolved as downstream consequences of the same root cause

## Accumulation precision

Tested `float32` matmul vs `bfloat16` matmul in the PyTorch path. No meaningful change in cosine similarity.

## Symmetric clamp bounds

Tested clamping to `[-7, 7]` vs `[-8, 7]`. No meaningful change in cosine similarity.
