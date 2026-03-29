import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.models.linear import SVDQW4A4Linear, SVDQW4A8Linear, FakeQuantFluxAttnProcessor
from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxAttention

PROMPT = "A rockstar cat playing an electric guitar"
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
WEIGHTS = "nunchaku-tech/nunchaku-flux.1-schnell/svdq-int4_r32-flux.1-schnell.safetensors"

transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(WEIGHTS)
pipeline = FluxPipeline.from_pretrained(MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")

for module in transformer.modules():
    if isinstance(module, NunchakuFluxAttention):
        for name, child in module.named_children():
            if isinstance(child, SVDQW4A4Linear):
                setattr(module, name, SVDQW4A8Linear.from_svdq_linear(child).cuda())
        module.processor = FakeQuantFluxAttnProcessor()

generator = torch.Generator(device="cuda").manual_seed(42)
img = pipeline(PROMPT, width=1024, height=1024, num_inference_steps=4, guidance_scale=0, generator=generator).images[0]
img.save("flux-schnell-attn-only-replace.png")

print("Saved flux-schnell-attn-only-replace.png")
