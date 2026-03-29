import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.models.linear import replace_with_fake_quant

PROMPT = "A rockstar cat playing an electric guitar"
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
WEIGHTS = "nunchaku-tech/nunchaku-flux.1-schnell/svdq-int4_r32-flux.1-schnell.safetensors"

transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(WEIGHTS)
pipeline = FluxPipeline.from_pretrained(MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")

generator = torch.Generator(device="cuda").manual_seed(42)
baseline = pipeline(PROMPT, width=1024, height=1024, num_inference_steps=4, guidance_scale=0, generator=generator).images[0]
baseline.save("flux-schnell-baseline.png")

pipeline.to("cpu")
torch.cuda.empty_cache()
replace_with_fake_quant(transformer)

pipeline.enable_sequential_cpu_offload()
generator = torch.Generator(device="cuda").manual_seed(42)
w4a8 = pipeline(PROMPT, width=1024, height=1024, num_inference_steps=4, guidance_scale=0, generator=generator).images[0]
w4a8.save("flux-schnell-w4a8.png")

print("Saved flux-schnell-baseline.png and flux-schnell-w4a8.png")
