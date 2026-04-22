from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from PIL import Image
import torch

print("Loading model...")
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini',
    subfolder='hunyuan3d-dit-v2-mini',
    torch_dtype=torch.float16
)

# Move to cuda without reassigning
pipeline.to('cuda')

# Check device
print("Device:", pipeline.device)
print("Pipeline callable:", callable(pipeline))

print("Generating 3D shape...")
image = Image.open('output.png').convert('RGB')

output = pipeline(
    image=image,
    num_inference_steps=30,
    guidance_scale=5.0
)

mesh = output[0]
mesh.export('output.stl')
print("Done! output.stl created")