import torch
from fastapi import FastAPI
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline
from pydantic import BaseModel
from PIL import Image
import io
import base64
from typing import Optional

app = FastAPI()

# Load Model Names
MODEL_NAME = "John6666/uber-realistic-porn-merge-xl-urpmxl-v6final-sdxl"
INPAINT_MODEL = "mrcuddle/urpm-inpainting"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Pipelines
txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_NAME).to(device)
img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(MODEL_NAME).to(device)
inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(INPAINT_MODEL).to(device)

# Enable performance optimizations
if device == "cuda":
    txt2img_pipe.enable_attention_slicing()
    img2img_pipe.enable_attention_slicing()
    inpaint_pipe.enable_attention_slicing()

# Utility function to convert PIL image to Base64
def pil_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ✅ Health check route
@app.get("/")
async def root():
    return {"message": "Stable Diffusion API is running!"}

# ✅ Text-to-Image API
class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: int = 50
    guidance_scale: float = 7.5
    batch_size: int = 1

@app.post("/txt2img")
async def txt2img(request: Txt2ImgRequest):
    images = txt2img_pipe(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.steps,
        guidance_scale=request.guidance_scale,
        num_images_per_prompt=request.batch_size
    ).images
    return {"images": [pil_to_base64(img) for img in images]}

# ✅ Image-to-Image API
class Img2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 0.75
    batch_size: int = 1
    image: str  # Base64 encoded image

@app.post("/img2img")
async def img2img(request: Img2ImgRequest):
    image_bytes = base64.b64decode(request.image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    images = img2img_pipe(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        image=image,
        strength=request.strength,
        num_inference_steps=request.steps,
        guidance_scale=request.guidance_scale,
        num_images_per_prompt=request.batch_size
    ).images

    return {"images": [pil_to_base64(img) for img in images]}

# ✅ Inpainting API
class InpaintRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 0.75
    batch_size: int = 1
    image: str  # Base64 encoded image
    mask: str   # Base64 encoded mask

@app.post("/inpaint")
async def inpaint(request: InpaintRequest):
    # Decode base64 images
    image_bytes = base64.b64decode(request.image)
    mask_bytes = base64.b64decode(request.mask)
    
    # Open images
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")  # Convert mask to grayscale

    images = inpaint_pipe(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        image=image,
        mask_image=mask,
        strength=request.strength,
        num_inference_steps=request.steps,
        guidance_scale=request.guidance_scale,
        num_images_per_prompt=request.batch_size
    ).images

    return {"images": [pil_to_base64(img) for img in images]}
