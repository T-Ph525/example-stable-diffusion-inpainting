import torch
from fastapi import FastAPI, File, UploadFile, Form
from diffusers import StableDiffusionXLPipeline, StableDiffusionImg2ImgXLPipeline
from pydantic import BaseModel
from PIL import Image
import io
import base64
from typing import List, Optional

app = FastAPI()

# Load the model (Optimized for performance)
MODEL_NAME = "mrcuddle/urpm-inpainting"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load both text-to-image and image-to-image pipelines
txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_NAME)
txt2img_pipe.to(device)

img2img_pipe = StableDiffusionImgXL2ImgPipeline.from_pretrained(MODEL_NAME)
img2img_pipe.to(device)

# Enable optimized memory management
if device == "cuda":
    txt2img_pipe.enable_attention_slicing()
    img2img_pipe.enable_attention_slicing()

# Request body models
class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: int = 50
    guidance_scale: float = 7.5
    batch_size: int = 1

class Img2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 0.75
    batch_size: int = 1
    image: str  # Base64 encoded image

# Utility function to convert PIL image to Base64
def pil_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ✅ Default route to check if API is running
@app.get("/")
async def root():
    return {"message": "Stable Diffusion API is running!"}

# ✅ Text-to-Image Generation API
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

# ✅ Image-to-Image Generation API
@app.post("/img2img")
async def img2img(request: Img2ImgRequest):
    # Decode the base64 input image
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
