from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from transformers import ViTFeatureExtractor, ViTForImageClassification, pipeline
import torch
from PIL import Image
import io
import base64
from typing import Optional
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline,StableDiffusionXLInpaintPipeline

# Initialize FastAPI app
app = FastAPI()

# Load the sentiment analysis pipeline for prompt safety checking
sentiment_pipeline = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")

# Load Vision Transformer model for image safety checking
vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

device = "cuda" if torch.cuda.is_available() else "cpu"
vit_model.to(device)

# Load Stable Diffusion model names
MODEL_NAME = "John6666/uber-realistic-porn-merge-xl-urpmxl-v6final-sdxl"
INPAINT_MODEL = "mrcuddle/urpm-inpainting"

# Load Stable Diffusion pipelines
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

# Keywords to flag for unsafe content
unsafe_keywords = [
    "illegal", "underage", "minor", "child", "violent", "gore", "torture",
    "disturbing", "harm", "abuse", "mutilation", "extreme violence"
]

# Safety checker function for prompts
def is_safe_prompt(prompt):
    prompt_lower = prompt.lower()
    for keyword in unsafe_keywords:
        if keyword in prompt_lower:
            return False

    result = sentiment_pipeline(prompt)
    sentiment = result[0]['label']
    score = result[0]['score']

    safe_sentiments = ["POSITIVE", "NEUTRAL"]
    safe_score_threshold = 0.5

    return sentiment in safe_sentiments and score >= safe_score_threshold

# Safety checker function for images using ViT
def is_safe_image(image: Image.Image) -> bool:
    inputs = vit_feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit_model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = vit_model.config.id2label[predicted_class_idx]

    # Define unsafe classes based on your criteria
    unsafe_classes = ["violence", "gore", "disturbing"]  # Example classes
    return predicted_class not in unsafe_classes

# Health check route
@app.get("/")
async def root():
    return {"message": "Stable Diffusion API is running!"}

# Text-to-Image API with safety check
class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: int = 50
    guidance_scale: float = 7.5
    batch_size: int = 1

@app.post("/txt2img")
async def txt2img(request: Txt2ImgRequest):
    if not is_safe_prompt(request.prompt):
        raise HTTPException(status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS, detail="Prompt is not safe.")

    images = txt2img_pipe(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.steps,
        guidance_scale=request.guidance_scale,
        num_images_per_prompt=request.batch_size
    ).images

    safe_images = [pil_to_base64(img) for img in images if is_safe_image(img)]
    if not safe_images:
        raise HTTPException(status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS, detail="Generated images are not safe.")

    return {"images": safe_images}

# Image-to-Image API with safety check
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
    if not is_safe_prompt(request.prompt):
        raise HTTPException(status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS, detail="Prompt is not safe.")

    image_bytes = base64.b64decode(request.image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if not is_safe_image(image):
        raise HTTPException(status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS, detail="Input image is not safe.")

    images = img2img_pipe(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        image=image,
        strength=request.strength,
        num_inference_steps=request.steps,
        guidance_scale=request.guidance_scale,
        num_images_per_prompt=request.batch_size
    ).images

    safe_images = [pil_to_base64(img) for img in images if is_safe_image(img)]
    if not safe_images:
        raise HTTPException(status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS, detail="Generated images are not safe.")

    return {"images": safe_images}

# Inpainting API with safety check
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
    if not is_safe_prompt(request.prompt):
        raise HTTPException(status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS, detail="Prompt is not safe.")

    image_bytes = base64.b64decode(request.image)
    mask_bytes = base64.b64decode(request.mask)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

    if not is_safe_image(image):
        raise HTTPException(status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS, detail="Input image is not safe.")

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

    safe_images = [pil_to_base64(img) for img in images if is_safe_image(img)]
    if not safe_images:
        raise HTTPException(status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS, detail="Generated images are not safe.")

    return {"images": safe_images}
