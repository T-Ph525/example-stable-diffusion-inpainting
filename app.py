import torch
import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from typing import List, Optional
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image
from io import BytesIO
import gradio as gr

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Stable Diffusion Models with optimization
model_id = "stabilityai/stable-diffusion-2-inpainting"

pipe_txt2img = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)

# Enable optimized inference mode
for pipe in [pipe_txt2img, pipe_img2img, pipe_inpaint]:
    pipe.enable_model_cpu_offload()
    pipe.to(device)
    if hasattr(torch, "compile"):
        pipe.unet = torch.compile(pipe.unet)  # Optimize with Torch 2.0 Compilation

app = FastAPI()

# Load LoRA dynamically
def load_lora(pipe, lora_path: Optional[str] = None):
    if lora_path:
        print(f"Loading LoRA: {lora_path}")
        pipe.load_lora_weights(lora_path)
    else:
        print("LoRA disabled.")
        pipe.unload_lora_weights()

# Efficient image processing function
async def process_image(prompt, init_image=None, mask_image=None, strength=0.75, use_lora=False, lora_path=None):
    pipe = pipe_txt2img if init_image is None and mask_image is None else pipe_img2img if mask_image is None else pipe_inpaint
    
    if use_lora:
        load_lora(pipe, lora_path)
    
    with torch.inference_mode():
        output = await asyncio.to_thread(pipe, prompt=prompt, image=init_image, mask_image=mask_image, strength=strength)
    
    img_io = BytesIO()
    output.images[0].save(img_io, format="PNG")
    img_io.seek(0)
    return img_io.getvalue()

# Batch processing with asyncio
@app.post("/generate_batch")
async def generate_batch(
    prompts: List[str] = Form(...),
    images: Optional[List[UploadFile]] = File(None),
    masks: Optional[List[UploadFile]] = File(None),
    strengths: Optional[List[float]] = Form(None),
    use_lora: bool = Form(False),
    lora_path: str = Form("")
):
    tasks = []
    for i, prompt in enumerate(prompts):
        image_bytes = await images[i].read() if images else None
        mask_bytes = await masks[i].read() if masks else None

        init_image = Image.open(BytesIO(image_bytes)).convert("RGB") if image_bytes else None
        mask_image = Image.open(BytesIO(mask_bytes)).convert("L") if mask_bytes else None
        strength = strengths[i] if strengths else 0.75

        tasks.append(process_image(prompt, init_image, mask_image, strength, use_lora, lora_path))

    results = await asyncio.gather(*tasks)
    return [{"status": "success", "output_image": result} for result in results]

# Gradio UI
def gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Optimized AI Image Generation (T2I, I2I, Inpainting)")

        with gr.Row():
            mode = gr.Radio(["Text-to-Image", "Image-to-Image", "Inpainting"], label="Generation Mode")

        with gr.Row():
            prompt = gr.Textbox(label="Prompt", placeholder="Describe your image")

        with gr.Row():
            init_image = gr.Image(label="Initial Image (for I2I/Inpainting)", type="pil")
            mask_image = gr.Image(label="Mask Image (for Inpainting)", type="pil")

        with gr.Row():
            strength = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.75, label="Strength (for I2I)")

        with gr.Row():
            use_lora = gr.Checkbox(label="Use LoRA")
            lora_path = gr.Textbox(label="LoRA Path", placeholder="Path to LoRA weights")

        submit_btn = gr.Button("Generate")
        output_img = gr.Image(label="Generated Image", type="pil")

        async def gr_generate(mode, prompt, init_image, mask_image, strength, use_lora, lora_path):
            init_image = init_image if mode != "Text-to-Image" else None
            mask_image = mask_image if mode == "Inpainting" else None
            return await process_image(prompt, init_image, mask_image, strength, use_lora, lora_path)

        submit_btn.click(gr_generate, inputs=[mode, prompt, init_image, mask_image, strength, use_lora, lora_path], outputs=output_img)

    demo.launch()

if __name__ == "__main__":
    gradio_ui()
