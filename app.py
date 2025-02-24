import torch
from diffusers import AutoPipelineForInpainting
import gradio as gr
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image

# Load the inpainting pipeline with LoRA support
pipeline = AutoPipelineForInpainting.from_pretrained(
    "mrcuddle/URPM-Inpainting", torch_dtype=torch.float16
).to("cuda")

pipeline.enable_model_cpu_offload()

# Function to enable LoRA (if weights are available)
def load_lora(lora_path=None):
    if lora_path:
        # Load LoRA weights into the model (assuming it's a LoRA-compatible checkpoint)
        pipeline.load_lora_weights(lora_path)
    else:
        # Disable LoRA
        pipeline.remove_lora_weights()

# Function to perform inpainting with optional LoRA
def image_inpaint(image_input, mask_input, prompt_input, lora_enabled=False, lora_path=None):
    if lora_enabled:
        load_lora(lora_path)  # Load LoRA weights if enabled
    
    image = pipeline(prompt=prompt_input, image=image_input, mask_image=mask_input).images[0]
    return image

# Gradio interface function
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Inpainting Service")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                mask_input = gr.Image(type="pil", label="Upload Mask")
                prompt_input = gr.Textbox(label="Enter Prompt", placeholder="Describe what you want to inpaint")
                lora_input = gr.Checkbox(label="Enable LoRA", value=False)
                lora_path_input = gr.Textbox(label="LoRA Path (optional)", placeholder="Enter LoRA model path", visible=False)
                submit_btn = gr.Button("Inpaint")

            with gr.Column():
                result_output = gr.Image(type="pil", label="Inpainted Image")

        # Show LoRA path textbox if LoRA is enabled
        lora_input.change(lambda value: {"visible": value}, inputs=[lora_input], outputs=[lora_path_input])
        
        # Connect the button click event with the image inpainting
        submit_btn.click(
            image_inpaint, 
            inputs=[image_input, mask_input, prompt_input, lora_input, lora_path_input], 
            outputs=result_output
        )

    demo.launch()

# FastAPI setup for backend
app = FastAPI()

# Function to process FastAPI image requests
@app.post("/inpaint")
async def inpaint_image(
    image_file: UploadFile = File(...),
    mask_file: UploadFile = File(...),
    prompt: str = "",
    lora_enabled: bool = False,
    lora_path: str = None
):
    # Read image and mask files
    image = Image.open(BytesIO(await image_file.read())).convert("RGB")
    mask = Image.open(BytesIO(await mask_file.read())).convert("L")
    
    # Perform inpainting with or without LoRA
    image_output = image_inpaint(image, mask, prompt, lora_enabled, lora_path)
    
    # Convert the output image to bytes and return as response
    img_byte_arr = BytesIO()
    image_output.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return StreamingResponse(img_byte_arr, media_type="image/png")

# Run FastAPI server (use Uvicorn for deployment)
if __name__ == "__main__":
    # You can run this using Uvicorn: uvicorn <filename>:app --reload
    gradio_interface()
