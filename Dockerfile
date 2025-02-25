# Use a PyTorch image with CUDA support for GPU acceleration
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy the application files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
