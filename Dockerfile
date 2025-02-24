FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files into the container
COPY requirements.txt app.py /app/

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT:-8000}

# Command to run the Gradio interface
CMD ["python", "app.py"]
