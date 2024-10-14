import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from threading import Lock
import torch

# Global variables for caching
model_cache = {"tokenizer": None, "model": None}
cache_lock = Lock()

# Set the path for saving quantized model in Google Drive
quantized_model_path = '/content/gdrive/MyDrive/quantized_llama_model/'

# Function to load and quantize the LLaMA model
def load_and_quantize_llama_model():
    # Path to your model in Google Drive
    cache_dir = '/content/gdrive/MyDrive/tinyLlama3model_cache'  # Your model directory in GDrive
    print(f"Loading model from: {cache_dir}")

    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"Model not found in cache directory: {cache_dir}. Please download and place the model in the cache.")

    # Check if the model and tokenizer are already loaded in memory
    with cache_lock:
        if model_cache["tokenizer"] is not None and model_cache["model"] is not None:
            print("Returning cached model and tokenizer")
            return model_cache["tokenizer"], model_cache["model"]

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cache_dir, local_files_only=True)

        # Configure quantization with BitsAndBytesConfig for 4-bit loading
        print("Loading and quantizing the model...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True  # Quantize to 4-bit
        )

        # Load the model with the quantization config
        model = AutoModelForCausalLM.from_pretrained(
            cache_dir,
            local_files_only=True,
            quantization_config=quantization_config,  # Apply quantization
            device_map="auto"  # Automatically select available device (GPU or CPU)
        )

        # Cache the loaded model and tokenizer
        model_cache["tokenizer"] = tokenizer
        model_cache["model"] = model

        return tokenizer, model

# Function to save the quantized model
def save_quantized_model(model):
    # Create directory if it doesn't exist
    if not os.path.exists(quantized_model_path):
        os.makedirs(quantized_model_path)

    print(f"Saving quantized model to {quantized_model_path}...")
    model.save_pretrained(quantized_model_path)

# Load, quantize and save the model
tokenizer, quantized_model = load_and_quantize_llama_model()
save_quantized_model(quantized_model)
