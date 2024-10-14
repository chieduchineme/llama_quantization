import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to the quantized model in Google Drive
quantized_model_path = '/content/gdrive/MyDrive/quantized_llama_model/'

# Function to load the quantized model from Google Drive
def load_quantized_model():
    print(f"Loading quantized model from: {quantized_model_path}")

    if not os.path.exists(quantized_model_path):
        raise FileNotFoundError(f"Quantized model not found in path: {quantized_model_path}.")

    # Load the tokenizer and model from the saved quantized model path
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
    model = AutoModelForCausalLM.from_pretrained(quantized_model_path)

    # # Check for CUDA availability and move the model to GPU if available
    # if torch.cuda.is_available():
    #     print("CUDA is available! Moving model to GPU...")
    #     model.to("cuda")  # Move model to GPU
    # else:
    #     print("CUDA is not available, using CPU.")

    return tokenizer, model

# Function to make a prediction
def generate_text(prompt, tokenizer, model, max_length=50):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # # Move the input to GPU if available
    # if torch.cuda.is_available():
    #     inputs = {key: value.to("cuda") for key, value in inputs.items()}

    # Generate the text
    print("Generating text...")
    output_tokens = model.generate(**inputs, max_length=max_length)

    # Decode the generated tokens to text
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return generated_text

# Load the quantized model and tokenizer
tokenizer, quantized_model = load_quantized_model()

# Example prompt
prompt = "Once upon a time in a small village,"

# Generate text from the quantized model
generated_text = generate_text(prompt, tokenizer, quantized_model)

# Print the generated text
print(f"Generated text: {generated_text}")
