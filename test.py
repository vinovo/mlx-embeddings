from mlx_embeddings import load, generate

# Load the model and tokenizer
model_name = "mlx-community/mxbai-embed-large-v1"
model, tokenizer = load(model_name)

# Prepare the text
text = "I like reading"

# Generate embeddings (handles tokenization automatically)
outputs = generate(model, tokenizer, texts=text)
text_embeds = outputs.text_embeds  # Normalized embeddings

print(f"Text: {text}")
print(f"Embedding shape: {text_embeds.shape}")
print(f"Embedding (first 5 values): {text_embeds[0][:5]}")