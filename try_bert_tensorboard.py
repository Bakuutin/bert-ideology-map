# %%
# install transformers and torch only (no tensorflow)
# !pip install transformers torch numpy tqdm

# %% 
# BERT Embeddings Export for Visualization
# This script prepares BERT embeddings for visualization without TensorFlow

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# %%
# Load BERT model and tokenizer
print("Loading BERT model and tokenizer...")
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def encode(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token

# %%
# Set up logs directory
log_dir = 'output'
os.makedirs(log_dir, exist_ok=True)

# %%


memes = open('memes.txt', 'r').read().splitlines()


# %%
# Save metadata (token labels) in TSV format
print("Saving token metadata...")
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    for meme in tqdm(memes, desc="Writing token metadata"):
        f.write(f"{meme}\n")

# %%
# Save embedding weights as TSV file (vectors)
print("Saving embedding vectors...")
with open(os.path.join(log_dir, 'embeddings.tsv'), "w") as f:
    for meme in tqdm(memes, desc="Writing embedding vectors"):
        embedding = encode(meme).squeeze().numpy()
        # Convert embedding to tab-separated values
        embedding_str = '\t'.join([str(x) for x in embedding])
        f.write(f"{embedding_str}\n")

print(f"Embeddings and metadata saved in: {log_dir}")
print(f"Files created:")
print(f"  - metadata.tsv: Token labels")
print(f"  - embeddings.tsv: Embedding vectors (tab-separated)")

# %%
# Example: Get embeddings for specific phrases


# Test some phrases
print("Computing phrase similarities...")
vec1 = encode("Abolish the police")
vec2 = encode("Defund the police")
vec3 = encode("Support law enforcement")

# Cosine similarity
similarity_1_2 = torch.nn.functional.cosine_similarity(vec1, vec2)
similarity_1_3 = torch.nn.functional.cosine_similarity(vec1, vec3)

print(f"\nSimilarity between 'Abolish the police' and 'Defund the police': {similarity_1_2.item():.4f}")
print(f"Similarity between 'Abolish the police' and 'Support law enforcement': {similarity_1_3.item():.4f}")

# %%
# Additional utility: Save phrase embeddings
phrases = [
    "Abolish the police",
    "Defund the police", 
    "Support law enforcement",
    "Community safety",
    "Public security"
]

print("Computing phrase embeddings...")
phrase_embeddings = []
for phrase in tqdm(phrases, desc="Processing phrases"):
    vec = encode(phrase)
    phrase_embeddings.append(vec.squeeze().numpy())

phrase_embeddings = np.array(phrase_embeddings)

# Save phrase embeddings as TSV file (vectors)
print("Saving phrase embeddings...")
with open(os.path.join(log_dir, 'phrase_embeddings.tsv'), "w") as f:
    for phrase_embedding in tqdm(phrase_embeddings, desc="Writing phrase embeddings"):
        # Convert embedding to tab-separated values
        embedding_str = '\t'.join([str(x) for x in phrase_embedding])
        f.write(f"{embedding_str}\n")

# Save phrase metadata
with open(os.path.join(log_dir, 'phrase_metadata.tsv'), 'w') as f:
    for phrase in phrases:
        f.write(f"{phrase}\n")

print(f"\nPhrase embeddings saved:")
print(f"  - phrase_embeddings.tsv: Phrase embedding vectors (tab-separated)")
print(f"  - phrase_metadata.tsv: Phrase labels") 
# %%
