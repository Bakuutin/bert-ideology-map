# %%
# !pip install -Ur requirements.txt

# %% 
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum
from uuid import UUID
import yaml
from encoder import CachedBERTEncoder

# %%

class Persona(BaseModel):
    name: str
    attrs: dict[str, int | str | float]
    memes: list[str]


with open('memes.yaml', 'r') as f:
    yaml_data = yaml.safe_load(f)


personas = []

for i,persona in enumerate(yaml_data['personas']):
    try:
        personas.append(Persona.model_validate({
            'id': i,
            **persona,
        }))
    except Exception as e:
        print(f"Error parsing persona: {e}")
        print(persona)
        raise e





# %%
# Initialize cached BERT encoder
encoder = CachedBERTEncoder(model_name="bert-base-uncased", cache_path="output/embeddings_cache.db")

# Print initial cache stats
print("Initial cache statistics:")
stats = encoder.get_cache_stats()
print(f"  Total embeddings: {stats['total_embeddings']}")
print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")

# %%
log_dir = 'output'
os.makedirs(log_dir, exist_ok=True)

# %%

memes = [(meme, persona) for persona in personas for meme in persona.memes]




# %%
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    f.write("Meme\tName\n")
    for meme, persona in tqdm(memes, desc="Writing token metadata"):
        f.write(f"{meme}\t{persona.name}\n")

# %%
# Save embedding weights as TSV file (vectors) - now with caching!
print("Saving embedding vectors with caching...")
with open(os.path.join(log_dir, 'embeddings.tsv'), "w") as f:
    for meme, _ in tqdm(memes, desc="Writing embedding vectors"):
        # Use cached encoder - will automatically cache new embeddings
        embedding = encoder.encode(meme).squeeze().numpy()
        # Convert embedding to tab-separated values
        embedding_str = '\t'.join([str(x) for x in embedding])
        f.write(f"{embedding_str}\n")

# Print final cache stats
print("\nFinal cache statistics:")
stats = encoder.get_cache_stats()
print(f"  Total embeddings: {stats['total_embeddings']}")
print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")

print(f"\nEmbeddings and metadata saved in: {log_dir}")
print(f"Files created:")
print(f"  - metadata.tsv: Token labels")
print(f"  - embeddings.tsv: Embedding vectors (tab-separated)")
print(f"  - embeddings_cache.db: SQLite cache for embeddings")

# %%

