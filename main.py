# %%
# !pip install -Ur requirements.txt

# %% 
# Import required libraries for the meme embedding pipeline
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel  # For BERT model
from tqdm import tqdm  # For progress bars
from pydantic import BaseModel, Field  # For data validation
from typing import Optional, List
from datetime import datetime
from enum import Enum
from uuid import UUID
import yaml  # For reading YAML configuration
from encoder import CachedBERTEncoder  # Custom cached encoder

# %%

# Define the Persona data structure to represent individuals and their memes
class Persona(BaseModel):
    name: str  # Person's name
    attrs: dict[str, int | str | float]  # Additional attributes (age, traits, etc.)
    influences: list[str]
    memes: list[str]  # List of memes associated with this person


# Load persona data from YAML configuration file
with open('memes.yaml', 'r') as f:
    yaml_data = yaml.safe_load(f)

# Initialize list to store validated persona objects
personas = []

# Parse and validate each persona from the YAML data
for i,persona in enumerate(yaml_data['personas']):
    try:
        # Create Persona object with validation
        personas.append(Persona.model_validate({
            'id': i,  # Add sequential ID
            **persona,  # Spread the persona data
        }))
    except Exception as e:
        print(f"Error parsing persona: {e}")
        print(persona)
        raise e

# %%
# Initialize the cached BERT encoder for efficient embedding generation
# This will cache embeddings to avoid recomputing the same memes

# model_name = "bert-base-uncased"
# model_name = "output/fine_tuned_bert"
model_name = "output/politics_finetuned"

encoder = CachedBERTEncoder(model_name=model_name, cache_path=f"output/{model_name.split('/')[-1]}_cache.db")

# Display initial cache statistics to monitor performance
print("Initial cache statistics:")
stats = encoder.get_cache_stats()
print(f"  Total embeddings: {stats['total_embeddings']}")
print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")

# %%
# Create output directory for saving results
log_dir = 'output'
os.makedirs(log_dir, exist_ok=True)

# %%

# Create a list of all (meme, persona) pairs for processing
# This flattens the data structure to process all memes across all personas

memes = []
for persona in personas:
    for meme in persona.memes:
        memes.append((meme, persona))
    # for influence in persona.influences:
    #     memes.append((f'I like {influence}', persona))

# memes = [(meme, persona) for persona in personas for meme in persona.memes]



# %%
# Save metadata mapping each meme to its associated persona name
# This creates a TSV file that can be used for visualization or analysis
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    f.write("Meme\tName\n")  # TSV header
    for meme, persona in tqdm(memes, desc="Writing token metadata"):
        f.write(f"{meme}\t{persona.name}\n")

# %%
# Generate and save embedding vectors for all memes
# This creates the numerical representations that can be used for clustering, similarity analysis, etc.
print("Saving embedding vectors...")
with open(os.path.join(log_dir, 'embeddings.tsv'), "w") as f:
    for meme, _ in tqdm(memes, desc="Writing embedding vectors"):
        # Use cached encoder - will automatically cache new embeddings for efficiency
        embedding = encoder.encode(meme).squeeze().numpy()
        # Convert embedding to tab-separated values for easy parsing
        embedding_str = '\t'.join([str(x) for x in embedding])
        f.write(f"{embedding_str}\n")

# Display final cache statistics to show efficiency gains
print("\nFinal cache statistics:")
stats = encoder.get_cache_stats()
print(f"  Total embeddings: {stats['total_embeddings']}")
print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")

# Print summary of generated files and their purposes
print(f"\nEmbeddings and metadata saved in: {log_dir}")
print(f"Files created:")
print(f"  - metadata.tsv: Token labels (meme-to-persona mapping)")
print(f"  - embeddings.tsv: Embedding vectors (tab-separated numerical representations)")
print(f"  - embeddings_cache.db: SQLite cache for embeddings")


# %%


# common = "I like that person"

# common_vector = encoder.encode(common).squeeze().numpy()

# Normalize "I like" embeddings by subtracting the common "I like that person" vector
print("Normalizing 'I like' embeddings...")
normalized_embeddings = []

for i, (meme, persona) in enumerate(memes):
    embedding = encoder.encode(meme).squeeze().numpy()
    normalized = embedding / np.linalg.norm(embedding)
    normalized_embeddings.append(normalized)

# Save normalized embeddings
print("Saving normalized embeddings...")
with open(os.path.join(log_dir, 'normalized_embeddings.tsv'), "w") as f:
    for embedding in tqdm(normalized_embeddings, desc="Writing normalized embeddings"):
        embedding_str = '\t'.join([str(x) for x in embedding])
        f.write(f"{embedding_str}\n")


print(f"Normalized embeddings saved to: {log_dir}/normalized_embeddings.tsv")

# %%


# calculate 2d t-sne
from sklearn.manifold import TSNE

embeddings = np.loadtxt(os.path.join(log_dir, 'normalized_embeddings.tsv'), delimiter='\t')

# Load metadata to get labels
metadata = []
with open(os.path.join(log_dir, 'metadata.tsv'), 'r') as f:
    next(f)  # Skip header
    for line in f:
        meme, name = line.strip().split('\t')
        metadata.append((meme, name))

# Drop duplicates based on meme text
unique_data = []
unique_embeddings = []
seen_memes = set()

for i, (meme, name) in enumerate(metadata):
    if meme not in seen_memes:
        seen_memes.add(meme)
        unique_data.append((meme, name))
        unique_embeddings.append(embeddings[i])

# Convert to numpy arrays
unique_embeddings = np.array(unique_embeddings)

print(f"Original data points: {len(embeddings)}")
print(f"Unique data points after dropping duplicates: {len(unique_embeddings)}")



# Define dragons (ideological positions) with their memes and colors
dragons = [
    {
        "name": "Republican",
        "color": "#ff0000",  # Red
        "memes": ["Lower taxes", "Protect the Second Amendment", "I support Trump", "Smaller government", "Free market capitalism"]
    },
    {
        "name": "Democrat", 
        "color": "#0000ff",  # Blue
        "memes": ["Universal healthcare", "Climate change is real", "I support Biden", "Social safety net", "Progressive policies"]
    },
    # {
    #     "name": "Libertarian",
    #     "color": "#00ff00",  # Green
    #     "memes": ["I am a libertarian", "Minimal government", "Free markets", "Individual liberty", "Constitutional rights"]
    # }
]

# Calculate average embeddings for each dragon
dragon_positions = {}
dragon_vectors = {}

for dragon in dragons:
    embeddings = []
    for meme in dragon["memes"]:
        embeddings.append(encoder.encode(meme).squeeze().numpy())
    
    # Average the embeddings
    dragon_positions[dragon["name"]] = np.mean(embeddings, axis=0)

average_dragon = np.mean([dragon_positions[dragon["name"]] for dragon in dragons], axis=0)

# Calculate difference vectors (each dragon vs others)
for i, dragon in enumerate(dragons):
    other_positions = [dragon_positions[d["name"]] for j, d in enumerate(dragons) if j != i]
    other_mean = np.mean(other_positions, axis=0)
    
    dragon_vectors[dragon["name"]] = (dragon_positions[dragon["name"]] - other_mean) / np.linalg.norm(dragon_positions[dragon["name"]] - other_mean)

# Calculate cosine similarities for each embedding
dragon_similarities = {dragon["name"]: [] for dragon in dragons}

for embedding in unique_embeddings:
    # Normalize the embedding
    norm_embedding = embedding / np.linalg.norm(embedding)
    
    # Calculate cosine similarities for each dragon
    for dragon in dragons:
        sim = np.dot(norm_embedding, dragon_vectors[dragon["name"]])
        dragon_similarities[dragon["name"]].append(sim)

# Convert to numpy arrays and normalize similarities to range 0-1
for dragon in dragons:
    dragon_similarities[dragon["name"]] = np.array(dragon_similarities[dragon["name"]])
    similarities = dragon_similarities[dragon["name"]]
    dragon_similarities[dragon["name"]] = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    print(f"{dragon['name']} similarities range: {dragon_similarities[dragon['name']].min():.3f} to {dragon_similarities[dragon['name']].max():.3f}")

# %%

# plot 2d t-sne with different perplexity values
import matplotlib.pyplot as plt

# Define range of perplexity values to test
perplexity_values = [5, 10, 15, 20, 30, 50, 100]

# Create subplots for each perplexity value
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
axes = axes.flatten()

for idx, perplexity in enumerate(perplexity_values):
    if idx >= len(axes):
        break
        
    ax = axes[idx]
    
    # Apply t-SNE with current perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=300, random_state=42)
    embeddings_2d = tsne.fit_transform(unique_embeddings)
    
    # Create RGB colors based on dragon colors and intensities
    colors = []
    for i in range(len(unique_embeddings)):
        # Start with black (no color)
        final_color = [0, 0, 0]
        
        # For each dragon, add its color weighted by its intensity
        for dragon in dragons:
            intensity = dragon_similarities[dragon["name"]][i]
            
            # Convert hex color to RGB using matplotlib
            from matplotlib.colors import to_rgb
            dragon_rgb = to_rgb(dragon["color"])
            
            # Add weighted dragon color to final color
            for c in range(3):
                final_color[c] += dragon_rgb[c] * intensity
        
        # Clamp to valid RGB range [0, 1]
        final_color = [max(0, min(1, c)) for c in final_color]
        
        colors.append(final_color)
    
    # Plot points with RGB colors
    for i, (meme, name) in enumerate(unique_data):
        ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                   alpha=0.7, s=30, c=[colors[i]], label=name if i == 0 else "")
    
    # Add a few sample labels (not all to avoid clutter)
    sample_indices = np.random.choice(len(unique_data), min(8, len(unique_data)), replace=False)
    for i in sample_indices:
        meme, name = unique_data[i]
        label = meme[:20] + "..." if len(meme) > 20 else meme
        ax.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    fontsize=6, alpha=0.8)
    
    dragon_names = [dragon["name"] for dragon in dragons]
    title = f't-SNE (perplexity={perplexity})\n'
    title += f'Dragons: {", ".join(dragon_names)}'
    ax.set_title(title)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')

# Remove any unused subplots
for idx in range(len(perplexity_values), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()

# %%

# %%

# %%

# Draw histograms for each dragon
fig_hist, ax_hist = plt.subplots(1, len(dragons), figsize=(6*len(dragons), 5))

for i, dragon in enumerate(dragons):
    ax_hist[i].hist(dragon_similarities[dragon["name"]], bins=20, color=dragon["color"], alpha=0.7, edgecolor='black')
    ax_hist[i].set_title(f'{dragon["name"]} Distribution')
    ax_hist[i].set_xlabel(f'{dragon["name"]} Similarity')
    ax_hist[i].set_ylabel('Count')

plt.tight_layout()
plt.show()

# %%
