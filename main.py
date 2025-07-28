# %%
# !pip install -Ur requirements.txt

# %% 
# Import required libraries for the meme embedding pipeline
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel  # For BERT model
from tqdm import tqdm  # For progress bars
from typing import Optional, List
from datetime import datetime
from enum import Enum
from uuid import UUID
import yaml  # For reading YAML configuration
from encoder import CachedBERTEncoder  # Custom cached encoder

# %%

# Load meme data from YAML configuration file
with open('memes.yaml', 'r') as f:
    yaml_data = yaml.safe_load(f)

# Extract all memes from personas into a simple list
yaml_memes = []
for persona in yaml_data['personas']:
    yaml_memes.extend(persona['memes'])

print(f"Loaded {len(yaml_memes)} memes from YAML")

# Load memes from reddit_politics.csv
csv_memes = []
if os.path.exists('reddit_politics.csv'):
    print("Loading memes from reddit_politics.csv...")
    df = pd.read_csv('reddit_politics.csv')
    
    # Filter for comments with body length between 20 and 200 characters
    df_filtered = df[
        (df['body'].str.len() >= 20) & 
        (df['body'].str.len() <= 200) &
        (df['body'].notna())
    ]
    
    # df_filtered = df_filtered.head(000)
    
    # Extract the body text as memes
    csv_memes = df_filtered['body'].tolist()
    print(f"Loaded {len(csv_memes)} memes from CSV")
else:
    print("reddit_politics.csv not found, skipping CSV memes")

# Combine all memes
memes = yaml_memes + csv_memes
# memes = csv_memes

# Clean memes by replacing tabs and newlines with spaces
cleaned_memes = []
for meme in memes:
    # Replace tabs and newlines with spaces, then normalize multiple spaces
    cleaned_meme = meme.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    # Remove multiple consecutive spaces
    import re
    cleaned_meme = re.sub(r'\s+', ' ', cleaned_meme).strip()
    cleaned_memes.append(cleaned_meme)

memes = cleaned_memes
print(f"Total memes loaded: {len(memes)}")

# %%
# Initialize the cached BERT encoder for efficient embedding generation
# This will cache embeddings to avoid recomputing the same memes

# model_name = "bert-base-uncased" # 110M
# model_name = "bert-large-uncased" # 340M
model_name = "output/politics_finetuned" # 110M finetuned on r/politics

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

# Save metadata mapping each meme to its index
# This creates a TSV file that can be used for visualization or analysis
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    f.write("Meme\tIndex\n")  # TSV header
    for i, meme in enumerate(tqdm(memes, desc="Writing token metadata")):
        f.write(f"{meme}\t{i}\n")

# %%
# Generate and save embedding vectors for all memes
# This creates the numerical representations that can be used for clustering, similarity analysis, etc.
print("Saving embedding vectors...")
with open(os.path.join(log_dir, 'embeddings.tsv'), "w") as f:
    for meme in tqdm(memes, desc="Writing embedding vectors"):
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
print(f"  - metadata.tsv: Token labels (meme-to-index mapping)")
print(f"  - embeddings.tsv: Embedding vectors (tab-separated numerical representations)")
print(f"  - embeddings_cache.db: SQLite cache for embeddings")


# %%


# common = "I like that person"

# common_vector = encoder.encode(common).squeeze().numpy()

# Normalize "I like" embeddings by subtracting the common "I like that person" vector
print("Normalizing embeddings...")
normalized_embeddings = []

for meme in memes:
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

# save metadata to metadata.tsv
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    f.write("Meme\tIndex\n")  # TSV header
    for i, meme in enumerate(tqdm(memes, desc="Writing token metadata")):
        f.write(f"{meme}\t{i}\n")

# %%


# calculate 2d t-sne
from sklearn.manifold import TSNE

embeddings = np.loadtxt(os.path.join(log_dir, 'normalized_embeddings.tsv'), delimiter='\t')

# Load metadata to get labels
metadata = []
with open(os.path.join(log_dir, 'metadata.tsv'), 'r') as f:
    next(f)  # Skip header
    for line in f:
        meme, index = line.strip().split('\t')
        metadata.append((meme, index))

# Drop duplicates based on meme text
unique_data = []
unique_embeddings = []
seen_memes = set()

for i, (meme, index) in enumerate(metadata):
    if meme not in seen_memes:
        seen_memes.add(meme)
        unique_data.append((meme, index))
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
        "memes": [
            "Build the wall",
            "Stop the steal",
            "Taxation is theft",
            "Guns don't kill people, people do",
            "America First",
            "Back the blue",
            "Drain the swamp",
            "Make America Great Again",
            "End woke culture",
            "Keep men out of women's sports"
        ]
    },
    {
        "name": "Democrat", 
        "color": "#0000ff",  # Blue
        "memes": [
            "Healthcare is a human right",
            "Black Lives Matter",
            "Protect trans kids",
            "No justice, no peace",
            "Tax the rich",
            "Ban assault weapons",
            "Green New Deal",
            "Reproductive rights are human rights",
            "Love is love",
            "Science is real"
        ]
    }
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
from sklearn.decomposition import PCA

# Define range of perplexity values to test
perplexity_values = [10]

# Sample random subset of ~500 memes for t-SNE
sample_size = min(1500, len(unique_embeddings))
print(f"Sampling {sample_size} memes from {len(unique_embeddings)} total memes for t-SNE")

# Create random indices for sampling
np.random.seed(42)  # For reproducibility
sample_indices = np.random.choice(len(unique_embeddings), sample_size, replace=False)

# Sample the embeddings and metadata
sampled_embeddings = unique_embeddings[sample_indices]
sampled_data = [unique_data[i] for i in sample_indices]
sampled_similarities = {dragon["name"]: dragon_similarities[dragon["name"]][sample_indices] for dragon in dragons}

# Apply PCA to reduce dimensionality before t-SNE
print("Applying PCA to reduce dimensionality...")
print(f"Original embedding dimensions: {sampled_embeddings.shape[1]}")

# Reduce to 50 dimensions (or 100 if you prefer)
n_components = min(50, sampled_embeddings.shape[1])
pca = PCA(n_components=n_components, random_state=42)
embeddings_pca = pca.fit_transform(sampled_embeddings)

print(f"PCA reduced dimensions: {embeddings_pca.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

# Create subplots for each perplexity value
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
axes = axes.flatten()

for idx, perplexity in enumerate(tqdm(perplexity_values)):
    if idx >= len(axes):
        break
        
    ax = axes[idx]
    
    # Apply t-SNE with current perplexity on PCA-reduced embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=300, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    # Plot each point multiple times - once for each dragon
    # Each point gets drawn with the color of each dragon and size proportional to connection strength
    for i, (meme, index) in enumerate(sampled_data):
        for dragon in dragons:
            # Get the connection strength to this dragon
            intensity = sampled_similarities[dragon["name"]][i]
        
            
            # Convert hex color to RGB
            from matplotlib.colors import to_rgb
            dragon_rgb = to_rgb(dragon["color"])
            
            # Plot this point with this dragon's color and size
            ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                       alpha=max(0, intensity/2), s=140, c=[dragon_rgb], 
                       label=f"{dragon['name']} connection" if i == 0 else "")
    
    # Add a few sample labels (not all to avoid clutter)
    label_indices = np.random.choice(len(sampled_data), min(30, len(sampled_data)), replace=False)
    for i in label_indices:
        meme, index = sampled_data[i]
        # label = meme
        label = meme[:20] if len(meme) > 20 else meme
        ax.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    fontsize=10, alpha=0.9)
    
    dragon_names = [dragon["name"] for dragon in dragons]
    title = f't-SNE (perplexity={perplexity})\n'
    title += f'Dragons: {", ".join(dragon_names)}'
    ax.set_title(title)
    ax.set_xlabel('попугаи')
    ax.set_ylabel('попугаи')

# Remove any unused subplots
for idx in range(len(perplexity_values), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()

# %%


# Define anarchy-authority axis using meme clusters
low_rung_pole = [
    "If you're not with us, you're against us",
    "They always lie",
    "That's just what people like them do",
    "Everyone knows this is true",
    "You can't trust those people",
    "Do your own research (but only our sources)",
    "They're all brainwashed",
    "It’s common sense",
    "This is how it's always been",
    "No point in talking to them"
]

high_rung_pole = [
    "What's the best argument against my view?",
    "Let's define our terms",
    "I might be wrong",
    "What evidence would change your mind?",
    "That’s a good point—I hadn’t considered it",
    "Let’s separate facts from interpretations",
    "Could both sides be partially right?",
    "What are the assumptions behind this belief?",
    "I'm still thinking this through",
    "Let's be precise"
]


# Calculate average embeddings for each pole
low_rung_embeddings = []
for meme in low_rung_pole:
    low_rung_embeddings.append(encoder.encode(meme).squeeze().numpy())
low_rung_mean = np.mean(low_rung_embeddings, axis=0)

high_rung_embeddings = []
for meme in high_rung_pole:
    high_rung_embeddings.append(encoder.encode(meme).squeeze().numpy())
high_rung_mean = np.mean(high_rung_embeddings, axis=0)

# Calculate the low-rung vs high-rung axis vector (flipped so low-rung is negative)
low_high_axis = (low_rung_mean - high_rung_mean) / np.linalg.norm(low_rung_mean - high_rung_mean)

# Calculate projections onto both axes for all memes
low_high_scores = []
democratic_republican_scores = []

for embedding in unique_embeddings:
    # Normalize the embedding
    norm_embedding = embedding / np.linalg.norm(embedding)
    
    # Project onto low-rung vs high-rung axis
    lh_score = np.dot(norm_embedding, low_high_axis)
    low_high_scores.append(lh_score)
    
    # Project onto democratic-republican axis (using existing dragon vectors)
    # Use the difference between Democrat and Republican vectors (flipped so Republican is positive)
    dr_axis = (dragon_vectors["Republican"] - dragon_vectors["Democrat"]) / np.linalg.norm(dragon_vectors["Republican"] - dragon_vectors["Democrat"])
    dr_score = np.dot(norm_embedding, dr_axis)
    democratic_republican_scores.append(dr_score)

# Convert to numpy arrays
low_high_scores = np.array(low_high_scores)
democratic_republican_scores = np.array(democratic_republican_scores)

# Normalize scores to range -1 to 1 for better visualization
low_high_scores = (low_high_scores - low_high_scores.min()) / (low_high_scores.max() - low_high_scores.min()) * 2 - 1
democratic_republican_scores = (democratic_republican_scores - democratic_republican_scores.min()) / (democratic_republican_scores.max() - democratic_republican_scores.min()) * 2 - 1

print(f"Low-Rung vs High-Rung scores range: {low_high_scores.min():.3f} to {low_high_scores.max():.3f}")
print(f"Democratic-Republican scores range: {democratic_republican_scores.min():.3f} to {democratic_republican_scores.max():.3f}")

# %%

# Create the combined ideological space visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Sample for visualization (same as t-SNE)
sample_size = min(1500000, len(unique_embeddings))
np.random.seed(42)
sample_indices = np.random.choice(len(unique_embeddings), sample_size, replace=False)

sampled_lh_scores = low_high_scores[sample_indices]
sampled_dr_scores = democratic_republican_scores[sample_indices]
sampled_data = [unique_data[i] for i in sample_indices]

# Create color mapping based on both axes
colors = []
for i in range(len(sampled_lh_scores)):
    # Map low-rung vs high-rung to red-green (red = low-rung, green = high-rung)
    # Map democratic-republican to blue-yellow (blue = democrat, yellow = republican)
    # lh_val = 0 # (sampled_lh_scores[i] + 1) / 2  # Normalize to 0-1
    # dr_val = (sampled_dr_scores[i] + 1) / 2  # Normalize to 0-1

    blueness = (sampled_dr_scores[i] + 1) / 2
    redness = 1 - blueness
    
    
    # Combine colors (simplified approach)
    color = (redness, 0, blueness)
    colors.append(color)

# Plot points
ax.scatter(sampled_lh_scores, sampled_dr_scores, c=colors, alpha=0.7, s=50)

# Add quadrant labels
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Add quadrant annotations
ax.text(0.2, 0.8, 'High-Rung\nDemocrat', transform=ax.transAxes, ha='center', va='center', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
ax.text(0.2, 0.2, 'Low-Rung\nDemocrat', transform=ax.transAxes, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
ax.text(0.8, 0.8, 'High-Rung\nRepublican', transform=ax.transAxes, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
ax.text(0.8, 0.2, 'Low-Rung\nRepublican', transform=ax.transAxes, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))

# Add evenly distributed labels with spacing to avoid overlap
def get_label_indices(x_coords, y_coords, max_labels=40, min_distance=30):
    """Select label indices with minimum distance between them"""
    selected_indices = []
    
    # Sort points by distance from center to ensure good coverage
    center_x, center_y = np.mean(x_coords), np.mean(y_coords)
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    sorted_indices = np.argsort(distances)
    
    for idx in sorted_indices:
        if len(selected_indices) >= max_labels:
            break
            
        # Check if this point is far enough from already selected points
        too_close = False
        for selected_idx in selected_indices:
            dist = abs(y_coords[idx] - y_coords[selected_idx])
            if dist < min_distance:
                too_close = True
                break
        
        if not too_close:
            selected_indices.append(idx)
    
    return selected_indices

# label_indices = get_label_indices(sampled_lh_scores, sampled_dr_scores, max_labels=140, min_distance=0.05)

# for i in label_indices:
#     meme, index = sampled_data[i]
#     max_length = 100
#     label = meme[:max_length] if len(meme) > max_length else meme
#     ax.annotate(label, (sampled_lh_scores[i], sampled_dr_scores[i]), 
#                 fontsize=8, alpha=0.8, xytext=(5, 0), textcoords='offset points')

ax.set_xlabel('Democrat ← → Republican')
ax.set_ylabel('Low-Rung ← → High-Rung')
ax.set_title('Ideological Space: Low-Rung vs High-Rung vs Democratic-Republican')
ax.grid(True, alpha=0.5)

# Center the plot by setting equal axis limits
x_min, x_max = sampled_lh_scores.min(), sampled_lh_scores.max()
y_min, y_max = sampled_dr_scores.min(), sampled_dr_scores.max()

# Add some padding and ensure square aspect ratio
padding = 0.5
x_range = x_max - x_min
y_range = y_max - y_min
max_range = max(x_range, y_range)

x_center = (x_min + x_max) / 2
y_center = (y_min + y_max) / 2

ax.set_xlim(x_center - max_range/2 - padding, x_center + max_range/2 + padding)
ax.set_ylim(y_center - max_range/2 - padding, y_center + max_range/2 + padding)

# Set equal aspect ratio to make it square
ax.set_aspect('equal')

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


# %%
