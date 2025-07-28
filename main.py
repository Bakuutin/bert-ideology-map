# %%
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import Optional, List
from datetime import datetime
from enum import Enum
from uuid import UUID
import yaml
from encoder import CachedBERTEncoder

# %%
with open('memes.yaml', 'r') as f:
    yaml_data = yaml.safe_load(f)

yaml_memes = []
for persona in yaml_data['personas']:
    yaml_memes.extend(persona['memes'])

print(f"Loaded {len(yaml_memes)} memes from YAML")

csv_memes = []
if os.path.exists('reddit_politics.csv'):
    print("Loading memes from reddit_politics.csv...")
    df = pd.read_csv('reddit_politics.csv')
    
    df_filtered = df[
        (df['body'].str.len() >= 20) & 
        (df['body'].str.len() <= 200) &
        (df['body'].notna())
    ]
    
    csv_memes = df_filtered['body'].tolist()
    print(f"Loaded {len(csv_memes)} memes from CSV")
else:
    print("reddit_politics.csv not found, skipping CSV memes")

memes = yaml_memes + csv_memes

cleaned_memes = []
for meme in memes:
    cleaned_meme = meme.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    import re
    cleaned_meme = re.sub(r'\s+', ' ', cleaned_meme).strip()
    cleaned_memes.append(cleaned_meme)

memes = cleaned_memes
print(f"Total memes loaded: {len(memes)}")

# %%
model_name = "output/politics_finetuned"

encoder = CachedBERTEncoder(model_name=model_name, cache_path=f"output/{model_name.split('/')[-1]}_cache.db")

print("Initial cache statistics:")
stats = encoder.get_cache_stats()
print(f"  Total embeddings: {stats['total_embeddings']}")
print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")

# %%
log_dir = 'output'
os.makedirs(log_dir, exist_ok=True)

with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    f.write("Meme\tIndex\n")
    for i, meme in enumerate(tqdm(memes, desc="Writing token metadata")):
        f.write(f"{meme}\t{i}\n")

# %%
print("Saving embedding vectors...")
with open(os.path.join(log_dir, 'embeddings.tsv'), "w") as f:
    for meme in tqdm(memes, desc="Writing embedding vectors"):
        embedding = encoder.encode(meme).squeeze().numpy()
        embedding_str = '\t'.join([str(x) for x in embedding])
        f.write(f"{embedding_str}\n")

print("\nFinal cache statistics:")
stats = encoder.get_cache_stats()
print(f"  Total embeddings: {stats['total_embeddings']}")
print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")

print(f"\nEmbeddings and metadata saved in: {log_dir}")
print(f"Files created:")
print(f"  - metadata.tsv: Token labels (meme-to-index mapping)")
print(f"  - embeddings.tsv: Embedding vectors (tab-separated numerical representations)")
print(f"  - embeddings_cache.db: SQLite cache for embeddings")

# %%
print("Normalizing embeddings...")
normalized_embeddings = []

for meme in memes:
    embedding = encoder.encode(meme).squeeze().numpy()
    normalized = embedding / np.linalg.norm(embedding)
    normalized_embeddings.append(normalized)

print("Saving normalized embeddings...")
with open(os.path.join(log_dir, 'normalized_embeddings.tsv'), "w") as f:
    for embedding in tqdm(normalized_embeddings, desc="Writing normalized embeddings"):
        embedding_str = '\t'.join([str(x) for x in embedding])
        f.write(f"{embedding_str}\n")

print(f"Normalized embeddings saved to: {log_dir}/normalized_embeddings.tsv")

with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    f.write("Meme\tIndex\n")
    for i, meme in enumerate(tqdm(memes, desc="Writing token metadata")):
        f.write(f"{meme}\t{i}\n")

# %%
from sklearn.manifold import TSNE

embeddings = np.loadtxt(os.path.join(log_dir, 'normalized_embeddings.tsv'), delimiter='\t')

metadata = []
with open(os.path.join(log_dir, 'metadata.tsv'), 'r') as f:
    next(f)
    for line in f:
        meme, index = line.strip().split('\t')
        metadata.append((meme, index))

unique_data = []
unique_embeddings = []
seen_memes = set()

for i, (meme, index) in enumerate(metadata):
    if meme not in seen_memes:
        seen_memes.add(meme)
        unique_data.append((meme, index))
        unique_embeddings.append(embeddings[i])

unique_embeddings = np.array(unique_embeddings)

print(f"Original data points: {len(embeddings)}")
print(f"Unique data points after dropping duplicates: {len(unique_embeddings)}")

# %%
dragons = [
    {
        "name": "Republican",
        "color": "#ff0000",
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
        "color": "#0000ff",
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

dragon_positions = {}
dragon_vectors = {}

for dragon in dragons:
    embeddings = []
    for meme in dragon["memes"]:
        embeddings.append(encoder.encode(meme).squeeze().numpy())
    
    dragon_positions[dragon["name"]] = np.mean(embeddings, axis=0)

average_dragon = np.mean([dragon_positions[dragon["name"]] for dragon in dragons], axis=0)

for i, dragon in enumerate(dragons):
    other_positions = [dragon_positions[d["name"]] for j, d in enumerate(dragons) if j != i]
    other_mean = np.mean(other_positions, axis=0)
    
    dragon_vectors[dragon["name"]] = (dragon_positions[dragon["name"]] - other_mean) / np.linalg.norm(dragon_positions[dragon["name"]] - other_mean)

dragon_similarities = {dragon["name"]: [] for dragon in dragons}

for embedding in unique_embeddings:
    norm_embedding = embedding / np.linalg.norm(embedding)
    
    for dragon in dragons:
        sim = np.dot(norm_embedding, dragon_vectors[dragon["name"]])
        dragon_similarities[dragon["name"]].append(sim)

for dragon in dragons:
    dragon_similarities[dragon["name"]] = np.array(dragon_similarities[dragon["name"]])
    similarities = dragon_similarities[dragon["name"]]
    dragon_similarities[dragon["name"]] = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    print(f"{dragon['name']} similarities range: {dragon_similarities[dragon['name']].min():.3f} to {dragon_similarities[dragon['name']].max():.3f}")

# %%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

perplexity_values = [10]

sample_size = min(1500, len(unique_embeddings))
print(f"Sampling {sample_size} memes from {len(unique_embeddings)} total memes for t-SNE")

np.random.seed(42)
sample_indices = np.random.choice(len(unique_embeddings), sample_size, replace=False)

sampled_embeddings = unique_embeddings[sample_indices]
sampled_data = [unique_data[i] for i in sample_indices]
sampled_similarities = {dragon["name"]: dragon_similarities[dragon["name"]][sample_indices] for dragon in dragons}

print("Applying PCA to reduce dimensionality...")
print(f"Original embedding dimensions: {sampled_embeddings.shape[1]}")

n_components = min(50, sampled_embeddings.shape[1])
pca = PCA(n_components=n_components, random_state=42)
embeddings_pca = pca.fit_transform(sampled_embeddings)

print(f"PCA reduced dimensions: {embeddings_pca.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

plt.figure(figsize=(12, 10))
ax = plt.gca()

perplexity = perplexity_values[0]

tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=300, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_pca)

for i, (meme, index) in enumerate(sampled_data):
    for dragon in dragons:
        intensity = sampled_similarities[dragon["name"]][i]
    
        from matplotlib.colors import to_rgb
        dragon_rgb = to_rgb(dragon["color"])
        
        ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                   alpha=max(0, intensity/2), s=140, c=[dragon_rgb], 
                   label=f"{dragon['name']} connection" if i == 0 else "")

label_indices = np.random.choice(len(sampled_data), min(30, len(sampled_data)), replace=False)
for i in label_indices:
    meme, index = sampled_data[i]
    label = meme[:20] if len(meme) > 20 else meme
    ax.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                fontsize=10, alpha=0.9)

dragon_names = [dragon["name"] for dragon in dragons]

ax.set_title(f't-SNE (perplexity={perplexity})')
ax.set_xlabel('unitless')
ax.set_ylabel('unitless')

plt.tight_layout()
plt.show()

# %%
low_rung_pole = [
    "If you're not with us, you're against us",
    "They always lie",
    "That's just what people like them do",
    "Everyone knows this is true",
    "You can't trust those people",
    "Do your own research (but only our sources)",
    "They're all brainwashed",
    "It's common sense",
    "This is how it's always been",
    "No point in talking to them"
]

high_rung_pole = [
    "What's the best argument against my view?",
    "Let's define our terms",
    "I might be wrong",
    "What evidence would change your mind?",
    "That's a good point—I hadn't considered it",
    "Let's separate facts from interpretations",
    "Could both sides be partially right?",
    "What are the assumptions behind this belief?",
    "I'm still thinking this through",
    "Let's be precise"
]

low_rung_embeddings = []
for meme in low_rung_pole:
    low_rung_embeddings.append(encoder.encode(meme).squeeze().numpy())
low_rung_mean = np.mean(low_rung_embeddings, axis=0)

high_rung_embeddings = []
for meme in high_rung_pole:
    high_rung_embeddings.append(encoder.encode(meme).squeeze().numpy())
high_rung_mean = np.mean(high_rung_embeddings, axis=0)

low_high_axis = (low_rung_mean - high_rung_mean) / np.linalg.norm(low_rung_mean - high_rung_mean)

low_high_scores = []
democratic_republican_scores = []

for embedding in unique_embeddings:
    norm_embedding = embedding / np.linalg.norm(embedding)
    
    lh_score = np.dot(norm_embedding, low_high_axis)
    low_high_scores.append(lh_score)
    
    dr_axis = (dragon_vectors["Republican"] - dragon_vectors["Democrat"]) / np.linalg.norm(dragon_vectors["Republican"] - dragon_vectors["Democrat"])
    dr_score = np.dot(norm_embedding, dr_axis)
    democratic_republican_scores.append(dr_score)

low_high_scores = np.array(low_high_scores)
democratic_republican_scores = np.array(democratic_republican_scores)

low_high_scores = (low_high_scores - low_high_scores.min()) / (low_high_scores.max() - low_high_scores.min()) * 2 - 1
democratic_republican_scores = (democratic_republican_scores - democratic_republican_scores.min()) / (democratic_republican_scores.max() - democratic_republican_scores.min()) * 2 - 1

print(f"Low-Rung vs High-Rung scores range: {low_high_scores.min():.3f} to {low_high_scores.max():.3f}")
print(f"Democratic-Republican scores range: {democratic_republican_scores.min():.3f} to {democratic_republican_scores.max():.3f}")

# %%
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

sample_size = len(unique_embeddings)

sample_indices = np.random.choice(len(unique_embeddings), sample_size, replace=False)

sampled_lh_scores = low_high_scores[sample_indices]
sampled_dr_scores = democratic_republican_scores[sample_indices]
sampled_data = [unique_data[i] for i in sample_indices]

colors = []
for i in range(len(sampled_lh_scores)):
    redness = (sampled_dr_scores[i] + 1) / 2
    blueness = 1 - redness
    colors.append((redness, 0, blueness))

ax.scatter(sampled_dr_scores, sampled_lh_scores, c=colors, alpha=0.7, s=5)

ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

bbox = {
    'boxstyle': 'round,pad=0.3',
    'alpha': 0.7
}

ax.text(
    0.2, 0.8, 'High-Rung\nDemocrat', transform=ax.transAxes, ha='center', va='center', 
    weight='bold',
    bbox={**bbox, 'facecolor': 'pink'}
)
ax.text(
    0.8, 0.8, 'High-Rung\nRepublican', transform=ax.transAxes, ha='center', va='center',
    weight='bold',
    bbox={**bbox, 'facecolor': 'lightblue'}
)

ax.text(
    0.2, 0.2, 'Low-Rung\nDemocrat', transform=ax.transAxes, ha='center', va='center',
    weight='bold',
    color='white',
    bbox={**bbox, 'facecolor': 'red'}
)
ax.text(
    0.8, 0.2, 'Low-Rung\nRepublican', transform=ax.transAxes, ha='center', va='center',
    weight='bold',
    color='white',
    bbox={**bbox, 'facecolor': 'blue'}
)

ax.set_xlabel('Democrat ← → Republican')
ax.set_ylabel('Low-Rung ← → High-Rung')
ax.set_title('Ideological Space')
ax.grid(True, alpha=0.5)

x_min, x_max = sampled_lh_scores.min(), sampled_lh_scores.max()
y_min, y_max = sampled_dr_scores.min(), sampled_dr_scores.max()

padding = 0
x_range = x_max - x_min
y_range = y_max - y_min
max_range = max(x_range, y_range)

x_center = (x_min + x_max) / 2
y_center = (y_min + y_max) / 2

ax.set_xlim(x_center - max_range/2 - padding, x_center + max_range/2 + padding)
ax.set_ylim(y_center - max_range/2 - padding, y_center + max_range/2 + padding)

ax.set_aspect('equal')

plt.tight_layout()
plt.show()

# %%
fig_hist, ax_hist = plt.subplots(1, len(dragons), figsize=(6*len(dragons), 5))

for i, dragon in enumerate(dragons[::-1]):
    ax_hist[i].hist(dragon_similarities[dragon["name"]], bins=20, color=dragon["color"], alpha=0.7, edgecolor='black')
    ax_hist[i].set_title(f'{dragon["name"]} Distribution')
    ax_hist[i].set_xlabel(f'{dragon["name"]} Similarity')
    ax_hist[i].set_ylabel('Count')

plt.tight_layout()
plt.show()
