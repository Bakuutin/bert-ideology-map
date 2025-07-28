# %%
# Fine-tune BERT model on Reddit politics comments
# This script downloads the politics dataset and fine-tunes the BERT model

import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import logging
from datetime import datetime
from encoder import CachedBERTEncoder

# %%

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
# Setup device for Apple Metal (MPS) if available
def setup_device():

    """Setup the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Metal (MPS) for training")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA for training")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training")
    
    return device

setup_device()

# %%
# %%
# Download and load the Reddit politics dataset
def download_politics_dataset():
    """Download the Reddit politics dataset from Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        csv_filename = "reddit_politics.csv"
        if not os.path.exists(csv_filename):
            logger.info("Downloading Reddit politics dataset from Kaggle...")
            api.dataset_download_files('gpreda/politics-on-reddit', path='.', unzip=True)
            logger.info("Dataset downloaded successfully!")
        else:
            logger.info("Dataset already exists locally.")
            
        return csv_filename
    except ImportError:
        logger.error("Kaggle API not available. Please install kaggle: pip install kaggle")
        return None
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return None

# %%
# Load and preprocess the dataset
def load_and_preprocess_data(csv_filename):
    """Load and preprocess the Reddit politics dataset."""
    logger.info("Loading Reddit politics dataset...")
    
    # Load the dataset
    df = pd.read_csv(csv_filename)
    
    # Display basic info
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Show first few rows
    logger.info("First few rows:")
    print(df.head())
    
    # Check for missing values
    logger.info("Missing values per column:")
    print(df.isnull().sum())
    
    # Clean the data
    # Remove rows with missing comments
    df_clean = df.dropna(subset=['body'])
    
    # Remove very short comments (likely not meaningful)
    df_clean = df_clean[df_clean['body'].str.len() > 10]
    
    # Remove very long comments (to avoid memory issues)
    df_clean = df_clean[df_clean['body'].str.len() < 1000]
    
    # Convert to lowercase for consistency
    df_clean['body'] = df_clean['body'].str.lower()
    
    logger.info(f"Cleaned dataset shape: {df_clean.shape}")
    
    return df_clean

# %%
# Create dataset class for fine-tuning
class PoliticsCommentDataset(Dataset):
    def __init__(self, comments, tokenizer, max_length=512, device=None):
        self.comments = comments
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
    
    def __len__(self):
        return len(self.comments)
    
    def __getitem__(self, idx):
        comment = str(self.comments.iloc[idx])
        
        # Tokenize the comment
        encoding = self.tokenizer(
            comment,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move tensors to device if specified
        if self.device is not None:
            return {
                'input_ids': encoding['input_ids'].flatten().to(self.device),
                'attention_mask': encoding['attention_mask'].flatten().to(self.device),
                'labels': encoding['input_ids'].flatten().to(self.device)  # For masked language modeling
            }
        else:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': encoding['input_ids'].flatten()  # For masked language modeling
            }

# %%
# Create a custom model for fine-tuning
class PoliticsBERTModel(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(PoliticsBERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        
        # Store the model name for proper saving
        self.model_name = model_name
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # For masked language modeling, we use the sequence output
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {"loss": loss, "logits": logits}
    
    def save_pretrained(self, save_directory):
        """Save the model in a format compatible with HuggingFace."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the BERT configuration
        self.bert.config.save_pretrained(save_directory)
        
        # Save the model state dict
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Create a simple config.json for the wrapper model
        import json
        config = {
            "model_type": "bert",
            "base_model": self.model_name,
            "architectures": ["PoliticsBERTModel"]
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

# %%
# Training function
def train_politics_model(model, train_dataset, val_dataset, output_dir, epochs=3, batch_size=80):
    """Train the model on politics comments."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = setup_device()
    model = model.to(device)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        # Add device configuration
        no_cuda=not torch.cuda.is_available(),
        dataloader_num_workers=0,  # Reduce workers for MPS compatibility
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model using our custom save method
    model.save_pretrained(output_dir)
    
    logger.info(f"Model saved to {output_dir}")
    
    return trainer

# %%
# Create output directory
output_dir = "output/politics_finetuned"
os.makedirs(output_dir, exist_ok=True)

# %%

# Download dataset
csv_filename = download_politics_dataset()
if csv_filename is None:
    raise Exception("Failed to download dataset. Exiting.")

# %%
# Load and preprocess data
df = load_and_preprocess_data(csv_filename)
df = df[df['body'].str.len() > 10]

# Limit the dataset size for faster training
max_samples = 10  # You can adjust this number
if len(df) > max_samples:
    df = df.sample(n=max_samples, random_state=42)
    logger.info(f"Limited dataset to {max_samples} samples for faster training")

# Split data into train/validation sets
train_comments, val_comments = train_test_split(
    df['body'], 
    test_size=0.1, 
    random_state=42
)

logger.info(f"Training samples: {len(train_comments)}")
logger.info(f"Validation samples: {len(val_comments)}")

# Initialize tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = PoliticsBERTModel(model_name)

# Setup device
device = setup_device()

# %%

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create datasets with device
train_dataset = PoliticsCommentDataset(train_comments, tokenizer, device=device)
val_dataset = PoliticsCommentDataset(val_comments, tokenizer, device=device)

logger.info(f"Train dataset size: {len(train_dataset)}")
logger.info(f"Validation dataset size: {len(val_dataset)}")

# Train the model
trainer = train_politics_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir=output_dir,
    epochs=3,
    batch_size=8
)

# Save tokenizer
tokenizer.save_pretrained(output_dir)
logger.info(f"Tokenizer saved to {output_dir}")

# %%
# Function to create a cached encoder with the fine-tuned model
def create_finetuned_encoder(output_dir):
    """Create a cached encoder using the fine-tuned model."""
    from transformers import AutoModel, AutoTokenizer, BertConfig
    
    try:
        # Try to load the fine-tuned model and tokenizer
        model = AutoModel.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
    except Exception as e:
        logger.warning(f"Could not load model with AutoModel: {e}")
        # Fallback: load the original BERT model and load our fine-tuned weights
        logger.info("Loading original BERT model and fine-tuned weights...")
        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Load the fine-tuned weights
        if os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
            state_dict = torch.load(os.path.join(output_dir, "pytorch_model.bin"), map_location='cpu')
            # Load only the BERT weights (skip the classifier)
            bert_state_dict = {k: v for k, v in state_dict.items() if k.startswith('bert.')}
            model.load_state_dict(bert_state_dict, strict=False)
            logger.info("Loaded fine-tuned BERT weights successfully")
    
    # Move model to device
    device = setup_device()
    model = model.to(device)
    model.eval()
    
    # Create a custom encoder that uses the fine-tuned model
    class FinetunedBERTEncoder(CachedBERTEncoder):
        def __init__(self, model, tokenizer, cache_path, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.cache = CachedBERTEncoder("bert-base-uncased", cache_path).cache
        
        def encode(self, text: str, use_cache: bool = True) -> torch.Tensor:
            if use_cache:
                cached_embedding = self.cache.get(text, "finetuned_bert")
                if cached_embedding is not None:
                    return cached_embedding
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
            
            if use_cache:
                self.cache.set(text, embedding, "finetuned_bert")
            
            return embedding
    
    finetuned_encoder = FinetunedBERTEncoder(model, tokenizer, f"{output_dir}/embeddings_cache.db", device)
    return finetuned_encoder

# Create a cached encoder with the fine-tuned model
logger.info("Creating cached encoder with fine-tuned model...")
cached_encoder = create_finetuned_encoder(output_dir)

# Test the fine-tuned model
test_comments = [
    "The president's new policy on healthcare is controversial.",
    "I think the economy is doing well under this administration.",
    "The opposition party has a strong case for their platform."
]

logger.info("Testing fine-tuned model on sample comments:")
for comment in test_comments:
    embedding = cached_encoder.encode(comment)
    logger.info(f"Comment: {comment}")
    logger.info(f"Embedding shape: {embedding.shape}")
    logger.info(f"Embedding norm: {torch.norm(embedding):.4f}")
    logger.info("---")

logger.info("Fine-tuning completed successfully!")
logger.info(f"Model and tokenizer saved to: {output_dir}")


# %%
# Optional: Test the fine-tuned model on new comments
# Uncomment the line below to run the test
# test_finetuned_model()


# %%
# Function to test the fine-tuned model
def test_finetuned_model():
    """Test the fine-tuned model on new politics comments."""
    
    output_dir = "output/politics_finetuned"
    
    if not os.path.exists(output_dir):
        raise Exception("Fine-tuned model not found. Please run the training first.")
    
    # Load the fine-tuned model
    cached_encoder = create_finetuned_encoder(output_dir)
    
    # Test comments
    test_comments = [
        "The election results show a clear mandate for change.",
        "This policy will have far-reaching consequences for the economy.",
        "The bipartisan bill addresses key concerns from both parties.",
        "The Supreme Court's decision sets an important precedent.",
        "Local elections often have the biggest impact on daily life."
    ]
    
    logger.info("Testing fine-tuned model on new politics comments:")
    
    embeddings = []
    for comment in test_comments:
        embedding = cached_encoder.encode(comment)
        embeddings.append(embedding.squeeze().detach().cpu().numpy())
        logger.info(f"Comment: {comment}")
        logger.info(f"Embedding norm: {torch.norm(embedding):.4f}")
    
    # Calculate similarities between embeddings
    embeddings_array = np.array(embeddings)
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings_array)
    
    logger.info("\nSimilarity matrix between test comments:")
    print(similarities)

# Uncomment to test the fine-tuned model
test_finetuned_model()

# %%
