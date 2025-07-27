# %%

# download https://www.kaggle.com/datasets/gpreda/politics-on-reddit dataset 

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import os

# %%

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# %%





csv_filename = "reddit_politics.csv"
if not os.path.exists(csv_filename):
    api.dataset_download_files('gpreda/politics-on-reddit', path='.', unzip=True)

# Load the dataset CSV file
df = pd.read_csv(csv_filename)

# # Optionally, display the first few rows to verify loading
# print(df.head())


# %%

df.head()
# %%

df.columns
# %%

