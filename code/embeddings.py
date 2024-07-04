import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import transformers
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

# Enable tqdm for pandas
tqdm.pandas()

# Check CUDA availability and set device
print("Cuda is available: ", torch.cuda.is_available())
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model
tokenizer = transformers.LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
model = transformers.LongformerModel.from_pretrained("allenai/longformer-base-4096")
model.to(DEV)

def embed_sequence(text):
    encoding = tokenizer.encode_plus(text, return_tensors="pt", max_length=4096)
    encoding = encoding.to(DEV)
    global_attention_mask = torch.tensor([([0]*encoding["input_ids"].shape[-1])])  # all zeros for local attention
    global_attention_mask = global_attention_mask.to(DEV)
    encoding["global_attention_mask"] = global_attention_mask
    
    with torch.no_grad():
        o = model(**encoding)
    
    sentence_embedding = o.last_hidden_state[:,0]
    return sentence_embedding

# Load new dataset
data_df = pd.read_csv('../data/combined_hate_ds.csv', lineterminator='\n')

# Debugging: Print column names
print("CSV Columns:", data_df.columns)

# Check if 'body\r' column exists
if 'body\r' in data_df.columns:
    # Rename 'body\r' to 'body'
    data_df.rename(columns={'body\r': 'body'}, inplace=True)

# Check if 'body' column exists
if 'body' not in data_df.columns:
    raise KeyError("'body' column is missing in the CSV file. Available columns are:", data_df.columns)

unk_embedding = embed_sequence('UNK')  # Placeholder embedding for missing text

# Initialize tensor to store embeddings
text_embeddings = torch.empty(size=(len(data_df), 768))

# Embed text
for i, text in enumerate(tqdm(data_df['body'].to_list(), desc="Embedding texts")):
    if pd.notna(text):
        text_embeddings[i] = embed_sequence(text)
    else:
        text_embeddings[i] = unk_embedding

# Convert target labels to tensor
target_labels = torch.tensor(data_df['hate'].to_numpy(), dtype=torch.long)

# Save embeddings and labels
torch.save(text_embeddings, '../data/body_embeddings.pt')
torch.save(target_labels, '../data/hate_labels.pt')
