import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os

# Load CodeBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# Set your working directory
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')


# Function to split source code into chunks of up to 510 tokens, using sliding window
def split_into_chunks(source_code, max_length=510, overlap=256):
    tokens = tokenizer.tokenize(str(source_code))  # Ensure source_code is a string
    token_chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        token_chunks.append(chunk)
    return token_chunks


# Function to pad each chunk to a fixed length of 510 tokens and create attention masks
def pad_chunk(tokens_ids, max_length=510):
    padding_length = max_length - len(tokens_ids)
    if padding_length > 0:
        tokens_ids += [tokenizer.pad_token_id] * padding_length  # Pad with tokenizer's pad token ID
    attention_mask = [1] * (max_length - padding_length) + [0] * padding_length
    return tokens_ids, attention_mask


# Function to get CodeBERT embeddings for each chunk and apply max-pooling
def get_codebert_embedding(source_code):
    chunks = split_into_chunks(source_code)
    embeddings_list = []
    max_length = 510  # Max token length for chunks
    tokenizer.add_prefix_space = True

    for chunk in chunks:
        # Add special tokens ([CLS] at the start and [SEP] at the end)
        chunk = [tokenizer.cls_token] + chunk + [tokenizer.sep_token]
        tokens_ids = tokenizer.convert_tokens_to_ids(chunk)
        tokens_ids, attention_mask = pad_chunk(tokens_ids, max_length=512)  # 512 accounts for [CLS] and [SEP]

        tokens_tensor = torch.tensor([tokens_ids])  # Shape: (1, sequence_length)
        attention_mask_tensor = torch.tensor([attention_mask])  # Shape: (1, sequence_length)

        with torch.no_grad():
            outputs = model(input_ids=tokens_tensor, attention_mask=attention_mask_tensor)

        # Apply max-pooling across the token dimension for this chunk
        chunk_embedding, _ = torch.max(outputs.last_hidden_state, dim=1)  # Max-pooling over token dimension
        embeddings_list.append(chunk_embedding.squeeze().numpy())

    # Aggregate embeddings across all chunks (e.g., by averaging or max-pooling across chunks)
    if len(embeddings_list) > 0:
        # Option 1: Average across chunk embeddings
        # final_embedding = np.mean(embeddings_list, axis=0)

        # Option 2: Max-pool across chunk embeddings
        final_embedding = np.max(embeddings_list, axis=0)

        print("Completed chunk processing!")
        return final_embedding
    else:
        print("Failure in chunk processing!")
        return np.zeros((768,))  # Return zero vector if no chunks are processed


# Load your dataset
df = pd.read_csv('2_Data Preprocessing/source_code_with_vulnerability.csv')  # Replace with your actual dataset path

# Check for NaN or empty values in the 'code' column and handle them
df['code'] = df['code'].fillna('')  # Replace NaN with an empty string

# Initialize a counter for completed files
completed_files = 0

# Apply CodeBERT embedding function to each row and store it as a list
df['CodeBERT_Embedding'] = df['code'].apply(lambda x: get_codebert_embedding(x).tolist())

# Update the completed files counter and print after processing each row
for index, row in df.iterrows():
    completed_files += 1
    print(f"Completed processing {completed_files} files.")

# Save the embeddings to a CSV in a flattened format
df.to_csv('3_Data Encoding/Source_Code_Embedding.csv', index=False)

# Alternatively, save the embeddings as a NumPy file for better preservation of array structures
np.save('3_Data Encoding/Source_Code_Embedding.npy', df['CodeBERT_Embedding'].values)

print("CodeBERT embeddings for full source code have been successfully generated and saved.")
