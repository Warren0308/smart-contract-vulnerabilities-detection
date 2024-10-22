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


# Function to split source code into chunks of up to 510 tokens
def split_into_chunks(source_code, max_length=510):
    tokens = tokenizer.tokenize(str(source_code))  # Ensure source_code is a string
    # Split into chunks of up to max_token_length (510 in this case to leave space for [CLS] and [SEP])
    token_chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return token_chunks


# Function to pad each chunk to a fixed length of 510 tokens and create attention masks
def pad_chunk(tokens_ids, max_length=510):
    # If the chunk is shorter than 510 tokens, pad with 0
    padding_length = max_length - len(tokens_ids)
    if padding_length > 0:
        tokens_ids += [0] * padding_length  # Pad with 0s

    # Create attention mask: 1 for real tokens, 0 for padding
    attention_mask = [1] * (max_length - padding_length) + [0] * padding_length
    return tokens_ids, attention_mask


# Function to get CodeBERT embeddings for each chunk and average them
def get_codebert_embedding(source_code):
    chunks = split_into_chunks(source_code)
    embeddings_list = []
    max_length = 510
    tokenizer.add_prefix_space = True
    for chunk in chunks:
        if len(chunk) != max_length:
            chunk = chunk + [tokenizer.pad_token] * (max_length - len(chunk))
        # Add special tokens ([CLS] at the start and [EOS] at the end)
        chunk = [tokenizer.cls_token] + chunk + [tokenizer.eos_token]
        # Convert tokens to token IDs
        tokens_ids = tokenizer.convert_tokens_to_ids(chunk)
        # Pad the chunk to a fixed length of 510 (including padding if needed)
        tokens_ids, attention_mask = pad_chunk(tokens_ids, max_length=512)  # 512 to account for [CLS] and [EOS] tokens

        # Convert inputs to tensors
        tokens_tensor = torch.tensor([tokens_ids])  # Shape: (1, sequence_length)
        attention_mask_tensor = torch.tensor([attention_mask])  # Shape: (1, sequence_length)

        with torch.no_grad():
            outputs = model(input_ids=tokens_tensor, attention_mask=attention_mask_tensor)

        # Take the mean of token embeddings for this chunk
        chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings_list.append(chunk_embedding)

    # Aggregate embeddings (mean) to form a single representation
    if len(embeddings_list) > 0:
        print("Completed chunk processing!")
        return np.mean(embeddings_list, axis=0)
    else:
        print("Failure in chunk processing!")
        return np.zeros((768,))  # 768 is the dimension of CodeBERT embeddings


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
df.to_csv('3_Data Encoding/Source_Code_Embedding.csv',index=False)

# Alternatively, save the embeddings as a NumPy file for better preservation of array structures
np.save('3_Data Encoding/Source_Code_Embedding.npy', df['CodeBERT_Embedding'].values)

print("CodeBERT embeddings for full source code have been successfully generated and saved.")
