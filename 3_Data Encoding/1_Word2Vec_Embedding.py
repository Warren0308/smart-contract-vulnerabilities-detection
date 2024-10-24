import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import os

# Set your working directory
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')

# Function to generate sliding 2-grams only
def generate_2grams(opcode_sequence):
    tokens = opcode_sequence.split()
    ngram_sequences = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]  # Only generate 2-grams
    return ngram_sequences

# Function to cap the n-gram sequences to a maximum length (e.g., 100)
def cap_ngrams(ngram_sequences, max_length=100):
    return ngram_sequences[:max_length]  # Truncate the sequence if it's too long

# Function to get Word2Vec embeddings for each n-gram sequence
def get_word2vec_embeddings(ngram_sequences, model):
    embedding_sequence = np.zeros((len(ngram_sequences), model.vector_size))  # No averaging, keep sequence
    for i, ngram in enumerate(ngram_sequences):
        if ngram in model.wv:
            embedding_sequence[i] = model.wv[ngram]
    return embedding_sequence

# Function to apply max-pooling across token embeddings (along the token axis)
def max_pool_embeddings(embedding_sequence):
    # Max pooling across the token axis (take max for each dimension across all tokens)
    return np.max(embedding_sequence, axis=0)

# Load your cleaned operation code data
df = pd.read_csv('2_Data Preprocessing/operation_code_with_vulnerability.csv')

# Check for NaN or empty values in the 'code' column and handle them
df['code'] = df['code'].fillna('')  # Replace NaN with an empty string

# Apply the sliding window 2-gram generation to the opcode column
df['ngrams'] = df['code'].apply(lambda x: cap_ngrams(generate_2grams(x), max_length=100))

# Train Word2Vec on the capped 2-grams with 300 dimensions
word2vec_model = Word2Vec(sentences=df['ngrams'], vector_size=300, min_count=1)  # Set vector_size=300 for each 2-gram

# Save the model if needed
word2vec_model.save('Operation_Code_Embedding_300.model')

# Apply the function to get Word2Vec embeddings for the n-grams and then max-pool them
df['Word2vec_Embeddings'] = df['ngrams'].apply(lambda x: max_pool_embeddings(get_word2vec_embeddings(x, word2vec_model)))

# Ensure that embeddings are 300-dimensional (if necessary, pad or truncate)
def ensure_dimension(embedding_sequence, target_dim=300):
    # Truncate or pad the sequence to fit exactly 300 dimensions
    if len(embedding_sequence) > target_dim:
        return embedding_sequence[:target_dim]  # Truncate if too long
    elif len(embedding_sequence) < target_dim:
        # Pad with zeros if too short
        return np.pad(embedding_sequence, (0, target_dim - len(embedding_sequence)), 'constant')
    return embedding_sequence

# Ensure embeddings are the right size (300 dimensions)
df['final_embeddings'] = df['Word2vec_Embeddings'].apply(lambda x: ensure_dimension(x, target_dim=300))

# Convert the embeddings into a DataFrame of shape (n_samples, 300)
embedding_df = pd.DataFrame(df['final_embeddings'].tolist(), columns=[f"embedding_{i+1}" for i in range(300)])

# Concatenate the original features with the new embedding columns
df_final = pd.concat([df[['address', 'code', 'vulnerability_type']], embedding_df], axis=1)

# Optionally, save the embeddings to a CSV
df_final.to_csv('3_Data Encoding/Operation_Code_Embedding_300.csv', index=False)
