import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import  os
# Set your working directory
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')

# Function to generate sliding n-grams (only 2-grams and 3-grams)
def generate_sliding_ngrams(opcode_sequence, n_sizes=[2, 3]):
    tokens = opcode_sequence.split()
    ngram_sequences = []
    # Generate n-grams for each n-size (only 2 and 3 here) and combine them
    for n in n_sizes:
        ngram_sequences.extend([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
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

# Load your cleaned operation code data
df = pd.read_csv('2_Data Preprocessing/operation_code_with_vulnerability.csv')

# Check for NaN or empty values in the 'code' column and handle them
df['code'] = df['code'].fillna('')  # Replace NaN with an empty string

# Apply the sliding window n-gram generation to the opcode column
df['ngrams'] = df['code'].apply(lambda x: cap_ngrams(generate_sliding_ngrams(x, n_sizes=[2, 3]), max_length=100))

# Train Word2Vec on the capped n-grams (as tokenized sequences)
word2vec_model = Word2Vec(sentences=df['ngrams'], vector_size=100, min_count=1)  # Set vector_size=100 for each n-gram

# Save the model if needed
word2vec_model.save('Operation_Code_Embedding.model')

# Apply the function to get Word2Vec embeddings for the n-grams (no averaging)
df['Word2vec_Embeddings'] = df['ngrams'].apply(lambda x: get_word2vec_embeddings(x, word2vec_model))

# Flatten the embeddings into 300 columns (100 per n-gram size, total of 300)
def flatten_embeddings(embedding_sequence):
    return embedding_sequence.flatten()[:200]  # Make sure the sequence fits into 300 columns

# Apply flattening to the embeddings and create a DataFrame for the final features
df['flattened_embeddings'] = df['Word2vec_Embeddings'].apply(lambda x: flatten_embeddings(x))

# Convert the embeddings into a DataFrame of shape (n_samples, 300)
embedding_df = pd.DataFrame(df['flattened_embeddings'].tolist(), columns=[f"embedding_{i+1}" for i in range(300)])

# Concatenate the original features with the new embedding columns
df_final = pd.concat([df[['address', 'code', 'vulnerability_type']], embedding_df], axis=1)

# Optionally, save the embeddings to a CSV
df_final.to_csv('3_Data Encoding/Operation_Code_Embedding.csv', index=False)
