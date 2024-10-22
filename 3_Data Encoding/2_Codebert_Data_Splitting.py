import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os
import numpy as np
import ast
# Set your working directory
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')
# Load the new CSV file
df = pd.read_csv('3_Data Encoding/Source_Code_Embedding.csv')
# Convert 'vulnerability_types' from string to list
df['vulnerability_type'] = df['vulnerability_type'].apply(eval)

# Convert the multi-labels into a binary matrix using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['vulnerability_type'])  # Multi-hot encoded labels

# Convert the multi-hot encoded matrix back to a DataFrame with the correct column names
y_df = pd.DataFrame(y, columns=mlb.classes_)

# Convert string representation of lists back to actual lists
df['CodeBERT_Embedding'] = df['CodeBERT_Embedding'].apply(ast.literal_eval)

# Now you can create the DataFrame with the correct shape
embedding_df = pd.DataFrame(df['CodeBERT_Embedding'].tolist(), columns=[f"embedding_{i+1}" for i in range(768)])

# Check the shape of the new embedding_df
print(embedding_df.shape)

# Concatenate the original features (code) with the newly created multi-label columns
df_split = pd.concat([df['address'],embedding_df, y_df], axis=1)

# Select columns from 'embedding_1' to 'embedding_768' for X
X = df_split.loc[:, [f'embedding_{i+1}' for i in range(768)]]  # You can also use embeddings or any other feature set

# Create MultilabelStratifiedKFold instance
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Use only the first split
train_index, val_index = next(mskf.split(X, y_df))

# Create train and validation datasets
X_train, X_val = X.iloc[train_index], X.iloc[val_index]
y_train, y_val = y[train_index], y[val_index]

# Convert back to DataFrames if needed
train_df = df_split.iloc[train_index]
val_df = df_split.iloc[val_index]

# Save the train and validation splits if required
train_df.to_csv('3_Data Encoding/SC_train.csv', index=False)
val_df.to_csv('3_Data Encoding/SC_test.csv', index=False)


