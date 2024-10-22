import pandas as pd
import os

# Set your working directory
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')
# Load your cleaned operation code data
df = pd.read_csv('3_Data Encoding/Operation_Code_Embedding.csv')

print(len(df['Word2vec_Embedding'][0]))