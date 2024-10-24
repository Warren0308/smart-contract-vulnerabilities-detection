import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


# Set your working directory
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')


# Load the already split source code datasets
source_code_train = pd.read_csv('3_Data Encoding/SC_train.csv')
source_code_test = pd.read_csv('3_Data Encoding/SC_test.csv')

# Load the operation code dataset
operation_code_df = pd.read_csv('3_Data Encoding/Operation_Code_Embedding_300.csv')
# Convert 'vulnerability_types' from string to list
operation_code_df['vulnerability_type'] = operation_code_df['vulnerability_type'].apply(eval)

# Convert the multi-labels into a binary matrix using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(operation_code_df['vulnerability_type'])  # Multi-hot encoded labels

# Convert the multi-hot encoded matrix back to a DataFrame with the correct column names
y_df = pd.DataFrame(y, columns=mlb.classes_)
# Get the addresses from the source code train and test sets
# Concatenate the original features (code) with the newly created multi-label columns
operation_code_df = pd.concat([operation_code_df, y_df], axis=1)
operation_code_df.drop(['code', 'vulnerability_type'], axis=1)
train_addresses = source_code_train['address']
test_addresses = source_code_test['address']

# Filter the operation code data based on the train and test addresses
operation_code_train = operation_code_df[operation_code_df['address'].isin(train_addresses)]
operation_code_test = operation_code_df[operation_code_df['address'].isin(test_addresses)]

# Save the filtered operation code train and test sets
operation_code_train.to_csv('3_Data Encoding/OP_train.csv', index=False)
operation_code_test.to_csv('3_Data Encoding/OP_test.csv', index=False)
