import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import ADASYN
import numpy as np

os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')
def read_vulnerability_file():
    vulnerabilities = {}
    with open('Data Collection/scrawld_majority_unique.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            address = parts[0].split('_ext.sol')[0]
            vuln_types = parts[1:]
            vulnerabilities[address] = vuln_types
    return vulnerabilities

def read_processed_file():
    csv_path = 'Data Preprocessing/processed_contracts.csv'
    df = pd.read_csv(csv_path)

    # Add a column for vulnerability types if it doesn't exist
    if 'Vulnerability Types' not in df.columns:
        df['Vulnerability Types'] = 'None'
    return df

def save_labelled_file(vulnerabilities,df):
    expanded_rows = []

    for index, row in df.iterrows():
        address = row['address']  # Assuming 'address' is the column name in the CSV

        if address in vulnerabilities:
            vuln_types = vulnerabilities[address]

            # If there are multiple vulnerability types, create separate rows
            for vuln_type in vuln_types:
                new_row = row.copy()
                new_row['Vulnerability Types'] = vuln_type
                expanded_rows.append(new_row)
        else:
            # If no vulnerabilities found, keep the row as is
            expanded_rows.append(row)

    # Create a new DataFrame with the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)

    # Optionally, save the expanded DataFrame to a new CSV file
    expanded_df.to_csv('Data Preprocessing/labelled_contracts.csv', index=False)
    return expanded_df

if __name__ == '__main__':
    mlb = MultiLabelBinarizer()
    expanded_df = save_labelled_file(read_vulnerability_file(), read_processed_file())

    # Group by 'Address' and aggregate the 'Vulnerability Type' into lists
    df_grouped = expanded_df.groupby('address')['Vulnerability Types'].apply(list).reset_index()

    # Convert vulnerabilities to a multi-label format using MultiLabelBinarizer
    y_multi_label = mlb.fit_transform(df_grouped['Vulnerability Types'])

    # Prepare your feature matrix 'X' - here using the index as a placeholder
    X = np.array(df_grouped.index).reshape(-1, 1)  # Replace with actual features if necessary

    # Initialize list to collect resampled data
    X_resampled_list = [X]
    y_resampled_list = [y_multi_label]

    # Loop through each label column (each vulnerability type)
    for i in range(y_multi_label.shape[1]):
        # Extract the binary labels for this vulnerability type
        y_binary = y_multi_label[:, i]

        # If there's only one class present, skip ADASYN for this label
        if len(np.unique(y_binary)) == 1:
            continue

        # Calculate the number of samples in the minority class
        n_minority_samples = sum(y_binary == 1)

        # Set n_neighbors to be less than or equal to the number of minority samples
        n_neighbors = min(5, n_minority_samples - 1)

        # Apply ADASYN to the binary label column
        ada = ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=n_neighbors)
        X_res, y_res = ada.fit_resample(X, y_binary)

        # Update resampled X and y lists
        X_resampled_list.append(X_res)

        # Convert resampled y_res into a 2D column vector and stack with y_multi_label
        y_resampled_col = np.vstack((y_multi_label[:, i].reshape(-1, 1), y_res.reshape(-1, 1)))

        # Update y_resampled_list with aligned columns
        y_resampled_list.append(y_resampled_col)

    # Combine all resampled feature arrays into one
    X_resampled_combined = np.vstack(X_resampled_list)

    # Combine all resampled label arrays horizontally to form a multi-label format
    y_resampled_combined = np.hstack(y_resampled_list)

    # Convert indices back to addresses and build the final resampled DataFrame
    df_resampled = pd.DataFrame({
        'Address': df_grouped.loc[X_resampled_combined.flatten()]['address'],
        'Vulnerability Type': mlb.inverse_transform(y_resampled_combined)
    })

    # Check the resampled DataFrame
    print(df_resampled.head())

    # Save to CSV if needed
    df_resampled.to_csv('resampled_data.csv', index=False)