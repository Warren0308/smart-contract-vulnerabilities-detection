import pandas as pd
from imblearn.over_sampling import ADASYN


import os
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')


# Load your dataset (assuming you have columns 'Address', 'Source Code', and 'Vulnerability Type')
df = pd.read_csv('2_Data Preprocessing/source_code_with_vulnerability.csv')
# Check the first few rows of your DataFrame
print(df.head())
df['vulnerability_type'] = df['vulnerability_type'].fillna('None')
# Extract feature (source code) and target (vulnerability type) columns
X = df['code'].values.reshape(-1, 1)  # Source code needs to be reshaped to a 2D array for ADASYN
y = df['vulnerability_type']  # This is your target variable, representing the vulnerability type
print(len(X))
print(X)
print(len(y))
print(y.unique())  # Check the unique values and their data types in y
print(y.dtype)  # Display the class distribution before resampling
print("Class distribution before resampling:\n", y.value_counts())

# Apply ADASYN to balance the dataset
adasyn = ADASYN(random_state=42, n_neighbors=5)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Convert resampled data back to a DataFrame
df_resampled = pd.DataFrame({
    'Source Code': X_resampled.flatten(),
    'Vulnerability Type': y_resampled
})

# Display the first few rows of the resampled DataFrame
print("\nResampled DataFrame:\n", df_resampled.head())

# Check the class distribution after resampling
print("\nClass distribution after resampling:\n", df_resampled['Vulnerability Type'].value_counts())

# Save the resampled data to a new CSV file if needed
df_resampled.to_csv('2_Data Preprocessing/resampled_source_code.csv', index=False)