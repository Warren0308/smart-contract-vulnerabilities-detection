import pandas as pd
import os
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')
address = "1_Data Collection/contracts.csv"
vulnerability_file ="1_Data Collection/scrawld_majority_unique.txt"
source_code = '2_Data Preprocessing/source_code_with_vulnerability.csv'
address_count = pd.read_csv(address)
print("Total Address Number: ", len(address_count))
vulnerability = pd.read_csv(vulnerability_file,header=None)
print("Total Vulnerable Address Number: ",len(vulnerability))

source_code_csv = pd.read_csv(source_code)
vulnerability_count = source_code_csv['vulnerability_type'].value_counts(dropna=False)
print(vulnerability_count)
print("Total Address in source code csv: ", len(source_code_csv['address'].unique()))
df_without_vulnerability = source_code_csv[source_code_csv['vulnerability_type']=="['None']"]
print("Total Unvulnerable Address Number: ",len(df_without_vulnerability))
df_without_vulnerability = source_code_csv[source_code_csv['vulnerability_type']=="['None']"]['address'].unique()
print("Total Vulnerable Address Number: ",len(df_without_vulnerability))