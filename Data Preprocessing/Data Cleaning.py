import pandas as pd
import os
import re

os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')
csv_address_file = 'Data Collection/contracts.csv'
def CommentRemover(filename):
    with open(filename) as file:
        code = file.read()
    # Remove multi-line comments (/* ... */)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    # Remove single-line comments (// ...)
    code = re.sub(r'//.*', '', code)

    # Remove newline characters and excessive whitespace
    code = re.sub(r'\n', ' ', code)
    code = re.sub(r'\s+', ' ', code).strip()

    # Return the processed code
    return code



if __name__ == '__main__':
    df = pd.read_csv(csv_address_file)
    hashes = df["address"].tolist()
    processed_data = []
    try:
        for i in range(len(hashes)):
            processed_code = CommentRemover("Data Collection/original/"+hashes[i] + "_ext.sol")
            processed_data.append({"address": hashes[i], "code": processed_code})
    except:
        print(hashes[i]+" cannot")

    # Write the processed data to a new CSV file
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv('Data Preprocessing/processed_contracts.csv', index=False)
    print("Processed data saved to processed_contracts.csv")
