import pandas as pd
import os
import re

os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')
csv_address_file = '1_Data Collection/contracts.csv'
def source_code_integration(filename):
    with open(filename) as file:
        code = file.read()
    # Remove multi-line comments (/* ... */)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    # Remove single-line comments (// ...)
    code = re.sub(r'//.*', '', code)

    # Remove newline characters and excessive whitespace
    code = re.sub(r'\n', ' ', code)
    code = re.sub(r'\s+', ' ', code).strip()

    # Regex pattern to match 'pragma solidity ^version;'
    code = re.sub(r'pragma solidity\s+\^?[0-9]+\.[0-9]+\.[0-9 ]+;', '', code)

    # Return the processed code
    return code

def operation_code_integration(filename):
    with open(filename) as file:
        code = file.read()
    code = code.split("<br>")
    processed_code = code.copy()
    for i in range(len(code)):
        if processed_code[i].__contains__("Unknown"):
            processed_code[i] = "UNK"
        elif len(processed_code[i].split(" ")) >=2:
            processed_code[i] = processed_code[i].split(" ")[0]
        processed_code[i] = re.sub(r'\d+','',processed_code[i])
    new_data = ' '.join(processed_code)
    return new_data


if __name__ == '__main__':
    df = pd.read_csv(csv_address_file)
    hashes = df["address"].tolist()
    processed_sc_data = []
    processed_op_data = []
    try:
        for i in range(len(hashes)):
            processed_sc = source_code_integration("1_Data Collection/Source_code/sol/"+hashes[i] + ".sol")
            processed_op = operation_code_integration("1_Data Collection/Operation_code/sol/" + hashes[i] + ".sol")
            processed_sc_data.append({"address": hashes[i], "code": processed_sc})
            processed_op_data.append({"address": hashes[i], "code": processed_op})
    except:
        print(hashes[i]+" cannot")

    # Write the processed data to a new CSV file
    processed_df = pd.DataFrame(processed_sc_data)
    processed_df2 = pd.DataFrame(processed_op_data)
    processed_df.to_csv('2_Data Preprocessing/source_code.csv', index=False)
    processed_df2.to_csv('2_Data Preprocessing/operation_code.csv', index=False)
    print("Processed data saved to processed_contracts.csv")
