import pandas as pd
import os


source_code = '2_Data Preprocessing/source_code.csv'
operation_code = '2_Data Preprocessing/operation_code.csv'
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')

# List of target vulnerabilities you want to include
target_vulnerabilities = ['LE', 'RENT', 'ARTHM', 'TimeO']
def dict_vulnerability():
    vulnerabilities = {}
    with open('1_Data Collection/scrawld_majority_unique.txt', 'r') as file:  # Replace with your actual TXT file path
        for line in file:
            parts = line.strip().split(' ')
            address = parts[0].split("_ext")[0]
            vul_types = parts[1:]
            vulnerabilities[address] = vul_types
    return vulnerabilities

def combine_vulnerability(df, vulnerabilities):
    rows = []
    for index, row in df.iterrows():
        address = row['address']
        code = row['code']

        if address in vulnerabilities:
            # Filter only the vulnerabilities that are in the target_vulnerabilities list
            filtered_vul_types = [vul for vul in vulnerabilities[address] if vul in target_vulnerabilities]

            # If there are matching target vulnerabilities, add them to the rows
            if filtered_vul_types:
                rows.append({'address': address, 'code': code, 'vulnerability_type': filtered_vul_types})
            else:
                rows.append({'address': address, 'code': code, 'vulnerability_type': ['None']})
        else:
            rows.append(
                {'address': address, 'code': code, 'vulnerability_type': ['None']})
            print(address, " doesnt exist any vulnerability")# If no vulnerability is found
    return rows

if __name__ == '__main__':
    df_sc = pd.read_csv(source_code)
    df_oc = pd.read_csv(operation_code)
    vulnerabilities = dict_vulnerability()
    # Create the final DataFrame
    sc_combined = pd.DataFrame(combine_vulnerability(df_sc, vulnerabilities))
    oc_combined = pd.DataFrame(combine_vulnerability(df_oc, vulnerabilities))
    # Save to a new CSV file
    sc_combined.to_csv('2_Data Preprocessing/source_code_with_vulnerability.csv', index=False)
    oc_combined.to_csv('2_Data Preprocessing/operation_code_with_vulnerability.csv', index=False)

