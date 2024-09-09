import re
import pandas as pd
from solcx import compile_source, install_solc, set_solc_version, get_installed_solc_versions
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')

def get_solc_version_from_source(source_code):
    """
    Extract the required Solidity compiler version from the source code pragma directive.
    """
    match = re.search(r'pragma solidity \^?([0-9]+\.[0-9]+\.[0-9]+);', source_code)
    print(match)
    if match:
        return match.group(1)
    else:
        raise ValueError("Solidity version pragma not found in source code.")


def compile_contract_to_bytecode(source_code):
    """
    Compiles Solidity source code to bytecode using the appropriate Solidity compiler version.
    """
    try:
        # Extract the required compiler version from the source code
        solc_version = get_solc_version_from_source(source_code)

        # Check if the required version is already installed, if not, install it
        if solc_version not in get_installed_solc_versions():
            print(f"Installing Solidity compiler version {solc_version}...")
            install_solc(solc_version)

        # Set the correct compiler version
        set_solc_version(solc_version)

        # Compile the Solidity source code to bytecode
        compiled_sol = compile_source(source_code)
        contract_interface = compiled_sol[next(iter(compiled_sol))]
        bytecode = contract_interface['bin']
        return bytecode

    except Exception as e:
        print(f"Error compiling contract: {e}")
        return None


def bytecode_to_opcodes(bytecode):
    """
    Converts bytecode to operation codes (opcodes).
    """
    if bytecode:
        # Split bytecode into chunks of 2 characters (1 byte)
        opcodes = [bytecode[i:i + 2] for i in range(0, len(bytecode), 2)]
        return opcodes
    else:
        return []


def extract_opcodes_from_source(source_code):
    """
    Extracts opcodes from Solidity source code by compiling it to bytecode.
    """
    # Compile source code to bytecode
    bytecode = compile_contract_to_bytecode(source_code)

    # Extract opcodes from bytecode
    opcodes = bytecode_to_opcodes(bytecode)

    return opcodes


def process_contract(row):
    """
    Processes a single contract to compile and extract opcodes.
    """
    source_code = row['code']
    try:
        opcodes = extract_opcodes_from_source(source_code)
        return opcodes
    except Exception as e:
        print(f"Error processing contract at address {row['Address']}: {e}")
        return None


if __name__ == '__main__':
    # Load the CSV file containing smart contracts
    df = pd.read_csv('Data Preprocessing/processed_contracts.csv')

    # Ensure the DataFrame has the correct columns
    if 'code' not in df.columns:
        raise ValueError("The DataFrame must have a 'original' column containing Solidity code.")

    # Use ThreadPoolExecutor to parallelize processing of contracts
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Map process_contract function to each row in the DataFrame
        futures = {executor.submit(process_contract, row): row for _, row in df.iterrows()}

        # Collect results as they complete
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Add operation codes to the DataFrame
    df['Operation Codes'] = results

    # Display the DataFrame with operation codes
    print(df[['address', 'Operation Codes']])

    # Save the updated DataFrame to a new CSV file
    df.to_csv('Data Preprocessing/processed_contracts_with_opcodes.csv', index=False)
