import pandas as pd
from solcx import compile_source, install_solc, set_solc_version
import re
import os

os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')


def extract_version_from_pragma(source_code):
    """Extract the required Solidity compiler version from the pragma directive."""
    match = re.search(r'pragma solidity \^([\d.]+);', source_code)
    if match:
        version = match.group(1)
        return version
    return None


def compile_contract_to_bytecode(source_code, version):
    """Compile Solidity source code to bytecode using a specific compiler version."""
    compiler_version = f'0.4.{version.split(".")[1]}'  # Adjust as needed
    if float(version.split(".")[0] + "." + version.split(".")[1]) < 0.4:
        raise ValueError(f"Compiler version {compiler_version} is not supported by py-solc-x.")

    if float(version.split(".")[0] + "." + version.split(".")[1]) < 0.11:
        raise ValueError(f"Solidity version ^{version} is not supported by py-solc-x.")

    try:
        install_solc(compiler_version)
    except Exception as e:
        print(f"Error installing compiler version {compiler_version}: {e}")
        raise

    set_solc_version(compiler_version)

    compiled_sol = compile_source(source_code)
    contract_interface = compiled_sol[next(iter(compiled_sol))]
    bytecode = contract_interface['bin']
    return bytecode


def bytecode_to_opcodes(bytecode):
    """Convert bytecode to operation codes (opcodes)."""
    opcodes = [bytecode[i:i + 2] for i in range(0, len(bytecode), 2)]
    return opcodes


def extract_opcodes_from_source(source_code):
    """Compile source code to bytecode and extract opcodes based on the required compiler version."""
    version = extract_version_from_pragma(source_code)
    if not version:
        raise ValueError("Unable to determine Solidity compiler version from source code.")

    major_minor_version = float(version.split(".")[0] + "." + version.split(".")[1])
    if major_minor_version < 0.4:
        raise ValueError(f"Solidity version ^{version} is not supported.")

    bytecode = compile_contract_to_bytecode(source_code, version)
    opcodes = bytecode_to_opcodes(bytecode)
    return opcodes


if __name__ == '__main__':
    df = pd.read_csv('Data Preprocessing/processed_contracts.csv')


    def filter_supported_versions(code):
        """Filter out unsupported versions."""
        try:
            version = extract_version_from_pragma(code)
            if version:
                major_minor_version = float(version.split(".")[0] + "." + version.split(".")[1])
                return major_minor_version >= 0.4
        except ValueError:
            pass
        return False


    supported_df = df[df['code'].apply(filter_supported_versions)]

    # Apply the opcode extraction
    supported_df['Operation Codes'] = supported_df['code'].apply(extract_opcodes_from_source)

    supported_df.to_csv('Data Preprocessing/operation_contracts.csv', index=False)
    print("Processed data saved to operation_contracts.csv")
