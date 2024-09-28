import os
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')

import os
import pandas as pd


def get_sol_files_in_folder(folder_path):
    """Retrieve the list of .sol filenames in the folder."""
    return [file.replace('.sol','') for file in os.listdir(folder_path)]


def get_sol_files_from_csv(csv_file_path, column_name):
    """Retrieve the list of expected .sol filenames from the CSV file."""
    df = pd.read_csv(csv_file_path)
    sol_files = df[column_name].tolist()  # Extract the filenames from the specified column
    return sol_files


def find_missing_files(folder_path, csv_file_path, column_name):
    """Find the .sol files listed in the CSV file but not present in the folder."""
    actual_sol_files = get_sol_files_in_folder(folder_path)
    print(len(actual_sol_files))
    expected_sol_files = get_sol_files_from_csv(csv_file_path, column_name)

    # Compare and find missing files
    missing_files = [file for file in expected_sol_files if file not in actual_sol_files]
    return missing_files


# Example usage
folder_path = '1_Data Collection/Operation_code/sol/'  # Replace with your folder path
csv_file_path = '1_Data Collection/contracts.csv'  # Replace with your CSV file path
column_name = 'address'  # Replace with the actual column name in the CSV that lists the .sol filenames

missing_sol_files = find_missing_files(folder_path, csv_file_path, column_name)

if missing_sol_files:
    print(f"Missing .sol files: {missing_sol_files}")
    print(len(missing_sol_files))
else:
    print("All .sol files have been downloaded.")
