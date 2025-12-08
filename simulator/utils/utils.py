import os
import pandas as pd

def set_csv_column(csv_path, column_name, value):
    """
    Set a specific column in a CSV file to a given value.
    """

    df = pd.read_csv(csv_path)
    if column_name in df.columns:
        df[column_name] = value
        df.to_csv(csv_path, index=False)
    else:
        raise ValueError(f"Column '{column_name}' does not exist in the CSV file.")
    
def set_file_row(file_path, prefix, value):
    # Read all lines from the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Modify the line with the given prefix
    for i, line in enumerate(lines):
        if line.startswith(prefix + ":"):
            lines[i] = f"{prefix}: {value}\n"
            break  # Stop after first match

    # Write back the modified lines
    with open(file_path, 'w') as f:
        f.writelines(lines)

def split_into_chunks(value, chunk_size):
    num_full_chunks = (value // chunk_size)
    remainder = value % chunk_size
    return num_full_chunks, remainder

def save_result(result_dict, csv_path):
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=result_dict.keys())
        df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)
    df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
    df.to_csv(csv_path, index=False)