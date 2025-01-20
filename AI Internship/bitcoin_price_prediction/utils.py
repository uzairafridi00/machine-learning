# Helper functions (e.g., CSV saving)

import os

# ------------------------------- Function: Save Data to CSV -----------------------------
def save_to_csv(df, filename="bitcoin_data.csv"):
    """
    Saves the DataFrame to a CSV file. Appends if the file already exists.
    
    Parameters:
        df (pd.DataFrame): DataFrame to save.
        filename (str): Name of the CSV file.
    """
    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode="a", header=False, index=False)
