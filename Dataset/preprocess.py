import pandas as pd

file_path = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

# Load the data
df = pd.read_csv(file_path)

# Print the basic info
print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())