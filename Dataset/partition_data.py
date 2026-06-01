import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# Split into 5 equal parts
# Using np.array_split is fine, but we will convert each back to a DataFrame
splits = np.array_split(df, 5)

for i, split in enumerate(splits):
    # Convert the array slice back to a DataFrame
    client_df = pd.DataFrame(split)
    # Save to CSV
    client_df.to_csv(f'client{i+1}.csv', index=False)

print("Data partitioned into 5 clients successfully!")
