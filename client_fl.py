import flwr as fl
import torch
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from ids_model import IDSModel

# Choose client file dynamically
client_num = sys.argv[1] if len(sys.argv) > 1 else "1"
file_path = f'Dataset/client{client_num}.csv'
print(f"Loading data from {file_path}...")

# encoding='utf-8-sig' handles invisible BOM characters like \ufeff
df = pd.read_csv(file_path, encoding='utf-8-sig')

# 1. Clean column names
df.columns = df.columns.astype(str).str.strip()

# 2. Extract the target label (always the very last column)
y_raw = df.iloc[:, -1].values

# 3. Dynamically select the first 10 numeric features from the remaining columns
# This excludes the last column to prevent training on the target label
feature_df = df.iloc[:, :-1]
numeric_cols = feature_df.select_dtypes(include=['number']).columns.tolist()
features_to_keep = numeric_cols[:10]

print(f"Features selected for client {client_num}: {features_to_keep}")

# Extract features and handle missing data
df_features = df[features_to_keep].copy()
df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
df_features.fillna(0, inplace=True)

# Convert features to float arrays and clip overflows
X_raw = df_features.values.astype(np.float32)
X_raw = np.clip(X_raw, -1e10, 1e10)

# Scale features
scaler = StandardScaler()
X_data = scaler.fit_transform(X_raw)

# Convert to PyTorch tensors
X = torch.tensor(X_data, dtype=torch.float32)
y = torch.tensor(pd.factorize(y_raw)[0], dtype=torch.long)

# Initialize model
num_features = X.shape[1]  # This is locked at 10
num_classes = len(np.unique(y_raw))
model = IDSModel(input_size=num_features, output_size=num_classes)

class IDSClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def fit(self, parameters, config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        outputs = model(X)
        loss = torch.nn.CrossEntropyLoss()(outputs, y)
        loss.backward()
        optimizer.step()
        # Save the trained model weights locally
        torch.save(model.state_dict(), "global_model.pth")
        
        return self.get_parameters(config={}), len(X), {"loss": loss.item()}

# Start the client using modern API syntax
if __name__ == "__main__":
    client = IDSClient().to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)