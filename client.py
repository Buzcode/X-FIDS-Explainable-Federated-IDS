import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import StandardScaler
from ids_model import IDSModel

file_path = 'Dataset/client1.csv'

if not os.path.exists(file_path):
    print(f"ERROR: Could not find {file_path}")
else:
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Capture labels
    labels = df.iloc[:, -1]
    y = torch.tensor(labels.astype('category').cat.codes.values, dtype=torch.long)
    
    # Process features
    df = df.select_dtypes(include=['number'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)
    df = df.clip(lower=-1e10, upper=1e10)
    
    scaler = StandardScaler()
    X_data = scaler.fit_transform(df.values)
    X = torch.tensor(X_data, dtype=torch.float32)

    num_classes = len(labels.unique())
    print(f"Number of classes: {num_classes}")

    # Initialize model
    model = IDSModel(input_size=X.shape[1], output_size=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    print("Training started...")
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("Training finished successfully!")