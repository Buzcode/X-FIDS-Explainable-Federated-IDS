import torch
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ids_model import IDSModel

print("Loading data for explanation...")
df = pd.read_csv('Dataset/client1.csv')

df = df.select_dtypes(include=['number'])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

X_raw = df.iloc[:, :-1].values
feature_names = df.iloc[:, :-1].columns.tolist()

X_raw = X_raw[:, :10]
feature_names = feature_names[:10]

# --- NEW: Map numeric column indices to actual security feature names ---
friendly_names = {
    '0': 'Destination Port',
    '1': 'Flow Duration',
    '2': 'Total Fwd Packets',
    '3': 'Total Bwd Packets',
    '4': 'Total Length Fwd Packets',
    '5': 'Total Length Bwd Packets',
    '6': 'Fwd Packet Length Max',
    '7': 'Fwd Packet Length Min',
    '8': 'Fwd Packet Length Mean',
    '9': 'Fwd Packet Length Std'
}
feature_names = [friendly_names.get(name, name) for name in feature_names]
# ------------------------------------------------------------------------

scaler = StandardScaler()
X_data = scaler.fit_transform(X_raw)

print("Loading the trained Federated model...")
model = IDSModel(input_size=10, output_size=2)
model.load_state_dict(torch.load("global_model.pth"))
model.eval()

def predict_probability(x):
    tensor_x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(tensor_x)
        probabilities = torch.softmax(outputs, dim=1)
    return probabilities.numpy()

print("Calculating SHAP values using KernelExplainer...")
background = X_data[:50] 
test_samples = X_data[100:115]  

explainer = shap.KernelExplainer(predict_probability, background)
shap_values = explainer.shap_values(test_samples)

print("Generating SHAP summary plot...")
plt.figure(figsize=(10, 6))

if isinstance(shap_values, list):
    shap_val_class1 = shap_values[1]
else:
    shap_val_class1 = shap_values[:, :, 1]

shap.summary_plot(shap_val_class1, test_samples, feature_names=feature_names, show=False)

plt.title("SHAP Feature Importance for Federated DDoS Detection", fontsize=14)
plt.tight_layout()
plt.savefig("shap_explanation.png")
print("SHAP explanation plot saved successfully as 'shap_explanation.png'!")