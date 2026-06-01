# Explainable Federated Learning for Cloud Intrusion Detection (X-FIDS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Flower](https://img.shields.io/badge/Framework-Flower-orange.svg)](https://flower.dev/)
[![Library: PyTorch](https://img.shields.io/badge/Library-PyTorch-red.svg)](https://pytorch.org/)

This repository hosts the official implementation of **X-FIDS** (Explainable Federated Intrusion Detection System), a privacy-preserving, transparent federated learning framework designed to secure cloud-native environments against Distributed Denial of Service (DDoS) attacks.

The framework simulates decentralized cloud nodes using the **Flower** framework and trains local **PyTorch** Multi-Layer Perceptrons (MLPs) on partitioned segments of the benchmark **CIC-IDS2017** dataset. Crucially, it integrates **SHAP** (SHapley Additive exPlanations) to provide feature-level interpretability, allowing Security Operations Center (SOC) analysts to verify and trust model decisions.

---

## 📖 Abstract
As cloud computing and IoT environments scale, securing cloud networks against Distributed Denial of Service (DDoS) attacks has become paramount. While machine learning offers robust intrusion detection systems (IDS), traditional centralized training models raise significant data privacy concerns. Federated Learning (FL) addresses this by enabling decentralized training; however, its practical adoption is hindered by the “black-box” nature of its global models. 

This project implements a privacy-preserving, explainable federated learning framework for cloud intrusion detection (X-FIDS). By utilizing the Flower framework and a local PyTorch multi-layer perceptron across simulated cloud nodes, we evaluate network traffic decentralized on the benchmark CIC-IDS2017 dataset. Crucially, we integrate SHAP (SHapley Additive exPlanations) to provide feature-level interpretability, enabling security analysts to verify model decisions. Our results demonstrate that the proposed framework achieves high classification accuracy while providing essential transparency for real-time cloud incident response.

---

## 📊 Key Results

### 1. Global Model Convergence
The federated model achieves stable convergence in just 3 rounds of training, with validation accuracy reaching **94.7%**.

| Federated Round | Avg. Training Loss | Validation Accuracy (%) |
|:---------------:|:------------------:|:-----------------------:|
| Round 1         | 0.7215             | 78.4%                   |
| Round 2         | 0.6935             | 89.1%                   |
| Round 3         | 0.6691             | **94.7%**               |

### 2. Performance vs. Centralized Baseline
Compared to a traditional centralized MLP trained on the entire raw dataset in a single repository, the decentralized **X-FIDS** framework retains comparable metrics while ensuring strict node privacy.

| Framework | Accuracy (%) | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|
| Centralized MLP | 95.8% | 0.942 | 0.938 | 0.940 |
| **X-FIDS (Proposed)** | **94.7%** | **0.932** | **0.930** | **0.931** |

---

## 🔍 Explainability Analysis (SHAP)
To decode the "black-box" decisions of the decentralized global model, we implement local SHAP Kernel Explanations. The feature attribution analysis highlights:
* **Destination Port** and **Total Length of Backward Packets** hold the highest global feature importance.
* Highly anomalous forward packet variance (**Forward Packet Length Std**) strongly pushes predictions toward the malicious **DDoS** class, providing immediate, diagnostic transparency to security analysts.

![SHAP Feature Importance](images/shap_summary.png)

---

## 🛠️ Installation & Usage

### 1. Clone the repository:
```bash
git clone https://github.com/Buzcode/X-FIDS-Explainable-Federated-IDS.git
cd X-FIDS-Explainable-Federated-IDS

2. Install dependencies:

pip install -r requirements.txt

3. Run the Simulation:

  - Step 1: Partition the Dataset
    Generate the K=5 decentralized client partitions from the raw CIC-IDS2017
    dataset.

    python src/data_partition.py

  - Step 2: Start the Flower Server

    python src/server.py

  - Step 3: Launch Clients (Open separate terminals for each simulated node)

    python src/client.py --client_id=1
    python src/client.py --client_id=2
    # Repeat up to client_id=5

  - Step 4: Generate SHAP Explanations

    python src/shap_explainer.py

📜 Citation

If you use this code or research in your work, please cite our paper:

@inproceedings{mawa2026explainable,
  title={Explainable Federated Learning for Cloud Intrusion Detection},
  author={Mawa, Jannatul and Zerin, Sumona Islam},
  booktitle={Proceedings of the IEEE International Conference on Signal Processing, Information, Communication and Systems (SPICSCON)},
  year={2026}
}

📄 License

This project is licensed under the MIT License - see the LICENSE file for
details.

