# Jupyter Notebook Equivalent: CNN Model Training (PyTorch)
import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Load CSV with pandas
# ---------------------------------------------------------
csv_path = os.path.join(DATA_DIR, "ndn_traffic.csv")
if not os.path.exists(csv_path):
    print(f"Dataset not found at {csv_path}. Please run generate_dummy_csv.py first.")
    exit()

df = pd.read_csv(csv_path)
print("Data shape:", df.shape)

df = df.dropna()

required_cols = ['InInterests', 'OutInterests', 'InSatisfiedInterests', 'OutSatisfiedInterests',
                 'InTimedOutInterests', 'OutTimedOutInterests', 'InNacks', 'OutNacks', 'InData', 'OutData']

for col in required_cols:
    if col not in df.columns:
        df[col] = df.get('InInterests', 0) if 'Interest' in col else 0

EPS = 1e-6
ti = df['InInterests'] + df['OutInterests']
ts = df['InSatisfiedInterests'] + df['OutSatisfiedInterests']
tt = df['InTimedOutInterests']  + df['OutTimedOutInterests']
tn = df['InNacks']              + df['OutNacks']

df['interest_rate'] = ti
df['data_rate'] = df['InData'] + df['OutData']
df['satisfaction_ratio'] = ts / (ti + EPS)
df['network_load'] = (df['interest_rate'] + df['data_rate']) * 100 * 8 
df['pit_occupancy'] = ti - ts
df['timeout_ratio'] = tt / (ti + EPS)
df['nack_ratio'] = tn / (ti + EPS)

features = required_cols + ['interest_rate', 'data_rate', 'satisfaction_ratio', 'timeout_ratio', 'nack_ratio', 'pit_occupancy', 'network_load']
X = df[features].values
y_labels = df['Label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_mapping = {'Normal': 0, 'IFA': 1, 'Slow_IFA': 2, 'Cache_Pollution': 3, 'Distributed_IFA': 4, 'Pulsing_IFA': 5}
y_encoded = np.array([label_mapping[label] for label in y_labels])

WINDOW_SIZE = 10

def create_windows(data, labels, window_size):
    X_windows, y_windows = [], []
    for i in range(len(data) - window_size):
        X_windows.append(data[i:i + window_size])
        y_windows.append(labels[i + window_size - 1])
    return np.array(X_windows), np.array(y_windows)

X_win, y_win = create_windows(X_scaled, y_encoded, WINDOW_SIZE)
# PyTorch Conv1d expects (batch, channels, length). We transpose to (batch, features, window_size)
X_win = np.transpose(X_win, (0, 2, 1))

X_train, X_test, y_train, y_test = train_test_split(X_win, y_win, test_size=0.2, random_state=42)

train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_tensor = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_tensor, batch_size=32, shuffle=True)
test_loader = DataLoader(test_tensor, batch_size=32, shuffle=False)

# ---------------------------------------------------------
# 2. Build Advanced 1D CNN in PyTorch
# ---------------------------------------------------------
class AdvancedCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 2, 128)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.output(x)
        return x

model = AdvancedCNN(num_features=len(features), num_classes=len(label_mapping))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------
# 3. Train Model
# ---------------------------------------------------------
epochs = 25
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# ---------------------------------------------------------
# 4. Evaluate and Save
# ---------------------------------------------------------
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Test Accuracy: {accuracy*100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
target_names = list(label_mapping.keys())
print(classification_report(all_labels, all_preds, target_names=target_names))

model_save_path = os.path.join(MODEL_DIR, "ndn_cnn_model.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved successfully to {model_save_path}")

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved successfully.")
