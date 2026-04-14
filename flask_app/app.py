from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Base path for project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ndn_cnn_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# Load compiled model and scaler
model = None
scaler = None

# New 6-Class Label Mapping
LABEL_MAPPING = {0:'Normal', 1:'IFA', 2:'Slow_IFA', 3:'Cache_Pollution', 4:'Distributed_IFA', 5:'Pulsing_IFA'}
WINDOW_SIZE = 10
NUM_FEATURES = 17

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

def load_resources():
    global model, scaler
    if os.path.exists(MODEL_PATH):
        try:
            model = AdvancedCNN(NUM_FEATURES, len(LABEL_MAPPING))
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print("Error loading model:", e)
    else:
        print("Model not found! Please ensure it is at", MODEL_PATH)

    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully.")
    else:
        print("Scaler not found. Will fit a new StandardScaler on incoming test batches.")

# Create templates and static dirs if they don't exist
os.makedirs(os.path.join(BASE_DIR, "flask_app", "templates"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "flask_app", "static"), exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        load_resources()
        if model is None:
            return jsonify({'error': 'Model not found!'}), 500

    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            # Fallback to default file if it exists
            default_path = os.path.join(BASE_DIR, "dataset", "ndn_traffic.csv")
            if os.path.exists(default_path):
                 df = pd.read_csv(default_path)
                 print(f"Using default dataset: {default_path}")
            else:
                 return jsonify({'error': 'No file uploaded and default dataset not found!'}), 400
        else:
            file = request.files['file']
            df = pd.read_csv(file)
            
        # Original columns for metrics calculation before processing
        raw_df = df.copy()

        # Extract Labels for Accuracy if available
        actual_labels = None
        if 'Label' in raw_df.columns:
            actual_labels = raw_df['Label'].fillna('Normal').values
        
        # Mirroring the Colab Feature Engineering Logic
        for col in ['Time', 'Node', 'FaceDescr', 'Label']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Check required columns for engineered features
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

        df['interest_rate']      = ti
        df['data_rate']          = df['InData'] + df['OutData']
        df['satisfaction_ratio'] = ts / (ti + EPS)
        # Network Load estimation (Interest + Data) - simplified bits/sec assuming 100 bytes avg
        df['network_load']       = (df['interest_rate'] + df['data_rate']) * 100 * 8 
        df['pit_occupancy']      = ti - ts
        
        # Fill Medians just like Training Notebook
        df.fillna(df.median(numeric_only=True), inplace=True)
        
        # Features for model prediction (11 raw + 6 engineered = 17 features)
        # Note: training used 17 features, let's ensure we match the exact set
        # Re-calculating additional ratios if needed to match training exactly
        df['timeout_ratio']      = tt / (ti + EPS)
        df['nack_ratio']         = tn / (ti + EPS)
        
        # Selecing the 17 features in order (hypothetical order based on previous code)
        # 10 raw + data_rate, satisfaction_ratio, timeout_ratio, nack_ratio, pit_estimate, interest_rate
        prediction_cols = required_cols + ['interest_rate', 'data_rate', 'satisfaction_ratio', 'timeout_ratio', 'nack_ratio', 'pit_occupancy']
        X_df = df[prediction_cols]
        X = X_df.values
        
        if X.shape[1] < NUM_FEATURES:
            pad_width = NUM_FEATURES - X.shape[1]
            X = np.pad(X, ((0, 0), (0, pad_width)), mode='constant')
        elif X.shape[1] > NUM_FEATURES:
            X = X[:, :NUM_FEATURES]
        
        if len(X) < WINDOW_SIZE:
             return jsonify({'error': f'Dataset needs at least {WINDOW_SIZE} rows to form a time window.'}), 400
             
        # Scale data
        active_scaler = scaler
        if active_scaler is None:
            active_scaler = StandardScaler()
            X_scaled = active_scaler.fit_transform(X)
        else:
            X_scaled = active_scaler.transform(X)
        
        # Prepare all windows for batch inference
        windows = []
        for i in range(len(X_scaled) - WINDOW_SIZE + 1):
            windows.append(X_scaled[i : i + WINDOW_SIZE])
        
        # Transpose to match PyTorch (batch, features, window_size)
        all_windows_batch = np.transpose(np.array(windows), (0, 2, 1))
        
        # Perform highly efficient batch prediction
        print(f"Starting batch inference for {len(all_windows_batch)} windows...")
        with torch.no_grad():
            inputs = torch.tensor(all_windows_batch, dtype=torch.float32)
            outputs = model(inputs)
            predictions = F.softmax(outputs, dim=1).numpy()
        print("Batch inference complete.")

        # Map predictions to results
        all_results = []
        for pred in predictions:
            class_idx = np.argmax(pred)
            confidence = float(np.max(pred))
            all_results.append({'class': class_idx, 'confidence': confidence})

        # Calculate final state (based on the last window)
        last_prediction = all_results[-1]
        status = LABEL_MAPPING.get(last_prediction['class'], "Unknown")
        is_attack = "Interest Flooding Attack Detected" if status != "Normal" else "Normal Traffic"
        
        # Scoring Logic
        attack_score = last_prediction['confidence'] * 100
        if status == "Normal":
            # If model says normal but with low confidence, or just for consistency
            attack_score = (1 - last_prediction['confidence']) * 30 # capped low for normal
        
        severity = "Low"
        if attack_score >= 70: severity = "High"
        elif attack_score >= 50: severity = "Medium"

        # Calculate Accuracy if Ground Truth exists
        accuracy = None
        if actual_labels is not None and len(actual_labels) >= WINDOW_SIZE:
            # We match the valid windows length
            valid_labels = actual_labels[WINDOW_SIZE-1:]
            if len(valid_labels) == len(all_results):
                matches = sum(1 for label, res in zip(valid_labels, all_results) if LABEL_MAPPING.get(res['class']) == label)
                accuracy = round((matches / len(all_results)) * 100, 2)
            
        # Summary Stats
        attack_count = sum(1 for r in all_results if LABEL_MAPPING.get(r['class']) != "Normal")
        total_windows = len(all_results)
        class_counts = {"IFA":0, "Slow_IFA":0, "Cache_Pollution":0, "Distributed_IFA":0, "Pulsing_IFA":0}
        
        for r in all_results:
            c_name = LABEL_MAPPING.get(r['class'])
            if c_name in class_counts:
                class_counts[c_name] += 1

        # Real-time Metrics (from last row)
        last_row = df.iloc[-1]
        metrics = {
            'interest_rate': round(last_row['interest_rate'], 2),
            'pit_occupancy': int(last_row['pit_occupancy']),
            'satisfaction_ratio': round(last_row['satisfaction_ratio'] * 100, 2),
            'network_load': round(last_row['network_load'], 2),
            'attack_score': round(attack_score, 2),
            'accuracy': accuracy if accuracy is not None else "-"
        }

        # Graph Data (last 30 rows)
        history = df.tail(30)
        
        response = {
            'status': is_attack,
            'severity_level': severity,
            'metrics': metrics,
            'alert_timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'class_counts': class_counts,
            'graph_data': {
                'labels': list(range(len(history))),
                'interest_rate': history['interest_rate'].tolist(),
                'pit_occupancy': history['pit_occupancy'].tolist(),
                'satisfaction_ratio': history['satisfaction_ratio'].tolist(),
                'timeout_ratio': history['timeout_ratio'].tolist(),
                'nack_ratio': history['nack_ratio'].tolist(),
                'network_load': history['network_load'].tolist()
            },
            'summary_report': {
                'total_attacks': attack_count,
                'total_windows': total_windows,
                'attack_percentage': round((attack_count / total_windows) * 100, 2) if total_windows > 0 else 0
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_resources()
    app.run(debug=True)
