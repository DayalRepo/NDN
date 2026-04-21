"""
Multi-Model Comparison for NDN Attack Detection
Comprehensive training, validation, and metrics calculation using XGBoost, 
Random Forest, Gradient Boosting, SVM, and Neural Network models.
Includes ROC curves, confusion matrices, learning curves, and ensemble approach.
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, roc_auc_score, 
    classification_report, cohen_kappa_score, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
REPORT_DIR = os.path.join(BASE_DIR, 'model_analysis')
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
WINDOW_SIZE = 10
NUM_CLASSES = 6

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================
def load_and_prepare_data():
    """Load CSV, engineer features, create windows"""
    csv_path = os.path.join(DATA_DIR, "ndn_traffic.csv")
    
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")
    
    # Feature engineering
    required_cols = ['InInterests', 'OutInterests', 'InSatisfiedInterests', 
                     'OutSatisfiedInterests', 'InTimedOutInterests', 'OutTimedOutInterests',
                     'InNacks', 'OutNacks', 'InData', 'OutData']
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = df.get('InInterests', 0) if 'Interest' in col else 0
    
    eps = 1e-6
    ti = df['InInterests'] + df['OutInterests']
    ts = df['InSatisfiedInterests'] + df['OutSatisfiedInterests']
    tt = df['InTimedOutInterests'] + df['OutTimedOutInterests']
    tn = df['InNacks'] + df['OutNacks']
    
    df['interest_rate'] = ti
    df['data_rate'] = df['InData'] + df['OutData']
    df['satisfaction_ratio'] = ts / (ti + eps)
    df['timeout_ratio'] = tt / (ti + eps)
    df['nack_ratio'] = tn / (ti + eps)
    df['pit_occupancy'] = ti - ts
    df['network_load'] = (ti + df['data_rate']) * 100 * 8
    
    df = df.fillna(0)
    
    # Extract features and labels
    feature_cols = required_cols + ['interest_rate', 'data_rate', 'satisfaction_ratio', 
                                     'timeout_ratio', 'nack_ratio', 'pit_occupancy', 'network_load']
    X = df[feature_cols].values
    y = df['Label'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Features shape: {X_scaled.shape}")
    print(f"Classes: {le.classes_}")
    print(f"Class distribution: {np.bincount(y_encoded)}")
    
    return X_scaled, y_encoded, le, scaler, df


def create_windows(X, y, window_size=10):
    """Create sliding windows for time-series data"""
    X_windows, y_windows = [], []
    for i in range(len(X) - window_size):
        X_windows.append(X[i:i + window_size])
        y_windows.append(y[i + window_size - 1])
    return np.array(X_windows), np.array(y_windows)


# ============================================================================
# 2. NEURAL NETWORK MODEL
# ============================================================================
class AttackDetectionNN(nn.Module):
    """1D CNN for attack detection"""
    def __init__(self, num_features, num_classes):
        super(AttackDetectionNN, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 2, 128)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def train_neural_network(X_train, X_test, y_train, y_test, num_features, num_classes, epochs=50):
    """Train PyTorch neural network with learning curve tracking"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reshape for CNN: (batch, features, window_size)
    X_train_nn = np.transpose(X_train, (0, 2, 1))
    X_test_nn = np.transpose(X_test, (0, 2, 1))
    
    train_dataset = TensorDataset(
        torch.tensor(X_train_nn, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = AttackDetectionNN(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    train_losses = []
    train_accs = []
    learning_rates = []
    
    print("\nTraining Neural Network...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(avg_loss)
        train_accs.append(accuracy)
        learning_rates.append(current_lr)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, LR: {current_lr:.6f}")
    
    # Test predictions
    model.eval()
    X_test_tensor = torch.tensor(X_test_nn, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)
    
    y_pred_nn = predictions.cpu().numpy()
    
    return model, y_pred_nn, train_losses, train_accs, learning_rates


# ============================================================================
# 3. TRAIN ALL MODELS
# ============================================================================
def train_all_models(X_train, X_test, y_train, y_test, num_features, num_classes):
    """Train XGBoost, Random Forest, Gradient Boosting, SVM"""
    results = {}
    models = {}
    
    # Flatten windows for tree-based models: (batch, window, features) -> (batch, window*features)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 1. XGBoost
    print("\n" + "="*60)
    print("Training XGBoost...")
    print("="*60)
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=num_classes,
        random_state=RANDOM_STATE,
        verbosity=0
    )
    
    # Train XGBoost
    xgb_model.fit(X_train_flat, y_train)
    
    y_pred_xgb = xgb_model.predict(X_test_flat)
    y_proba_xgb = xgb_model.predict_proba(X_test_flat)
    
    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    xgb_prec = precision_score(y_test, y_pred_xgb, average='weighted', zero_division=0)
    xgb_rec = recall_score(y_test, y_pred_xgb, average='weighted')
    xgb_f1 = f1_score(y_test, y_pred_xgb, average='weighted')
    
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {
        'accuracy': xgb_acc,
        'precision': xgb_prec,
        'recall': xgb_rec,
        'f1': xgb_f1,
        'y_pred': y_pred_xgb,
        'y_proba': y_proba_xgb
    }
    print(f"XGBoost - Accuracy: {xgb_acc:.4f}, Precision: {xgb_prec:.4f}, Recall: {xgb_rec:.4f}, F1: {xgb_f1:.4f}")
    
    # 2. Random Forest
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train_flat, y_train)
    y_pred_rf = rf_model.predict(X_test_flat)
    y_proba_rf = rf_model.predict_proba(X_test_flat)
    
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_prec = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    rf_rec = recall_score(y_test, y_pred_rf, average='weighted')
    rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')
    
    models['RandomForest'] = rf_model
    results['RandomForest'] = {
        'accuracy': rf_acc,
        'precision': rf_prec,
        'recall': rf_rec,
        'f1': rf_f1,
        'y_pred': y_pred_rf,
        'y_proba': y_proba_rf
    }
    print(f"RandomForest - Accuracy: {rf_acc:.4f}, Precision: {rf_prec:.4f}, Recall: {rf_rec:.4f}, F1: {rf_f1:.4f}")
    
    # 3. Gradient Boosting
    print("\n" + "="*60)
    print("Training Gradient Boosting...")
    print("="*60)
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    gb_model.fit(X_train_flat, y_train)
    y_pred_gb = gb_model.predict(X_test_flat)
    y_proba_gb = gb_model.predict_proba(X_test_flat)
    
    gb_acc = accuracy_score(y_test, y_pred_gb)
    gb_prec = precision_score(y_test, y_pred_gb, average='weighted', zero_division=0)
    gb_rec = recall_score(y_test, y_pred_gb, average='weighted')
    gb_f1 = f1_score(y_test, y_pred_gb, average='weighted')
    
    models['GradientBoosting'] = gb_model
    results['GradientBoosting'] = {
        'accuracy': gb_acc,
        'precision': gb_prec,
        'recall': gb_rec,
        'f1': gb_f1,
        'y_pred': y_pred_gb,
        'y_proba': y_proba_gb
    }
    print(f"GradientBoosting - Accuracy: {gb_acc:.4f}, Precision: {gb_prec:.4f}, Recall: {gb_rec:.4f}, F1: {gb_f1:.4f}")
    
    # 4. Neural Network
    print("\n" + "="*60)
    print("Training Neural Network (1D CNN)...")
    print("="*60)
    nn_model, y_pred_nn, train_losses, train_accs, learning_rates = train_neural_network(
        X_train, X_test, y_train, y_test, num_features, num_classes, epochs=50
    )
    
    nn_acc = accuracy_score(y_test, y_pred_nn)
    nn_prec = precision_score(y_test, y_pred_nn, average='weighted', zero_division=0)
    nn_rec = recall_score(y_test, y_pred_nn, average='weighted')
    nn_f1 = f1_score(y_test, y_pred_nn, average='weighted')
    
    # Get probabilities for NN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_nn = np.transpose(X_test, (0, 2, 1))
    X_test_tensor = torch.tensor(X_test_nn, dtype=torch.float32).to(device)
    nn_model.eval()
    with torch.no_grad():
        outputs = nn_model(X_test_tensor)
        y_proba_nn = torch.softmax(outputs, dim=1).cpu().numpy()
    
    models['NeuralNetwork'] = nn_model
    results['NeuralNetwork'] = {
        'accuracy': nn_acc,
        'precision': nn_prec,
        'recall': nn_rec,
        'f1': nn_f1,
        'y_pred': y_pred_nn,
        'y_proba': y_proba_nn,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'learning_rates': learning_rates
    }
    print(f"NeuralNetwork - Accuracy: {nn_acc:.4f}, Precision: {nn_prec:.4f}, Recall: {nn_rec:.4f}, F1: {nn_f1:.4f}")
    
    return models, results


# ============================================================================
# 4. ADVANCED METRICS & ANALYSIS
# ============================================================================
def calculate_advanced_metrics(y_test, y_pred, y_proba, model_name, num_classes):
    """Calculate comprehensive metrics including ROC, confusion matrix, etc."""
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred),
        'matthews_cc': matthews_corrcoef(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'error_rate': 1 - accuracy_score(y_test, y_pred)
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class
    
    # ROC-AUC (One-vs-Rest for multiclass)
    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        metrics['roc_auc'] = roc_auc
    except:
        metrics['roc_auc'] = None
    
    return metrics


def generate_all_metrics_report(models, results, y_test, le, num_classes):
    """Generate comprehensive metrics for all models"""
    all_metrics = {}
    
    for model_name, predictions in results.items():
        y_pred = predictions['y_pred']
        y_proba = predictions['y_proba']
        
        metrics = calculate_advanced_metrics(y_test, y_pred, y_proba, model_name, num_classes)
        all_metrics[model_name] = metrics
        
        print(f"\n{'='*60}")
        print(f"Detailed Metrics for {model_name}")
        print(f"{'='*60}")
        print(f"Accuracy:       {metrics['accuracy']:.4f}")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1-Score:       {metrics['f1']:.4f}")
        print(f"Cohen Kappa:    {metrics['cohen_kappa']:.4f}")
        print(f"Matthews CC:    {metrics['matthews_cc']:.4f}")
        print(f"Error Rate:     {metrics['error_rate']:.4f}")
        if metrics['roc_auc']:
            print(f"ROC-AUC:        {metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    
    return all_metrics


# ============================================================================
# 5. VISUALIZATION
# ============================================================================
def plot_confusion_matrices(all_metrics, num_classes, le):
    """Plot confusion matrices for all models"""
    models_list = list(all_metrics.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(models_list):
        cm = all_metrics[model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                   xticklabels=le.classes_, yticklabels=le.classes_)
        axes[idx].set_title(f'{model_name}\nAccuracy: {all_metrics[model_name]["accuracy"]:.4f}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    print("✓ Confusion matrices saved")
    plt.close()


def plot_roc_curves(all_metrics, y_test, results, num_classes):
    """Plot ROC curves for all models (One-vs-Rest)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    models_list = list(all_metrics.keys())
    
    for model_idx, model_name in enumerate(models_list):
        ax = axes[model_idx]
        y_proba = results[model_name]['y_proba']
        
        # Plot One-vs-Rest ROC curves
        for class_idx in range(num_classes):
            y_test_binary = (y_test == class_idx).astype(int)
            fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, class_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'Class {class_idx} (AUC={roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - ROC Curves')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    print("✓ ROC curves saved")
    plt.close()


def plot_model_comparison(all_metrics):
    """Compare all models on key metrics"""
    models = list(all_metrics.keys())
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    data_dict = {metric: [all_metrics[m].get(metric, 0) for m in models] for metric in metrics_to_plot}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.15
    
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - 2) * width
        values = data_dict[metric]
        ax.bar(x + offset, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Comparison - Key Metrics', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - 2) * width
        for j, val in enumerate(data_dict[metric]):
            ax.text(j + offset, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    print("✓ Model comparison chart saved")
    plt.close()


def plot_neural_network_training(results):
    """Plot NN training metrics: loss, accuracy, learning rate"""
    if 'NeuralNetwork' not in results:
        return
    
    nn_result = results['NeuralNetwork']
    if 'train_losses' not in nn_result:
        return
    
    train_losses = nn_result['train_losses']
    train_accs = nn_result['train_accs']
    learning_rates = nn_result['learning_rates']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curve
    axes[0].plot(train_losses, linewidth=2, color='#d62728')
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Training Loss', fontweight='bold')
    axes[0].set_title('Training Loss Over Epochs', fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].fill_between(range(len(train_losses)), train_losses, alpha=0.3, color='#d62728')
    
    # Accuracy curve
    axes[1].plot(train_accs, linewidth=2, color='#2ca02c')
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Training Accuracy', fontweight='bold')
    axes[1].set_title('Training Accuracy Over Epochs', fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].fill_between(range(len(train_accs)), train_accs, alpha=0.3, color='#2ca02c')
    axes[1].set_ylim([0, 1])
    
    # Learning rate decay
    axes[2].plot(learning_rates, linewidth=2, color='#1f77b4')
    axes[2].set_xlabel('Epoch', fontweight='bold')
    axes[2].set_ylabel('Learning Rate', fontweight='bold')
    axes[2].set_title('Learning Rate Decay', fontweight='bold')
    axes[2].grid(alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'neural_network_training.png'), dpi=300, bbox_inches='tight')
    print("✓ Neural network training curves saved")
    plt.close()


def plot_per_class_metrics(all_metrics, le):
    """Plot per-class precision, recall, f1"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = list(all_metrics.keys())
    x = np.arange(len(le.classes_))
    width = 0.2
    
    for metric_idx, metric_name in enumerate(['precision_per_class', 'recall_per_class', 'f1_per_class']):
        ax = axes[metric_idx]
        for model_idx, model_name in enumerate(models):
            values = all_metrics[model_name][metric_name]
            offset = (model_idx - len(models)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model_name)
        
        ax.set_xlabel('Class', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'{metric_name.replace("_per_class", "").capitalize()} per Class', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(le.classes_, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    print("✓ Per-class metrics chart saved")
    plt.close()


def plot_error_analysis(all_metrics):
    """Plot error rates and other diagnostic metrics"""
    models = list(all_metrics.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Error rate comparison
    error_rates = [all_metrics[m]['error_rate'] for m in models]
    axes[0, 0].bar(models, error_rates, color='#d62728')
    axes[0, 0].set_ylabel('Error Rate', fontweight='bold')
    axes[0, 0].set_title('Error Rates', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(error_rates):
        axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # Cohen Kappa comparison
    kappas = [all_metrics[m]['cohen_kappa'] for m in models]
    axes[0, 1].bar(models, kappas, color='#ff7f0e')
    axes[0, 1].set_ylabel('Cohen Kappa', fontweight='bold')
    axes[0, 1].set_title('Cohen Kappa Score', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(kappas):
        axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # Matthews CC comparison
    matthews = [all_metrics[m]['matthews_cc'] for m in models]
    axes[1, 0].bar(models, matthews, color='#2ca02c')
    axes[1, 0].set_ylabel('Matthews Correlation', fontweight='bold')
    axes[1, 0].set_title('Matthews Correlation Coefficient', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(matthews):
        axes[1, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # ROC-AUC comparison
    roc_aucs = [all_metrics[m].get('roc_auc', 0) or 0 for m in models]
    axes[1, 1].bar(models, roc_aucs, color='#1f77b4')
    axes[1, 1].set_ylabel('ROC-AUC', fontweight='bold')
    axes[1, 1].set_title('ROC-AUC Score', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(roc_aucs):
        axes[1, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'error_analysis.png'), dpi=300, bbox_inches='tight')
    print("✓ Error analysis chart saved")
    plt.close()


# ============================================================================
# 6. ENSEMBLE & BEST MODEL SELECTION
# ============================================================================
def evaluate_ensemble(X_test, y_test, results, all_metrics):
    """Create and evaluate voting ensemble"""
    from sklearn.ensemble import VotingClassifier
    
    # Use predictions from all tree-based models for voting
    ensemble_preds = np.zeros((len(y_test), len(list(all_metrics.keys()))))
    
    for idx, model_name in enumerate(list(all_metrics.keys())):
        ensemble_preds[:, idx] = results[model_name]['y_pred']
    
    # Majority voting
    ensemble_pred = np.median(ensemble_preds, axis=1).astype(int)
    ensemble_pred = np.clip(ensemble_pred, 0, 5)  # Ensure valid class range
    
    ens_acc = accuracy_score(y_test, ensemble_pred)
    ens_prec = precision_score(y_test, ensemble_pred, average='weighted', zero_division=0)
    ens_rec = recall_score(y_test, ensemble_pred, average='weighted')
    ens_f1 = f1_score(y_test, ensemble_pred, average='weighted')
    
    ensemble_metrics = {
        'model': 'Ensemble',
        'accuracy': ens_acc,
        'precision': ens_prec,
        'recall': ens_rec,
        'f1': ens_f1,
        'y_pred': ensemble_pred
    }
    
    print(f"\n{'='*60}")
    print(f"Ensemble Model Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {ens_acc:.4f}")
    print(f"Precision: {ens_prec:.4f}")
    print(f"Recall:    {ens_rec:.4f}")
    print(f"F1-Score:  {ens_f1:.4f}")
    
    return ensemble_metrics


def select_best_model(all_metrics):
    """Select best model based on F1 score"""
    best_model = max(all_metrics.items(), key=lambda x: x[1]['f1'])
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model[0]}")
    print(f"{'='*60}")
    print(f"Accuracy:  {best_model[1]['accuracy']:.4f}")
    print(f"Precision: {best_model[1]['precision']:.4f}")
    print(f"Recall:    {best_model[1]['recall']:.4f}")
    print(f"F1-Score:  {best_model[1]['f1']:.4f}")
    
    return best_model


# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
def save_results(models, all_metrics, ensemble_metrics):
    """Save models and metrics to files"""
    
    # Save metrics
    metrics_summary = {
        'individual_models': {name: {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'error_rate': metrics['error_rate'],
            'cohen_kappa': metrics['cohen_kappa'],
            'matthews_cc': metrics['matthews_cc'],
            'roc_auc': metrics.get('roc_auc', None)
        } for name, metrics in all_metrics.items()},
        'ensemble': {
            'accuracy': ensemble_metrics['accuracy'],
            'precision': ensemble_metrics['precision'],
            'recall': ensemble_metrics['recall'],
            'f1': ensemble_metrics['f1']
        }
    }
    
    with open(os.path.join(MODEL_DIR, 'metrics_summary.pkl'), 'wb') as f:
        pickle.dump(metrics_summary, f)
    
    # Save individual models
    for model_name, model in models.items():
        if model_name != 'NeuralNetwork':
            model_path = os.path.join(MODEL_DIR, f'{model_name.lower()}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved {model_name} model")
    
    print(f"\n✓ All results saved to {MODEL_DIR}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*80)
    print("NDN ATTACK DETECTION - COMPREHENSIVE MULTI-MODEL ANALYSIS".center(80))
    print("="*80)
    
    # Load and prepare data
    print("\nPhase 1: Loading and preparing data...")
    data_result = load_and_prepare_data()
    if data_result is None:
        return
    
    X_scaled, y_encoded, le, scaler, df = data_result
    
    # Create windows
    print("Creating sliding windows...")
    X_win, y_win = create_windows(X_scaled, y_encoded, window_size=WINDOW_SIZE)
    print(f"Windows created: {X_win.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_win, y_win, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_win
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    num_features = X_win.shape[2]
    
    # Train all models
    print("\n" + "="*80)
    print("Phase 2: Training all models...")
    print("="*80)
    models, results = train_all_models(X_train, X_test, y_train, y_test, num_features, NUM_CLASSES)
    
    # Calculate comprehensive metrics
    print("\n" + "="*80)
    print("Phase 3: Calculating advanced metrics...")
    print("="*80)
    all_metrics = generate_all_metrics_report(models, results, y_test, le, NUM_CLASSES)
    
    # Ensemble model
    print("\n" + "="*80)
    print("Phase 4: Evaluating ensemble...")
    print("="*80)
    ensemble_metrics = evaluate_ensemble(X_test, y_test, results, all_metrics)
    
    # Select best model
    best_model_name, best_model_metrics = select_best_model(all_metrics)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Phase 5: Generating visualizations...")
    print("="*80)
    plot_confusion_matrices(all_metrics, NUM_CLASSES, le)
    plot_roc_curves(all_metrics, y_test, results, NUM_CLASSES)
    plot_model_comparison(all_metrics)
    plot_neural_network_training(results)
    plot_per_class_metrics(all_metrics, le)
    plot_error_analysis(all_metrics)
    
    # Save results
    print("\n" + "="*80)
    print("Phase 6: Saving models and results...")
    print("="*80)
    save_results(models, all_metrics, ensemble_metrics)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE".center(80))
    print("="*80)
    
    return {
        'models': models,
        'all_metrics': all_metrics,
        'ensemble_metrics': ensemble_metrics,
        'best_model': best_model_name,
        'results': results,
        'label_encoder': le,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test
    }


if __name__ == '__main__':
    analysis_results = main()
