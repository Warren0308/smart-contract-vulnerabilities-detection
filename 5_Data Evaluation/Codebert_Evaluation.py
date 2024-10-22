import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
import os

# Set your working directory
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')

# Define the MLP Classifier (same as during training)
class MLPClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=1024, num_labels=5, dropout=0.2):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the test CSV file
df_test = pd.read_csv('3_Data Encoding/SC_test.csv')  # Replace with the path to your test dataset

# Define X_test (features) and y_test (true labels)
X_test = df_test[[f'embedding_{i+1}' for i in range(768)]].values  # 768-dimensional embeddings
y_test = df_test[['ARTHM', 'LE', 'None', 'RENT', 'TimeO']].values  # Multi-label binary targets

# Load the saved model and hyperparameters from the checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("4_Data Modelling/SC_model.pth")

# Extract saved hyperparameters from the checkpoint
best_params = checkpoint['best_params']
hidden_size = best_params['hidden_size']
dropout = best_params['dropout']

# Initialize the model with the saved hyperparameters
model = MLPClassifier(input_size=768, hidden_size=hidden_size, num_labels=5, dropout=dropout).to(device)

# Load the model state
model.load_state_dict(checkpoint['model_state_dict'])

# Load the thresholds if they were saved
best_thresholds = checkpoint.get('thresholds', [0.5] * 5)  # Use default 0.5 if not in checkpoint
model.eval()  # Set the model to evaluation mode

# Convert test data to tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Perform predictions
with torch.no_grad():
    outputs = model(X_test_tensor)
    sigmoid_outputs = torch.sigmoid(outputs).cpu().numpy()  # Convert logits to probabilities

# Apply different thresholds for each label (vulnerability type)
num_labels = sigmoid_outputs.shape[1]  # Number of labels, should be 5 in this case
y_pred_best = np.zeros_like(sigmoid_outputs)  # Initialize array for binary predictions

for label_idx in range(num_labels):
    y_pred_best[:, label_idx] = (sigmoid_outputs[:, label_idx] > best_thresholds[label_idx]).astype(int)

# Evaluate the model for each vulnerability type and overall
vulnerability_types = ['ARTHM', 'LE', 'None', 'RENT', 'TimeO']

# Evaluate the predictions using F1-score, precision, recall
y_true = y_test_tensor.cpu().numpy()  # Ground truth labels
f1 = f1_score(y_true, y_pred_best, average='macro', zero_division=0)
f1_per_class = f1_score(y_true, y_pred_best, average=None)
precision_per_class = precision_score(y_true, y_pred_best, average=None)
recall_per_class = recall_score(y_true, y_pred_best, average=None)

# Correcting the overall accuracy calculation (no average param needed)
overall_accuracy = accuracy_score(y_true, y_pred_best)  # Calculate overall accuracy

# Correct ROC AUC for multi-label classification (use average=None to get per-class AUCs)
overall_auc_per_class = roc_auc_score(y_true, sigmoid_outputs, average=None)  # AUC for each class

# Print the metrics for each vulnerability type
for i, vuln_type in enumerate(vulnerability_types):
    print(f"Metrics for {vuln_type}:")
    print(f"  Accuracy: {overall_accuracy:.4f}")
    print(f"  AUC: {overall_auc_per_class[i]:.4f}")
    print(f"  F1-Score: {f1_per_class[i]:.4f}")
    print(f"  Precision: {precision_per_class[i]:.4f}")
    print(f"  Recall: {recall_per_class[i]:.4f}")
    print()

# Print the final overall metrics (macro-averaged across all classes)
f1_macro = f1_score(y_true, y_pred_best, average='macro')
precision_macro = precision_score(y_true, y_pred_best, average='macro')
recall_macro = recall_score(y_true, y_pred_best, average='macro')
accuracy = accuracy_score(y_true, y_pred_best)  # Calculate accuracy
auc = roc_auc_score(y_true, sigmoid_outputs, average='macro', multi_class='ovo')  # Calculate AUC

print(f"Overall Performance (Macro-Averaged):")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  AUC: {auc:.4f}")
print(f"  F1-Score: {f1_macro:.4f}")
print(f"  Precision: {precision_macro:.4f}")
print(f"  Recall: {recall_macro:.4f}")

# Optionally, show a detailed classification report for all classes
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred_best, target_names=vulnerability_types))
