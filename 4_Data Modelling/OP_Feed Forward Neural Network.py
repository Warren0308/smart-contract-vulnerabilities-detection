import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os

# Set your working directory
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')

# Load the CSV file with embeddings and labels
df = pd.read_csv('3_Data Encoding/OP_train_final.csv')

# Define X (features) and y (labels)
X = df[[f'embedding_{i + 1}' for i in range(768)]].values  # 768-dimensional embeddings
y = df[['ARTHM', 'LE', 'None', 'RENT', 'TimeO']].values  # Multi-label binary targets

# Define a custom dataset class for loading embeddings and labels
class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'X': torch.tensor(self.X[idx], dtype=torch.float32),  # 768-dimensional feature vector
            'y': torch.tensor(self.y[idx], dtype=torch.float32)  # Multi-label binary target
        }

# Define the MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=512, num_labels=5, dropout=0.1):
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

# Function to train and evaluate the model with various hyperparameters
def train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=60):
    best_f1 = 0.0  # Track the best F1 score
    best_precision = 0.0  # Track the best precision
    best_recall = 0.0  # Track the best recall
    best_thresholds = [0.5] * 5  # Track the best threshold for each of the 5 vulnerability types
    best_model_state = None  # To store the best model state

    model.train()  # Set model to training mode
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            X_batch = batch['X'].to(device)
            y_batch = batch['y'].to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Adjust the learning rate
        scheduler.step()

    # After training, evaluate on the validation set
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_preds = []
    sigmoid_outputs_list = []

    with torch.no_grad():
        for batch in val_loader:
            X_batch = batch['X'].to(device)
            y_batch = batch['y'].to(device)

            outputs = model(X_batch)
            sigmoid_outputs = torch.sigmoid(outputs).detach().cpu().numpy()  # Convert to probabilities
            sigmoid_outputs_list.append(sigmoid_outputs)
            all_labels.append(y_batch.cpu().numpy())

    # Concatenate all labels and predictions
    y_true = np.concatenate(all_labels, axis=0)  # Shape: (num_samples, 5)
    sigmoid_outputs = np.concatenate(sigmoid_outputs_list, axis=0)  # Shape: (num_samples, 5)

    # Tune thresholds for each vulnerability type (each of the 5 labels)
    num_labels = y_true.shape[1]  # Number of labels (5)
    best_thresholds = [0.5] * num_labels  # Initialize best thresholds for each label
    for label_idx in range(num_labels):
        best_f1_label = 0.0  # Track best F1 score for this label

        for threshold in np.arange(0.1, 0.9, 0.02):
            y_pred = np.copy(sigmoid_outputs)  # Copy of sigmoid outputs
            y_pred[:, label_idx] = (sigmoid_outputs[:, label_idx] > threshold).astype(int)  # Apply threshold to this label

            # Calculate metrics only for this label
            f1 = f1_score(y_true[:, label_idx], y_pred[:, label_idx], zero_division=0)
            precision = precision_score(y_true[:, label_idx], y_pred[:, label_idx], zero_division=0)
            recall = recall_score(y_true[:, label_idx], y_pred[:, label_idx], zero_division=0)

            # If F1 improves, save the threshold for this label
            if f1 > best_f1_label:
                best_f1_label = f1
                best_thresholds[label_idx] = threshold

    # After tuning, calculate overall F1, precision, and recall using the best thresholds
    y_pred_best = np.copy(sigmoid_outputs)
    for label_idx in range(num_labels):
        y_pred_best[:, label_idx] = (sigmoid_outputs[:, label_idx] > best_thresholds[label_idx]).astype(int)

    overall_f1 = f1_score(y_true, y_pred_best, average='macro', zero_division=0)
    overall_precision = precision_score(y_true, y_pred_best, average='macro', zero_division=0)
    overall_recall = recall_score(y_true, y_pred_best, average='macro', zero_division=0)
    overall_accuracy = accuracy_score(y_true, y_pred_best)  # Calculate accuracy
    overall_auc = roc_auc_score(y_true, sigmoid_outputs, average='macro', multi_class='ovo')  # Calculate AUC

    return overall_f1, overall_precision, overall_recall, overall_accuracy, overall_auc, best_thresholds, model.state_dict()

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters to try
learning_rates = [1e-3, 1e-4]
batch_sizes = [64, 128]
hidden_sizes = [512, 1024]
dropout_rates = [0.1, 0.3]
epochs = 60  # Fixed number of epochs

best_accuracy = 0.0  # Track the best accuracy
best_params = {}  # Track the best hyperparameters
checkpoint = {}  # Save the best model state

# Cross-Validation Setup
kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Loop over all combinations of hyperparameters
for lr in learning_rates:
    for batch_size in batch_sizes:
        for hidden_size in hidden_sizes:
            for dropout in dropout_rates:
                print(f"Training with lr={lr}, batch_size={batch_size}, hidden_size={hidden_size}, dropout={dropout}")

                # Initialize lists to store metrics for each fold
                accuracy_scores = []

                # Cross-Validation Loop
                for train_index, val_index in kf.split(X, y):
                    # Split the data
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    # Create DataLoader objects
                    train_dataset = EmbeddingDataset(X_train, y_train)
                    val_dataset = EmbeddingDataset(X_val, y_val)

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                    # Initialize the model, optimizer, scheduler, and loss function
                    model = MLPClassifier(input_size=768, hidden_size=hidden_size, num_labels=5, dropout=dropout).to(device)
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
                    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

                    # Train and evaluate the model on this fold
                    _, _, _, accuracy, _, _, _ = train_and_evaluate_model(
                        model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=epochs)

                    # Store metrics for this fold
                    accuracy_scores.append(accuracy)

                # Calculate mean accuracy over all folds
                mean_accuracy = np.mean(accuracy_scores)

                print(f"Mean Accuracy for lr={lr}, batch_size={batch_size}, hidden_size={hidden_size}, dropout={dropout}: {mean_accuracy:.4f}")

                # Save the best model
                if mean_accuracy > best_accuracy:
                    best_accuracy = mean_accuracy
                    best_params = {
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'hidden_size': hidden_size,
                        'dropout': dropout
                    }
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'best_accuracy': best_accuracy
                    }

# Save the best model
torch.save(checkpoint, "4_Data Modelling/OP_Model.pth")

# Print the best hyperparameters and results
print("\nBest Hyperparameters:")
print(f"Learning Rate: {best_params['learning_rate']}")
print(f"Batch Size: {best_params['batch_size']}")
print(f"Hidden Size: {best_params['hidden_size']}")
print(f"Dropout: {best_params['dropout']}")
print(f"Best Accuracy: {best_accuracy:.4f}")
