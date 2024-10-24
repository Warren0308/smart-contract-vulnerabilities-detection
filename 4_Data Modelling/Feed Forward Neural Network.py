import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import StandardScaler
import os

# Set your working directory
os.chdir('/Users/warren/PycharmProjects/smart-contract-vulnerabilities-detection/')

# Load the CSV file with embeddings and labels
df = pd.read_csv('3_Data Encoding/SC_train_final.csv')

# Define X (features) and y (labels)
X = df[[f'embedding_{i + 1}' for i in range(768)]].values  # 768-dimensional embeddings
y = df[['ARTHM', 'LE', 'None', 'RENT', 'TimeO']].values  # Multi-label binary targets

# Scale the features (StandardScaler for normalization)
scaler = StandardScaler()
X = scaler.fit_transform(X)


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


# Define the MLP Classifier with Batch Normalization
class MLPClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=512, num_labels=5, dropout=0.1):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)  # Batch Normalization layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)  # Apply BatchNorm after the first fully connected layer
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Function to train and evaluate the model with various hyperparameters
def train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=60):
    best_f1 = 0.0  # Track the best F1 score
    no_improvement_epochs = 0  # Early stopping counter
    early_stopping_patience = 10  # Stop training if no improvement after 10 epochs

    for epoch in range(epochs):
        model.train()  # Set model to training mode
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

        # Evaluate on the validation set after each epoch
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
        y_true = np.concatenate(all_labels, axis=0)
        sigmoid_outputs = np.concatenate(sigmoid_outputs_list, axis=0)

        # Calculate F1 score using default thresholds
        y_pred_best = (sigmoid_outputs > 0.5).astype(int)
        overall_f1 = f1_score(y_true, y_pred_best, average='macro', zero_division=0)

        # Early stopping based on F1 score
        if overall_f1 > best_f1:
            best_f1 = overall_f1
            no_improvement_epochs = 0  # Reset the counter if F1 improves
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        # Adjust the learning rate based on validation performance
        scheduler.step(overall_f1)

    return best_f1


# Main entry point to avoid multiprocessing error
if __name__ == '__main__':
    # Set device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters to try
    learning_rates = [1e-3, 1e-4]
    batch_sizes = [64, 128]
    hidden_sizes = [512, 1024]
    dropout_rates = [0.1, 0.3]
    epochs = 60

    best_f1 = 0.0  # Track the best F1 score across all hyperparameter combinations
    best_params = {}  # Track the best hyperparameters
    checkpoint = {}  # Save the best model state

    # Cross-Validation Setup
    kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Loop over all combinations of hyperparameters
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for hidden_size in hidden_sizes:
                for dropout in dropout_rates:
                    print(
                        f"Training with lr={lr}, batch_size={batch_size}, hidden_size={hidden_size}, dropout={dropout}")

                    accuracy_scores = []  # To store accuracy for each fold

                    for train_index, val_index in kf.split(X, y):
                        # Split the data
                        X_train, X_val = X[train_index], X[val_index]
                        y_train, y_val = y[train_index], y[val_index]

                        # Create DataLoader objects
                        train_dataset = EmbeddingDataset(X_train, y_train)
                        val_dataset = EmbeddingDataset(X_val, y_val)

                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                  num_workers=4)  # num_workers > 0 triggers multiprocessing
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

                        # Initialize the model, optimizer, scheduler, and loss function
                        model = MLPClassifier(input_size=768, hidden_size=hidden_size, num_labels=5,
                                              dropout=dropout).to(device)
                        pos_weight = torch.tensor(np.sum(y_train == 0, axis=0) / np.sum(y_train == 1, axis=0),
                                                  dtype=torch.float32).to(device)
                        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Handle class imbalance
                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 regularization
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

                        # Train and evaluate the model on this fold
                        fold_f1 = train_and_evaluate_model(
                            model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=epochs
                        )

                        accuracy_scores.append(fold_f1)

                    # Calculate mean F1 over all folds
                    mean_f1 = np.mean(accuracy_scores)

                    print(
                        f"Mean F1 Score for lr={lr}, batch_size={batch_size}, hidden_size={hidden_size}, dropout={dropout}: {mean_f1:.4f}")

                    # Save the best model
                    if mean_f1 > best_f1:
                        best_f1 = mean_f1
                        best_params = {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'hidden_size': hidden_size,
                            'dropout': dropout
                        }
                        checkpoint = {
                            'model_state_dict': model.state_dict(),
                            'best_f1': best_f1,
                            'best_params': best_params  # Save hyperparameters
                        }

    # Save the best model and hyperparameters
    torch.save(checkpoint, "4_Data Modelling/SC_model.pth")

    # Print the best hyperparameters and results
    print("\nBest Hyperparameters:")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"Batch Size: {best_params['batch_size']}")
    print(f"Hidden Size: {best_params['hidden_size']}")
    print(f"Dropout: {best_params['dropout']}")
    print(f"Best F1 Score: {best_f1:.4f}")
