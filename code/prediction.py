import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm
import pandas as pd

# Check CUDA availability and set device
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Logging configuration
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

class HateClassificationNN(nn.Module):
    def __init__(self, input_dim=768, learning_rate=1e-4, device='cpu'):
        super(HateClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4)
        self.fc3 = nn.Linear(input_dim // 4, 1)
        
        self.device = device
        self.to(device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCELoss().to(device)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, 0.6)
        x = F.leaky_relu(self.fc2(x))
        x = F.dropout(x, 0.6)
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()

    def fit(self, train_loader, valid_loader, results_file, epochs=20):
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)

            self.eval()
            valid_loss = 0.0
            predictions, true_labels = [], []
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                    outputs = self.forward(inputs)
                    loss = self.loss_fn(outputs, labels)
                    valid_loss += loss.item()
                    predictions.extend(outputs.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
                    
            valid_loss /= len(valid_loader)
            auc = roc_auc_score(true_labels, predictions)
            logging.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f} - AUC: {auc:.4f}")

        self.save_predictions(valid_loader, results_file)

    def save_predictions(self, valid_loader, results_file):
        self.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(self.device)
                outputs = self.forward(inputs)
                predictions.extend(outputs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        results_df = pd.DataFrame({'TrueLabel': true_labels, 'PredictedScore': predictions})
        results_df.to_csv(results_file, index=False)

if __name__ == '__main__':
    # Load embeddings and labels
    embeddings = torch.load('../data/body_embeddings.pt')
    labels = torch.load('../data/hate_labels.pt')

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=RandomSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=32, sampler=SequentialSampler(val_dataset))

    # Initialize and train model
    model = HateClassificationNN(device=dev)
    results_file = '../data/validation_results.csv'
    model.fit(train_loader, val_loader, results_file, epochs=20)
