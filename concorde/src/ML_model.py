#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP model for CDF vector prediction using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
from .analysis import generate_cdf_vectors

# Try to import torch_npu, fallback to cpu if not available
try:
    import torch_npu 
    from torch_npu.contrib import transfer_to_npu
    torch_npu.npu.set_compile_mode(jit_compile=False)
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron for CDF vector input.
    
    Args:
        input_dim: Dimension of input CDF vector (default: 50 for quantiles 0.00 to 0.98)
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (for regression or classification)
        dropout_rate: Dropout probability
        use_batch_norm: Whether to use batch normalization
        activation: Activation function ('relu', 'tanh', 'elu')
    """
    def __init__(self, input_dim=50, hidden_dims=[128, 64, 32], output_dim=1,
                 dropout_rate=0.2, use_batch_norm=True, activation='relu'):
        super(MLPModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.model(x)


class CDFVectorDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for CDF vectors.
    
    Args:
        cdf_vectors: Dict from generate_cdf_vectors {name: array}
        labels: Array or dict of labels corresponding to each CDF vector
        label_key: If labels is dict, use this key as label
    """
    def __init__(self, cdf_vectors, labels=None, label_key=None):
        self.names = list(cdf_vectors.keys())
        self.vectors = [cdf_vectors[name] for name in self.names]
        
        if labels is None:
            # For unsupervised or when labels will be added later
            self.labels = np.zeros(len(self.names))
        elif isinstance(labels, dict):
            if label_key:
                self.labels = [labels[name][label_key] for name in self.names]
            else:
                self.labels = [labels[name] for name in self.names]
        else:
            self.labels = labels
        
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        vector = torch.FloatTensor(self.vectors[idx])
        label = torch.FloatTensor([self.labels[idx]])
        return vector, label, self.names[idx]


class MLPTrainer:
    """
    Trainer class for MLP model.
    
    Args:
        model: MLPModel instance
        device: Device to train on ('npu' or 'cpu')
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization strength
    """
    def __init__(self, model, device='auto', learning_rate=0.001, weight_decay=1e-5):
        self.model = model
        
        # Determine device
        if device == 'auto':
            if HAS_NPU:
                self.device = 'npu:0'
            elif torch.cuda.is_available():
                self.device = 'cuda:0'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # For regression tasks
        self.criterion_regression = nn.MSELoss()
        # For classification tasks
        self.criterion_classification = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader, task='regression'):
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            task: 'regression' or 'classification'
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for vectors, labels, names in dataloader:
            # Move to device
            vectors = vectors.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(vectors)
            
            # Compute loss
            if task == 'regression':
                loss = self.criterion_regression(outputs, labels)
            else:
                labels = labels.long().squeeze()
                loss = self.criterion_classification(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, dataloader, task='regression'):
        """
        Evaluate model on validation/test data.
        
        Args:
            dataloader: DataLoader for evaluation data
            task: 'regression' or 'classification'
            
        Returns:
            tuple: (average_loss, predictions, labels, names)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        all_names = []
        
        with torch.no_grad():
            for vectors, labels, names in dataloader:
                # Move to device
                vectors = vectors.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(vectors)
                
                # Compute loss
                if task == 'regression':
                    loss = self.criterion_regression(outputs, labels)
                else:
                    labels_long = labels.long().squeeze()
                    loss = self.criterion_classification(outputs, labels_long)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_names.extend(names)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, all_predictions, all_labels, all_names
    
    def predict(self, dataloader):
        """
        Predict on new data.
        
        Args:
            dataloader: DataLoader for prediction data
            
        Returns:
            tuple: (predictions, names)
        """
        self.model.eval()
        all_predictions = []
        all_names = []
        
        with torch.no_grad():
            for vectors, _, names in dataloader:
                # Move to device
                vectors = vectors.to(self.device)
                
                # Forward pass
                outputs = self.model(vectors)
                
                # Store predictions
                all_predictions.extend(outputs.cpu().numpy())
                all_names.extend(names)
                
        return all_predictions, all_names
    
    def save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def create_dataloaders(cdf_vectors, labels=None, batch_size=32, 
                       train_ratio=0.8, shuffle=True, label_key=None):
    """
    Create train and validation dataloaders from CDF vectors.
    
    Args:
        cdf_vectors: Dict from generate_cdf_vectors
        labels: Labels for supervised learning
        batch_size: Batch size for dataloaders
        train_ratio: Ratio of data to use for training
        shuffle: Whether to shuffle data
        label_key: Key for label in labels dict
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    dataset = CDFVectorDataset(cdf_vectors, labels, label_key)
    
    # Split into train and validation
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Example: Create dummy CDF vectors
    dummy_cdf_vectors = {
        f"series_{i}": np.random.rand(50) for i in range(100)
    }
    dummy_labels = np.random.rand(100)  # Regression labels
    
    # Create model
    model = MLPModel(
        input_dim=50,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        dropout_rate=0.2,
        use_batch_norm=True,
        activation='relu'
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dummy_cdf_vectors,
        labels=dummy_labels,
        batch_size=32,
        train_ratio=0.8
    )
    
    # Create trainer
    trainer = MLPTrainer(model, device='npu', learning_rate=0.001)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader, task='regression')
        val_loss, preds, labels, names = trainer.evaluate(val_loader, task='regression')
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Save model
    trainer.save_model("mlp_model.pth")
