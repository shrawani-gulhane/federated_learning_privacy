import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class FederatedClient:
    def __init__(self, client_id, model, device):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.privacy_engine = None
    
    def train_local_model(self, X_train, y_train, epochs=5, batch_size=16, dp_epsilon=None):
        """Train model locally with optional differential privacy"""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train.values).to(self.device)
        y_tensor = torch.FloatTensor(y_train.values.reshape(-1, 1)).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        # Setup DP if epsilon provided
        if dp_epsilon is not None:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=loader,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
            )
        
        # Training loop
        criterion = nn.BCELoss()
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
        return self.model.state_dict()
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        
        X_tensor = torch.FloatTensor(X_test.values).to(self.device)
        y_tensor = torch.FloatTensor(y_test.values.reshape(-1, 1)).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_tensor).float().mean().item()
        
        return accuracy