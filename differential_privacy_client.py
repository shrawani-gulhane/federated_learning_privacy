import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

class DPFederatedClient:
    def __init__(self, client_id, model, device):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.privacy_engine = None
        self.epsilon = None
        self.delta = None
    
    def train_with_dp(self, X_train, y_train, epochs=5, batch_size=16, 
                      noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5):
        """Train model with differential privacy"""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train.values).to(self.device)
        y_tensor = torch.FloatTensor(y_train.values.reshape(-1, 1)).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        # Setup PrivacyEngine for differential privacy
        self.privacy_engine = PrivacyEngine()
        
        self.model, self.optimizer, loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
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
        
        # Get privacy budget
        epsilon = self.privacy_engine.accountant.get_epsilon(delta=delta)
        self.epsilon = epsilon
        self.delta = delta
        
        return self.model.state_dict(), epsilon
    
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