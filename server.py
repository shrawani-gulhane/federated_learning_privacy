import torch
import copy

class FederatedServer:
    def __init__(self, model):
        self.global_model = copy.deepcopy(model)
        self.round = 0
    
    def aggregate(self, client_models):
        """Average model weights from all clients (FedAvg)"""
        
        global_state = self.global_model.state_dict()
        
        # Initialize aggregated weights to zeros
        aggregated_state = {key: torch.zeros_like(param) 
                           for key, param in global_state.items()}
        
        # Average weights from all clients
        num_clients = len(client_models)
        for client_model in client_models:
            client_state = client_model.state_dict()
            for key, param in client_state.items():
                aggregated_state[key] += param / num_clients
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
        self.round += 1
        
        return self.global_model.state_dict()
    
    def get_global_model(self):
        """Return current global model"""
        return copy.deepcopy(self.global_model)