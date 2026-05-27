import torch
import pandas as pd
import numpy as np
from preprocessing import load_and_preprocess_data, split_data_for_clients
from federated_client import SimpleNet, FederatedClient
from server import FederatedServer

def run_federated_learning(num_rounds=5, num_clients=10):
    """Run complete federated learning pipeline"""
    
    print("=" * 60)
    print("FEDERATED LEARNING PIPELINE")
    print("=" * 60)
    
    # 1. Load and preprocess
    print("\n[1/4] Loading and preprocessing data...")
    X, y, scaler = load_and_preprocess_data('merged_data.csv')
    
    # 2. Split into clients
    print("[2/4] Splitting data into", num_clients, "heterogeneous clients...")
    clients_data = split_data_for_clients(X, y, num_clients=num_clients)
    
    # 3. Initialize server and clients
    print("[3/4] Initializing federated server and clients...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Using device: {device}")
    
    input_dim = X.shape[1]
    initial_model = SimpleNet(input_dim)
    server = FederatedServer(initial_model)
    
    clients = {}
    for client_id in range(num_clients):
        model = SimpleNet(input_dim)
        client = FederatedClient(client_id, model, device)
        clients[client_id] = client
    
    # 4. Federated training rounds
    print("[4/4] Running federated training rounds...")
    print()
    
    results = {
        'round': [],
        'global_accuracy': [],
        'client_accuracies': []
    }
    
    for round_num in range(num_rounds):
        print(f"--- ROUND {round_num + 1}/{num_rounds} ---")
        
        # Send global model to all clients
        global_weights = server.get_global_model().state_dict()
        for client in clients.values():
            client.model.load_state_dict(global_weights)
        
        # Local training
        trained_models = []
        for client_id, client in clients.items():
            X_train = clients_data[client_id]['X_train']
            y_train = clients_data[client_id]['y_train']
            client.train_local_model(X_train, y_train, epochs=3)
            trained_models.append(client.model)
        
        # Server aggregation
        server.aggregate(trained_models)
        
        # Evaluate on all clients
        accuracies = []
        for client_id, client in clients.items():
            X_test = clients_data[client_id]['X_test']
            y_test = clients_data[client_id]['y_test']
            acc = client.evaluate(X_test, y_test)
            accuracies.append(acc)
        
        avg_accuracy = np.mean(accuracies)
        print(f"Average test accuracy: {avg_accuracy:.4f}")
        print(f"Min accuracy: {min(accuracies):.4f}, Max accuracy: {max(accuracies):.4f}")
        print()
        
        # Store results
        results['round'].append(round_num + 1)
        results['global_accuracy'].append(avg_accuracy)
        results['client_accuracies'].append(accuracies)
    
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return results, server, clients, clients_data

if __name__ == "__main__":
    results, server, clients, clients_data = run_federated_learning(num_rounds=5, num_clients=10)
    
    # Print final results
    print("\nFINAL RESULTS:")
    for i, round_num in enumerate(results['round']):
        print(f"Round {round_num}: {results['global_accuracy'][i]:.4f}")