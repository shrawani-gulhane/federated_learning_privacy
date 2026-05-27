import torch
from preprocessing import load_and_preprocess_data, split_data_for_clients
from federated_client import SimpleNet, FederatedClient

def main():
    # 1. Load and preprocess data
    print("=== Loading Data ===")
    X, y, scaler = load_and_preprocess_data('merged_data.csv')
    
    # 2. Split into client datasets
    print("\n=== Splitting Data for Federated Clients ===")
    clients_data = split_data_for_clients(X, y, num_clients=10)
    
    # 3. Initialize clients
    print("\n=== Initializing Federated Clients ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_dim = X.shape[1]
    clients = {}
    
    for client_id in range(10):
        model = SimpleNet(input_dim)
        client = FederatedClient(client_id, model, device)
        clients[client_id] = client
    
    # 4. Train locally (simulating federated round)
    print("\n=== Training Round 1 ===")
    for client_id, client in clients.items():
        X_train = clients_data[client_id]['X_train']
        y_train = clients_data[client_id]['y_train']
        
        print(f"Training client {client_id}...", end=" ")
        client.train_local_model(X_train, y_train, epochs=3)
        print("Done!")
    
    # 5. Evaluate
    print("\n=== Evaluating ===")
    for client_id, client in clients.items():
        X_test = clients_data[client_id]['X_test']
        y_test = clients_data[client_id]['y_test']
        
        accuracy = client.evaluate(X_test, y_test)
        print(f"Client {client_id} test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()