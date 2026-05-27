import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess_data, split_data_for_clients
from federated_client import SimpleNet, FederatedClient
from differential_privacy_client import DPFederatedClient
from server import FederatedServer

def run_privacy_utility_benchmark():
    """Compare accuracy vs privacy across different noise multipliers"""
    
    print("=" * 70)
    print("PRIVACY-UTILITY TRADEOFF BENCHMARK")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading data...")
    X, y, scaler = load_and_preprocess_data('merged_data.csv')
    clients_data = split_data_for_clients(X, y, num_clients=10)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X.shape[1]
    
    # Test different noise multipliers
    noise_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0]
    results = {
        'noise_multiplier': [],
        'epsilon': [],
        'accuracy': []
    }
    
    print("\n[2/3] Running privacy-utility experiments...")
    
    for noise_mult in noise_multipliers:
        print(f"\nTesting noise_multiplier={noise_mult}...")
        
        # Initialize clients with DP
        clients = {}
        for client_id in range(10):
            model = SimpleNet(input_dim)
            client = DPFederatedClient(client_id, model, device)
            clients[client_id] = client
        
        # Train all clients
        epsilons = []
        for client_id, client in clients.items():
            X_train = clients_data[client_id]['X_train']
            y_train = clients_data[client_id]['y_train']
            
            if noise_mult == 0.0:
                # Non-private training (for baseline)
                fc = FederatedClient(client_id, SimpleNet(input_dim), device)
                fc.train_local_model(X_train, y_train, epochs=5)
                clients[client_id] = fc
                eps = float('inf')
            else:
                _, eps = client.train_with_dp(
                    X_train, y_train, epochs=5, 
                    noise_multiplier=noise_mult
                )
                epsilons.append(eps)
        
        # Evaluate
        accuracies = []
        for client_id, client in clients.items():
            X_test = clients_data[client_id]['X_test']
            y_test = clients_data[client_id]['y_test']
            acc = client.evaluate(X_test, y_test)
            accuracies.append(acc)
        
        avg_accuracy = np.mean(accuracies)
        avg_epsilon = np.mean(epsilons) if epsilons else float('inf')
        
        results['noise_multiplier'].append(noise_mult)
        results['epsilon'].append(avg_epsilon)
        results['accuracy'].append(avg_accuracy)
        
        print(f"  Epsilon: {avg_epsilon:.4f}, Accuracy: {avg_accuracy:.4f}")
    
    print("\n[3/3] Generating visualizations...")
    
    # Plot privacy-utility curve
    fig, ax = plt.subplots(figsize=(10, 6))
    
    valid_indices = [i for i, eps in enumerate(results['epsilon']) if eps != float('inf')]
    epsilons_plot = [results['epsilon'][i] for i in valid_indices]
    accuracies_plot = [results['accuracy'][i] for i in valid_indices]
    
    ax.plot(epsilons_plot, accuracies_plot, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Privacy Budget (Epsilon)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Privacy-Utility Tradeoff in Federated Learning', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add baseline (non-private)
    baseline_acc = results['accuracy'][0]
    ax.axhline(y=baseline_acc, color='red', linestyle='--', label=f'Non-private baseline: {baseline_acc:.4f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('privacy_utility_curve.png', dpi=300, bbox_inches='tight')
    print("Saved: privacy_utility_curve.png")
    
    return results

if __name__ == "__main__":
    results = run_privacy_utility_benchmark()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Save results to CSV
    df_results.to_csv('benchmark_results.csv', index=False)
    print("\nSaved: benchmark_results.csv")