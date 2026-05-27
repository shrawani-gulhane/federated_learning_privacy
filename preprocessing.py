import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    """Load MIMIC data and prepare for federated learning"""
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    
    # Drop subject_id (not a feature)
    X = df.drop(['subject_id', 'hospital_expire_flag'], axis=1)
    y = df['hospital_expire_flag']
    
    # Handle missing values (mean imputation)
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Processed shape: {X_scaled.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X_scaled, y, scaler

def split_data_for_clients(X, y, num_clients=10):
    """Split data into heterogeneous client datasets"""
    
    clients_data = {}
    
    # Split by patient (simulating different hospitals)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Create heterogeneous splits (different sizes)
    split_sizes = np.random.dirichlet(np.ones(num_clients)) * len(X)
    split_sizes = split_sizes.astype(int)
    
    start_idx = 0
    for client_id in range(num_clients):
        end_idx = start_idx + split_sizes[client_id]
        client_indices = indices[start_idx:end_idx]
        
        X_client = X.iloc[client_indices]
        y_client = y.iloc[client_indices]
        
        # Split into train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=0.2, random_state=42
        )
        
        clients_data[client_id] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'size': len(X_client)
        }
        
        print(f"Client {client_id}: {len(X_client)} samples (pos: {y_client.sum()}, neg: {len(y_client) - y_client.sum()})")
        start_idx = end_idx
    
    return clients_data

if __name__ == "__main__":
    X, y, scaler = load_and_preprocess_data('merged_data.csv')
    clients_data = split_data_for_clients(X, y, num_clients=10)
    print("\nData preprocessing complete!")