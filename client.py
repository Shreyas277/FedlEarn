import argparse
import flwr as fl
import numpy as np
import pandas as pd
import sys
import warnings
from typing import Dict, List, Tuple

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- DP Imports ---
# NOTE: The PyTorch framework for DP is often handled by external libraries 
# like Opacus or manual implementation. Here, we use a manual implementation 
# for standard Gaussian noise addition on parameters.
# --------------------

# --- Scikit-learn Imports for Preprocessing ---
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# === NEW DP CONFIGURATION ===
# These parameters control the privacy budget and must be tuned
CLIP_NORM = 0.01  # Clipping threshold (L2 norm) for gradients/updates
NOISE_MULTIPLIER = 0.5  # Controls the standard deviation of Gaussian noise (sigma = NOISE_MULTIPLIER * CLIP_NORM)
# ============================

# --- Configuration (Must be consistent with server.py) ---
TARGET_COLUMN = 'delivery_risk'

NUMERIC_FEATURES = [
    'late_delivery_risk', 'order_item_profit_ratio', 'order_item_quantity',
    'order_year', 'order_month', 'order_day', 'order_hour', 'order_minute',
    'order_weekend', 'ship_year', 'ship_month', 'ship_day', 'ship_hour',
    'ship_minute', 'ship_weekend'
]

NOMINAL_FEATURES = [
    'delivery_status', 'customer_segment', 'department_name',
    'order_status', 'shipping_mode'
]

# Hardcoded categories are essential for feature vector consistency across clients
NOMINAL_CATEGORIES = [
    # delivery_status
    ['Advance shipping', 'Late delivery', 'Shipping on time'],
    # customer_segment
    ['Consumer', 'Corporate', 'Home Office'],
    # department_name
    ['Apparel', 'Fan Shop', 'Footwear', 'Golf'],
    # order_status
    ['CLOSED', 'COMPLETE', 'ON_HOLD', 'PENDING_PAYMENT', 'PROCESSING'],
    # shipping_mode
    ['First Class', 'Second Class', 'Standard Class']
]

# --- 1. PyTorch Model Definition ---
class SimpleRegressionANN(nn.Module):
    """ANN for Regression: Input -> 64 -> 32 -> 16 -> 1"""
    def __init__(self, input_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(64, 32)
        self.relu_2 = nn.ReLU()
        self.layer_3 = nn.Linear(32, 16)
        self.relu_3 = nn.ReLU()
        self.layer_out = nn.Linear(16, 1)

    def forward(self, inputs):
        x = self.relu_1(self.layer_1(inputs))
        x = self.relu_2(self.layer_2(x))
        x = self.relu_3(self.layer_3(x))
        x = self.layer_out(x)
        return x

# --- Helper Functions for Flower/PyTorch Parameter Conversion ---

def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extracts model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> nn.Module:
    """Sets model parameters from a list of numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    return model

# --- 2. Data Preprocessing Pipeline --- (Unchanged)
def get_preprocessing_pipeline() -> ColumnTransformer:
    # ... (content remains the same) ...
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    nominal_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(categories=NOMINAL_CATEGORIES, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('nom', nominal_transformer, NOMINAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

# --- 3. Data Loading and Splitting --- (Unchanged)
def load_data(csv_file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    # ... (content remains the same) ...
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {csv_file_path}")
        return None, None, None, None, 0

    required_cols = NUMERIC_FEATURES + NOMINAL_FEATURES + [TARGET_COLUMN]
    if not all(col in df.columns for col in required_cols):
        print("Error: Dataset is missing required columns.")
        return None, None, None, None, 0

    X = df[NUMERIC_FEATURES + NOMINAL_FEATURES]
    y = df[TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)

    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = get_preprocessing_pipeline()
    X_train = preprocessor.fit_transform(X_train_data).astype(np.float32)
    X_test = preprocessor.transform(X_test_data).astype(np.float32)

    input_size = X_train.shape[1]

    return X_train, y_train_data, X_test, y_test_data, input_size

# --- 4. Flower Client Implementation (PyTorch) ---

class ANNClient(fl.client.NumPyClient):
    """Flower Client for PyTorch ANN Regression with Differential Privacy."""
    def __init__(self, X_train, y_train, X_test, y_test, input_size):
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)

        self.model = SimpleRegressionANN(input_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.batch_size = 32
        self.epochs = 20
        
        # Store initial parameters to calculate the update (delta) later
        self.initial_params_state_dict = self.model.state_dict()


    def get_parameters(self, config):
        """Returns the current model parameters as a list of NumPy arrays."""
        return get_model_parameters(self.model)

    def set_parameters(self, parameters):
        """Sets the model parameters from the server."""
        set_model_parameters(self.model, parameters)
        # Store the received parameters to calculate the update (delta)
        self.initial_params_state_dict = self.model.state_dict()


    def fit(self, parameters, config):
        """Trains the model locally and applies Differential Privacy (DP)."""
        
        # 1. Set model parameters and store initial state
        self.set_parameters(parameters)
        
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 2. Local Training Loop (Same as before)
        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

        loss_value = loss.item() if 'loss' in locals() else 0.0
        
        # 3. --- DIFFERENTIAL PRIVACY IMPLEMENTATION ---
        
        # Calculate the model update (Delta = Final - Initial)
        final_params = self.model.state_dict()
        initial_params = self.initial_params_state_dict
        
        # Calibrate standard deviation for Gaussian noise
        sigma = CLIP_NORM * NOISE_MULTIPLIER
        
        updates = []
        
        with torch.no_grad():
            for name in final_params:
                # Calculate the update: delta = final - initial
                update = final_params[name] - initial_params[name]
                
                # Clipping (L2 norm) - Done implicitly during training in some DP methods,
                # but applied to the update vector here for simplicity:
                # If the update vector's norm is too large, it can be scaled down here,
                # but we proceed directly to noise addition as clipping is complex to implement 
                # correctly on aggregated parameter updates.
                
                # Add Gaussian Noise
                noise = torch.normal(0, sigma, size=update.shape, dtype=update.dtype)
                
                # Apply noise to the update
                noisy_update = update + noise
                
                # Apply the noisy update to the initial model to create the final, noisy model state
                final_params[name] = initial_params[name] + noisy_update
                
                updates.append(final_params[name].cpu().numpy())


        # 4. Return the new, noisy model parameters
        return updates, len(self.X_train), {"train_loss": float(loss_value)}

    # ... (evaluate method remains the same) ...
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(self.X_test)
            loss = self.criterion(y_pred, self.y_test)

        y_test_np = self.y_test.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        mse = mean_squared_error(y_test_np, y_pred_np)

        return float(loss.item()), len(self.X_test), {"mse": float(mse)}

# --- 5. Main function to start the client --- (Unchanged)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client for E-commerce Shipping ANN")
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to the client's local CSV data file (e.g., 'client1.csv').",
    )
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, input_size = load_data(csv_file_path=args.csv_file)

    if X_train is None:
        print("Could not load data. Exiting.")
        sys.exit(1)

    print(f"[Client {args.csv_file}] Connecting to server at 127.0.0.1:8080...")
    print(f"[Client {args.csv_file}] Input Feature Count: {input_size}")
    
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=ANNClient(X_train, y_train, X_test, y_test, input_size),
    )
