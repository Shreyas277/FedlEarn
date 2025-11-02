import flwr as fl
import numpy as np
import os.path
import sys
import torch
import torch.nn as nn
import pandas as pd
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Parameters, FitRes, Scalar
from flwr.server.client_proxy import ClientProxy
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# === Configuration (Must be consistent with client.py) ===
MODEL_FILE_FINAL = "final_ann_model.npz" 
DATA_FILE_TO_INFER_SHAPE = "global.csv" 

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
# --- End Configuration ---

# --- PyTorch Model Definition ---
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

def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extracts model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# --- Preprocessing Pipeline for feature count inference ---

def get_preprocessing_pipeline() -> ColumnTransformer:
    """Creates the static preprocessing pipeline."""
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

def get_input_size_and_initial_parameters() -> Tuple[int, Parameters]:
    """Infers the final feature vector size and gets initial PyTorch parameters."""
    try:
        # Load the data to fit the preprocessor and infer feature count
        df = pd.read_csv(DATA_FILE_TO_INFER_SHAPE)
        X = df[NUMERIC_FEATURES + NOMINAL_FEATURES]
        
        preprocessor = get_preprocessing_pipeline()
        # NOTE: Fitting here is ONLY to infer the final column count (input_size)
        X_dummy = preprocessor.fit_transform(X).astype(np.float32)
        input_size = X_dummy.shape[1]
        
        # Instantiate the PyTorch model with the correct size
        initial_model = SimpleRegressionANN(input_size)
        
        print(f"[Server] Inferred Input Size: {input_size} features.")

        # Try loading a previously saved model if it exists
        if os.path.exists(MODEL_FILE_FINAL):
            print(f"[Server] Found existing final model file: {MODEL_FILE_FINAL}. Loading parameters.")
            loaded = np.load(MODEL_FILE_FINAL, allow_pickle=True)
            initial_params = [loaded[key] for key in sorted(loaded.files, key=lambda x: int(x.split('_')[-1]))]
        else:
            print("[Server] No saved final model found. Using randomly initialized parameters.")
            initial_params = get_model_parameters(initial_model)

        return input_size, ndarrays_to_parameters(initial_params)

    except Exception as e:
        print(f"[Server] Error during model initialization/shape inference: {e}")
        sys.exit(1)


# --- Flower Strategy and Server Logic ---

class CustomFedAvg(fl.server.strategy.FedAvg):
    """Custom FedAvg to store the global model parameters after each round."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_global_parameters = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregates model parameters and stores them."""
        
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        self.previous_global_parameters = aggregated_parameters
        
        # Pass the current round number back in the metrics (optional, but good for debugging)
        metrics["current_round"] = server_round 
        
        return aggregated_parameters, metrics

# === Main Server Loop ===

def save_model_params(parameters: Parameters, round_number: int):
    """Saves model parameters to both a sequential file and the final file."""
    ndarrays = parameters_to_ndarrays(parameters)
    params_dict_to_save = {f"layer_{i}": param for i, param in enumerate(ndarrays)}
    
    # Sequential filename (e.g., model_upto1.npz)
    sequential_filename = f"model_upto{round_number}.npz"
    np.savez(sequential_filename, **params_dict_to_save)
    print(f"[Server] Aggregated model saved sequentially to '{sequential_filename}'.")
    
    # Overwrite the final filename (checkpointing)
    np.savez(MODEL_FILE_FINAL, **params_dict_to_save)
    print(f"[Server] Final model checkpoint updated to '{MODEL_FILE_FINAL}'.")


def main_server_loop():
    """Starts the Flower server loop with sequential saving."""
    input_size, latest_model_parameters = get_input_size_and_initial_parameters()
    
    round_count = 0 
    
    try:
        while True:
            round_count += 1
            
            # 1. Initialize the strategy with the latest parameters
            # FIX: Removed the invalid 'fit_config' argument
            strategy = CustomFedAvg(
                initial_parameters=latest_model_parameters,
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_available_clients=2,
                min_fit_clients=2,
                min_evaluate_clients=2,
            )

            print(f"\n[Server] Starting Round {round_count}. Waiting for 2 clients to connect...")

            # 2. Start the server for ONE round
            fl.server.start_server(
                server_address="0.0.0.0:8080",
                config=fl.server.ServerConfig(num_rounds=1),
                strategy=strategy,
            )

            # 3. After the round, save the new model
            if strategy.previous_global_parameters is not None:
                latest_model_parameters = strategy.previous_global_parameters
                
                # Save the model using the current round count
                save_model_params(latest_model_parameters, round_count)
                
            else:
                print(f"[Server] Round {round_count} finished, but no model updates to save.")

    except KeyboardInterrupt:
        print("\n[Server] Shutting down (Ctrl+C pressed)... Server is now offline.")
    except Exception as e:
        print(f"\n[Server] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main_server_loop()
