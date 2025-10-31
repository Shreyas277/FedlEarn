import flwr as fl
import numpy as np
import os.path
import sys
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Parameters, FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

# --- Helper function (No change) ---
def calculate_parameter_change(params1: List[np.ndarray], params2: List[np.ndarray]) -> float:
    """Calculates the L1 norm of the difference between two parameter lists."""
    if params1 is None:
        # This happens on round 1 if no model was loaded
        return 0.0
    total_change = 0.0
    for layer_old, layer_new in zip(params1, params2):
        total_change += np.sum(np.abs(layer_old - layer_new))
    return total_change

# --- 1. Define our Custom Strategy (No change) ---
# This strategy is model-agnostic and works perfectly.
class CustomFedAvg(fl.server.strategy.FedAvg):

    def __init__(self, initial_parameters: Optional[Parameters], *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set the initial model
        self.initial_parameters = initial_parameters
        # 'previous_global_parameters' will store the *result* of this round
        self.previous_global_parameters: Optional[List[np.ndarray]] = None
        
        if initial_parameters is not None:
             print("[Strategy] Initial parameters loaded.")
             self.previous_global_parameters = parameters_to_ndarrays(initial_parameters)
        else:
             print("[Strategy] No initial parameters provided. Waiting for first client.")


    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        """Use the initial parameters we passed in."""
        print("[Strategy] initialize_parameters() called.")
        return self.initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        print(f"\n[Server (aggregate_fit)] Round {server_round}")
        print(f"  Received {len(results)} parameter updates from clients.")
        if not results:
            print("  No valid results. Skipping aggregation.")
            return None, {}

        # This is where the averaging happens
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            current_global_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            print("  Calculating change from previous global model...")

            # Compare to the model we started the round with
            change = calculate_parameter_change(
                self.previous_global_parameters,
                current_global_ndarrays
            )
            
            if server_round == 1 and self.previous_global_parameters is None:
                print("  (Round 1: This is the first model, no change to report)")
            else:
                print(f"  Total Parameter Change (L1 Norm): {change:.4f}")

            # Store the *result* of this round
            self.previous_global_parameters = current_global_ndarrays

        print("[Server (aggregate_fit)] Aggregation complete.")

        return aggregated_parameters, aggregated_metrics

# --- 2. Helper function to load the model (No change) ---
def load_saved_model(model_file: str) -> Optional[Parameters]:
    """Loads the .npz file if it exists."""
    if not os.path.exists(model_file):
        print(f"[Server] No saved model found at '{model_file}'.")
        print("[Server] CRITICAL: Run 'init_model.py' first to create one.")
        return None

    print(f"[Server] Found saved model '{model_file}'. Loading parameters.")
    try:
        loaded_npz = np.load(model_file)
        # Load all 'layer_i' arrays and put them back into a list
        param_list = [loaded_npz[key] for key in sorted(loaded_npz.files)]
        print(f"[Server] Loaded {len(param_list)} parameter arrays.")
        return ndarrays_to_parameters(param_list)
    except Exception as e:
        print(f"[Server] Error loading saved model: {e}. Exiting.")
        sys.exit(1)

# --- 3. The Main Server Loop ---
print("--- Starting PERMANENT Federated Learning Server ---")
MODEL_FILE = "final_model.npz"

# Load the model from disk (created by init_model.py)
latest_model_parameters = load_saved_model(MODEL_FILE)
if latest_model_parameters is None:
    print("Error: Model file not found. Please run 'init_model.py' first.")
    sys.exit(1)

try:
    while True:
        # 1. Create the strategy *inside* the loop, passing in the latest model
        strategy = CustomFedAvg(
            initial_parameters=latest_model_parameters,
            fraction_fit=1.0,        # Use 100% of connected clients
            fraction_evaluate=1.0,   # Use 100% of connected clients
            min_available_clients=2, # Wait for at least 2 clients
            min_fit_clients=2,       # Minimum 2 clients to train
            min_evaluate_clients=2,  # Minimum 2 clients to evaluate
        )

        print("\n[Server] Waiting for a batch of 2 clients to connect...")

        # 2. Start the server for ONE round
        # This is a *blocking call*. It will wait here until 1 round is done.
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=1), # Only EVER run 1 round
            strategy=strategy,
        )

        # 3. After the round, save the new model
        if strategy.previous_global_parameters is not None:
            print(f"[Server] Saving new aggregated model to '{MODEL_FILE}'...")

            # Update the latest model in memory
            latest_model_parameters = ndarrays_to_parameters(strategy.previous_global_parameters)

            # Save to disk
            params_dict_to_save = {f"layer_{i}": param for i, param in enumerate(strategy.previous_global_parameters)}
            np.savez(MODEL_FILE, **params_dict_to_save)
            print("[Server] Model saved successfully.")
        else:
            print("[Server] No model updates to save this round.")

except KeyboardInterrupt:
    print("\n[Server] Shutting down (Ctrl+C pressed)... Server is now offline.")

