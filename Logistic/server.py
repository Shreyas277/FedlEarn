import flwr as fl
import numpy as np
import os.path
import sys
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Parameters, FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

# --- Add imports from init_model.py ---
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
# --------------------------------------

warnings.filterwarnings("ignore", category=UserWarning)

# === START OF LOGIC FROM init_model.py ===

# --- Configuration ---
MODEL_FILE = "final_model.npz"
DATA_FILE_TO_INFER_SHAPE = "train.csv" # Using train.csv is more robust

# These lists must be manually kept in sync with the dataset
NUMERIC_FEATURES = ['Cost_of_the_Product', 'Customer_care_calls', 'Weight_in_gms']
ORDINAL_FEATURES = ['Product_importance']
ORDINAL_CATEGORIES = [['low', 'medium', 'high']]
NOMINAL_FEATURES = ['Warehouse_block', 'Mode_of_Shipment']
NOMINAL_CATEGORIES = [
    ['A', 'B', 'C', 'D', 'F'],
    ['Ship', 'Flight', 'Road']
]
# --- End Configuration ---

def get_preprocessing_pipeline() -> ColumnTransformer:
    """
    Creates the static preprocessing pipeline.
    Hardcoded categories are ESSENTIAL for federated learning
    to ensure all clients produce identical feature vectors.
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(categories=ORDINAL_CATEGORIES))
    ])

    nominal_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(categories=NOMINAL_CATEGORIES, handle_unknown='ignore'))
    ])

    # Create the master preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('ord', ordinal_transformer, ORDINAL_FEATURES),
            ('nom', nominal_transformer, NOMINAL_FEATURES),
        ],
        remainder='drop' # Drop any columns not specified
    )
    return preprocessor

def create_initial_model_file():
    """
    Runs the model initialization logic.
    """
    print(f"--- [Server] No '{MODEL_FILE}' found. Running Initializer ---")
    print(f"Loading '{DATA_FILE_TO_INFER_SHAPE}' to determine feature shape...")
    
    try:
        df = pd.read_csv(DATA_FILE_TO_INFER_SHAPE)
    except FileNotFoundError:
        print(f"Error: File not found: '{DATA_FILE_TO_INFER_SHAPE}'")
        print("Please make sure 'train.csv' is in the directory.")
        sys.exit(1)

    # Separate features (X) and target (y)
    try:
        y = df['Reached.on.Time_Y.N']
        X = df.drop('Reached.on.Time_Y.N', axis=1)
    except KeyError:
        print("Error: The CSV file is missing the required columns.")
        sys.exit(1)

    # Get and apply the preprocessor
    preprocessor = get_preprocessing_pipeline()
    
    print("Fitting preprocessing pipeline...")
    X_processed = preprocessor.fit_transform(X)
    num_features = X_processed.shape[1]
    print(f"Data processed. Number of features after encoding: {num_features}")

    model = LogisticRegression()

    # Find one sample of each class (0 and 1) to fit on.
    try:
        idx_0 = y[y == 0].index[0]
        idx_1 = y[y == 1].index[0]
        
        X_init = X_processed[[idx_0, idx_1], :]
        y_init = y.loc[[idx_0, idx_1]]

        print(f"Fitting on two samples (indices {idx_0}, {idx_1}) to initialize...")
        model.fit(X_init, y_init)

    except IndexError:
        print(f"Error: The '{DATA_FILE_TO_INFER_SHAPE}' file does not contain samples of both classes (0 and 1).")
        sys.exit(1)
    except Exception as e:
        print(f"Error during model initialization: {e}")
        sys.exit(1)

    # Get the (empty/random) parameters
    initial_params = [model.coef_, model.intercept_]
    print(f"  - Coef shape: {initial_params[0].shape}")
    print(f"  - Intercept shape: {initial_params[1].shape}")

    # Save these initial parameters
    params_dict_to_save = {f"layer_{i}": param for i, param in enumerate(initial_params)}
    np.savez(MODEL_FILE, **params_dict_to_save)

    print(f"Successfully saved initial model parameters to '{MODEL_FILE}'.")
    print("--- [Server] Initializer complete ---")

# === END OF LOGIC FROM init_model.py ===


# === START OF LOGIC FROM server.py ===

# --- Helper function (No change) ---
def calculate_parameter_change(params1: List[np.ndarray], params2: List[np.ndarray]) -> float:
    """Calculates the L1 norm of the difference between two parameter lists."""
    if params1 is None:
        return 0.0
    total_change = 0.0
    for layer_old, layer_new in zip(params1, params2):
        total_change += np.sum(np.abs(layer_old - layer_new))
    return total_change

# --- 1. Define our Custom Strategy (No change) ---
class CustomFedAvg(fl.server.strategy.FedAvg):

    def __init__(self, initial_parameters: Optional[Parameters], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_parameters = initial_parameters
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

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            current_global_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            print("  Calculating change from previous global model...")

            change = calculate_parameter_change(
                self.previous_global_parameters,
                current_global_ndarrays
            )
            
            if server_round == 1 and self.previous_global_parameters is None:
                print("  (Round 1: This is the first model, no change to report)")
            else:
                print(f"  Total Parameter Change (L1 Norm): {change:.4f}")

            self.previous_global_parameters = current_global_ndarrays

        print("[Server (aggregate_fit)] Aggregation complete.")
        return aggregated_parameters, aggregated_metrics

# --- 2. Helper function to load the model (Simplified) ---
def load_saved_model(model_file: str) -> Optional[Parameters]:
    """Loads the .npz file."""
    # This function is now called *after* we guarantee the file exists.
    print(f"[Server] Loading model parameters from '{model_file}'...")
    try:
        loaded_npz = np.load(model_file)
        param_list = [loaded_npz[key] for key in sorted(loaded_npz.files)]
        print(f"[Server] Loaded {len(param_list)} parameter arrays.")
        return ndarrays_to_parameters(param_list)
    except Exception as e:
        print(f"[Server] CRITICAL Error loading model file: {e}. Exiting.")
        sys.exit(1)

# --- 3. The Main Server Loop (Updated) ---
def main_server_loop():
    print("--- Starting PERMANENT Federated Learning Server ---")
    
    # --- NEW: Check and create model file ---
    if not os.path.exists(MODEL_FILE):
        create_initial_model_file()
    # ----------------------------------------

    # Load the model from disk
    latest_model_parameters = load_saved_model(MODEL_FILE)
    if latest_model_parameters is None:
         # This should not happen now, but good to double-check
        print("Error: Model file is still missing. Exiting.")
        sys.exit(1)

    try:
        while True:
            # 1. Create the strategy *inside* the loop
            strategy = CustomFedAvg(
                initial_parameters=latest_model_parameters,
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_available_clients=2,
                min_fit_clients=2,
                min_evaluate_clients=2,
            )

            print("\n[Server] Waiting for a batch of 2 clients to connect...")

            # 2. Start the server for ONE round
            fl.server.start_server(
                server_address="0.0.0.0:8080",
                config=fl.server.ServerConfig(num_rounds=1),
                strategy=strategy,
            )

            # 3. After the round, save the new model
            if strategy.previous_global_parameters is not None:
                print(f"[Server] Saving new aggregated model to '{MODEL_FILE}'...")
                latest_model_parameters = ndarrays_to_parameters(strategy.previous_global_parameters)
                params_dict_to_save = {f"layer_{i}": param for i, param in enumerate(strategy.previous_global_parameters)}
                np.savez(MODEL_FILE, **params_dict_to_save)
                print("[Server] Model saved successfully.")
            else:
                print("[Server] No model updates to save this round.")

    except KeyboardInterrupt:
        print("\n[Server] Shutting down (Ctrl+C pressed)... Server is now offline.")

# === END OF LOGIC FROM server.py ===

if __name__ == "__main__":
    main_server_loop()


