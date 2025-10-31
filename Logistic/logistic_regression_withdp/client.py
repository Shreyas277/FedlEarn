import argparse
import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import sys
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Define Data Preprocessing (Must match init_model.py) ---

# These lists must be manually kept in sync with the dataset
NUMERIC_FEATURES = ['Cost_of_the_Product', 'Customer_care_calls', 'Weight_in_gms']

ORDINAL_FEATURES = ['Product_importance']
ORDINAL_CATEGORIES = [['low', 'medium', 'high']]

NOMINAL_FEATURES = ['Warehouse_block', 'Mode_of_Shipment']
NOMINAL_CATEGORIES = [
    ['A', 'B', 'C', 'D', 'F'],
    ['Ship', 'Flight', 'Road']
]

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

# --- 2. Define the Data Loader (No change) ---
def load_data(csv_file_path: str):
    """
    Loads the specified CSV file and preprocesses it.
    """
    print(f"\n[Client] Loading private local data from '{csv_file_path}'...")

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: '{csv_file_path}' not found.")
        return None, None, None, None

    if df.empty:
        print("Error: No data found in CSV.")
        return None, None, None, None

    # Separate features (X) and target (y)
    try:
        y = df['Reached.on.Time_Y.N']
        X = df.drop('Reached.on.Time_Y.N', axis=1)
    except KeyError:
        print("Error: The CSV file is missing required columns.")
        return None, None, None, None
        
    # Create an 80/20 train/test split for this client's local data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Get the preprocessing pipeline
    pipeline = get_preprocessing_pipeline()

    # Fit the pipeline on the *training* data and transform both
    print("[Client] Fitting and transforming data with preprocessing pipeline...")
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    print(f"[Client] Data loaded: {len(y_train)} train samples, {len(y_test)} test samples.")
    return X_train_processed, y_train.values, X_test_processed, y_test.values

# --- 3. Define the Flower Client ---
class ShippingClient(fl.client.NumPyClient):
    
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        """Get model parameters as a list of NumPy arrays."""
        # For LogisticRegression, parameters are coef_ and intercept_
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        """Set model parameters from the central server."""
        # parameters is a list: [coef_, intercept_]
        if len(parameters) == 2:
            self.model.coef_ = parameters[0]
            self.model.intercept_ = parameters[1]
            
            # We must also set the 'classes_' attribute,
            # otherwise the model is considered "unfitted"
            self.model.classes_ = np.array([0, 1])
        else:
            print(f"Error: Received {len(parameters)} parameter arrays, expected 2.")

    def fit(self, parameters, config):
        """
        - Receives parameters from the server.
        - Clips and adds noise to the local update (DP-FedAvg).
        - Returns the new noisy local parameters to the server.
        """
        print(f"  ...Local training (fit) with DP...")

        # --- DP Configuration ---
        C = 1.0       # Clipping norm (max L2 norm of the update)
        sigma = 0.5   # Noise multiplier
        # ------------------------

        # 1. Get old parameters and set the model
        old_params = parameters
        self.set_parameters(old_params)

        # 2. Train the model locally
        self.model.fit(self.X_train, self.y_train)

        # 3. Get new parameters
        new_params = self.get_parameters(config={})

        # 4. Calculate the update delta (new - old)
        delta = [new - old for new, old in zip(new_params, old_params)]

        # 5. Clip the delta (L2 norm)
        # 5a. Calculate the total L2 norm of the delta
        delta_flat = np.concatenate([arr.flatten() for arr in delta])
        delta_norm = np.linalg.norm(delta_flat)

        # 5b. Calculate clipping factor
        clip_factor = max(1.0, delta_norm / C)

        # 5c. Clip the delta
        clipped_delta = [arr / clip_factor for arr in delta]
        
        print(f"  ...Delta norm: {delta_norm:.4f}, Clip factor: {clip_factor:.4f}")

        # 6. Add Gaussian noise
        # The noise standard deviation is (sigma * C)
        std_dev = sigma * C
        
        noisy_clipped_delta = []
        for arr in clipped_delta:
            noise = np.random.normal(0, std_dev, arr.shape)
            noisy_clipped_delta.append(arr + noise)

        # 7. Reconstruct the new parameters to send to the server
        # (old_parameters + noisy_clipped_delta)
        final_params = [old + noisy_delta for old, noisy_delta in zip(old_params, noisy_clipped_delta)]

        print(f"  ...Local training complete. Returning noisy parameters.")
        
        # Return the *full parameters* (not the delta), num_samples, and no metrics
        return final_params, len(self.X_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test set."""
        print(f"  ...Local evaluation...")
        self.set_parameters(parameters)
        
        # Predict probabilities to calculate log_loss
        try:
            y_proba = self.model.predict_proba(self.X_test)
            loss = log_loss(self.y_test, y_proba)
        except ValueError as e:
            print(f"  Error during log_loss calculation: {e}")
            loss = 0.0
        
        # Predict classes to calculate accuracy
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"  ...Evaluation complete: Accuracy={accuracy:.4f}, LogLoss={loss:.4f}")

        # Flower expects: loss, num_examples, metrics_dict
        return float(loss), len(self.X_test), {"accuracy": float(accuracy)}

# --- 4. Main function to start the client (No change) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client for E-commerce Shipping")
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to the client's local CSV data file (e.g., 'client1.csv').",
    )
    args = parser.parse_args()

    # Load model and data for the specified client
    model = LogisticRegression(
        warm_start=True,
        max_iter=5, # Number of local training iterations
        solver='lbfgs' 
    )
    
    # Load this client's data
    X_train, y_train, X_test, y_test = load_data(csv_file_path=args.csv_file)

    if X_train is None:
        print("Could not load data. Exiting.")
        sys.exit(1)

    # Start the client
    print(f"[Client {args.csv_file}] Connecting to server at 127.0.0.1:8080...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=ShippingClient(model, X_train, y_train, X_test, y_test),
    )

    # This line executes *after* the client disconnects.
    print(f"\n[Client {args.csv_file}] Thank you for your contribution! Connection closed.")
