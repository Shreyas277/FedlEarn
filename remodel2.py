import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# --- Configuration (Must match training setup) ---
MODEL_FILE = 'model_upto2.npz'
TRAINING_STATS_FILE = 'global.csv' # Used to fit the preprocessor
TEST_FILE = 'test1.csv'             # Used for final validation
EXPORT_FILE = 'delivery_risk_model.pkl'

# Define the features and categories exactly as used in client.py/server.py
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
    ['Advance shipping', 'Late delivery', 'Shipping on time'], 
    ['Consumer', 'Corporate', 'Home Office'],
    ['Apparel', 'Fan Shop', 'Footwear', 'Golf'],
    ['CLOSED', 'COMPLETE', 'ON_HOLD', 'PENDING_PAYMENT', 'PROCESSING'],
    ['First Class', 'Second Class', 'Standard Class']
]
TARGET_COLUMN = 'delivery_risk' 

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

# --- 2. Main Export and Evaluation Function ---

def main_export_and_evaluate():
    print(f"--- Starting Model Export and Final Evaluation ---")
    
    # 2.1 Sanity Check for Input Files
    if not os.path.exists(TRAINING_STATS_FILE):
        print(f"[FATAL ERROR] Training stats file '{TRAINING_STATS_FILE}' not found. Cannot fit preprocessor.")
        sys.exit(1)
    if not os.path.exists(MODEL_FILE):
        print(f"[FATAL ERROR] Model weights file '{MODEL_FILE}' not found.")
        sys.exit(1)
    if not os.path.exists(TEST_FILE):
        print(f"[FATAL ERROR] Test data file '{TEST_FILE}' not found. Cannot evaluate.")
        sys.exit(1)

    # 2.2 Fit Preprocessor (Crucial Step: Use training data stats)
    df_full = pd.read_csv(TRAINING_STATS_FILE)
    X_full = df_full[NUMERIC_FEATURES + NOMINAL_FEATURES]
    
    preprocessor = get_preprocessing_pipeline()
    X_dummy = preprocessor.fit_transform(X_full)
    input_size = X_dummy.shape[1]
    
    print(f"[INFO] Preprocessor fitted using '{TRAINING_STATS_FILE}'. Input size is {input_size}.")

    # 2.3 Reconstruct Model and Load Weights
    model = SimpleRegressionANN(input_size)
    loaded_params = np.load(MODEL_FILE, allow_pickle=True)
    state_dict = model.state_dict()
    
    saved_arrays = [loaded_params[key] for key in sorted(loaded_params.files, key=lambda x: int(x.split('_')[-1]))]
    
    new_state_dict = {
        k: torch.tensor(v, dtype=torch.float32) 
        for k, v in zip(state_dict.keys(), saved_arrays)
    }
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    print("[INFO] PyTorch model reconstructed and weights loaded.")

    # 2.4 Save the complete deployment package (.pkl file)
    deployment_package = {
        'model_state_dict': model.state_dict(),
        'preprocessor': preprocessor,
        'input_size': input_size
    }

    with open(EXPORT_FILE, 'wb') as f:
        pickle.dump(deployment_package, f)
        
    print(f"\n[SUCCESS] Deployment package saved to '{EXPORT_FILE}'.")
    
    # 2.5 Load and Preprocess Test Data
    df_test = pd.read_csv(TEST_FILE)
    X_test_data = df_test[NUMERIC_FEATURES + NOMINAL_FEATURES]
    y_test_data = df_test[TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)

    # Transform test data using the preprocessor fitted on global.csv
    X_test_transformed = preprocessor.transform(X_test_data).astype(np.float32)
    
    X_test_tensor = torch.tensor(X_test_transformed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_data, dtype=torch.float32)

    # 2.6 Evaluation
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        loss = criterion(y_pred, y_test_tensor)
        
    y_test_np = y_test_tensor.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    test_mse = mean_squared_error(y_test_np, y_pred_np)

    print("\n--- Final Model Evaluation on test1.csv ---")
    print(f"Total Test Examples: {len(X_test_tensor)}")
    print(f"Test Set Loss (PyTorch MSE): {loss.item():.4f}")
    print(f"Test Set Mean Squared Error (MSE): {test_mse:.4f}")
    print("---------------------------------------------")


if __name__ == "__main__":
    main_export_and_evaluate()
