import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys

# --- Scikit-learn Imports for Preprocessing ---
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Configuration (Must be consistent with client.py and server.py) ---
TRAIN_FILE = 'combined1.csv'  # Combined training data file
TEST_FILE = 'test1.csv'        # Final testing data file
TARGET_COLUMN = 'delivery_risk'
RANDOM_STATE = 42
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.01

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

# --- 2. Data Preprocessing Pipeline ---

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

# --- 3. Main Centralized Training Function ---

def main_centralized_training():
    print(f"--- Starting Centralized Benchmark Training ---")
    
    # 3.1 Load Data
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError as e:
        print(f"[ERROR] Data file not found: {e}. Ensure '{TRAIN_FILE}' and '{TEST_FILE}' exist.")
        sys.exit(1)

    print(f"Training on '{TRAIN_FILE}' ({len(df_train)} samples).")
    print(f"Testing on '{TEST_FILE}' ({len(df_test)} samples).")

    # Separate features and target
    X_train_data = df_train[NUMERIC_FEATURES + NOMINAL_FEATURES]
    y_train_data = df_train[TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)
    
    X_test_data = df_test[NUMERIC_FEATURES + NOMINAL_FEATURES]
    y_test_data = df_test[TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)


    # 4. Preprocessing: Fit on Training Data, Transform Both
    preprocessor = get_preprocessing_pipeline()
    
    # FIT preprocessor ONLY on the combined training data
    X_train_processed = preprocessor.fit_transform(X_train_data).astype(np.float32)
    # TRANSFORM test data using the stats learned from the training data (DO NOT FIT)
    X_test_processed = preprocessor.transform(X_test_data).astype(np.float32)
    
    input_size = X_train_processed.shape[1]
    print(f"Input Feature Count: {input_size}")

    # Convert to PyTorch Tensors and DataLoaders
    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_data, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_data, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 5. Model Setup
    model = SimpleRegressionANN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    print("\nStarting ANN Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f'Epoch {epoch}/{EPOCHS}, Training Loss (MSE): {avg_loss:.4f}')

    # 7. Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        final_loss = criterion(y_pred_tensor, y_test_tensor)
        
    y_test_np = y_test_tensor.numpy()
    y_pred_np = y_pred_tensor.numpy()
    
    test_mse = mean_squared_error(y_test_np, y_pred_np)

    print("\n--- Centralized Benchmark Results ---")
    print(f"Test Set Loss (PyTorch MSE): {final_loss.item():.4f}")
    print(f"Test Set Mean Squared Error (MSE): {test_mse:.4f}")
    print("-------------------------------------")

if __name__ == "__main__":
    main_centralized_training()
