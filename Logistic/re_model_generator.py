import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
MODEL_FILE = "final_model.npz"
TEST_DATA_FILE = "test.csv"
# --- End Configuration ---


# --- 1. Define Data Preprocessing (Must be IDENTICAL to client.py) ---

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
    Hardcoded categories are ESSENTIAL to ensure the test data
    is transformed into the same shape as the training data.
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

# --- 2. Main Evaluation Function ---
def main():
    print("--- Federated Model Evaluation Script ---")

    # 1. Load the trained model parameters
    print(f"Loading trained parameters from '{MODEL_FILE}'...")
    try:
        loaded_params = np.load(MODEL_FILE)
        # Parameters are saved as 'layer_0' (coef_) and 'layer_1' (intercept_)
        model_coefs = loaded_params['layer_0']
        model_intercept = loaded_params['layer_1']
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_FILE}' not found.")
        print("Please run the federated training first to create this file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model file: {e}")
        sys.exit(1)

    # 2. Load the test dataset
    print(f"Loading test data from '{TEST_DATA_FILE}'...")
    try:
        df = pd.read_csv(TEST_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Test data file '{TEST_DATA_FILE}' not found.")
        sys.exit(1)

    # 3. Separate features (X) and target (y)
    try:
        y_test = df['Reached.on.Time_Y.N']
        X_test_raw = df.drop('Reached.on.Time_Y.N', axis=1)
    except KeyError:
        print("Error: The CSV file is missing the required columns.")
        sys.exit(1)

    # 4. Preprocess the test data
    print("Preprocessing test data...")
    preprocessor = get_preprocessing_pipeline()
    
    # We fit the preprocessor (e.g., the StandardScaler) and transform the data
    # This is safe *because* the categories for OHE and Ordinal are hardcoded
    try:
        X_test_processed = preprocessor.fit_transform(X_test_raw)
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        print("This might be due to a mismatch between the data and the hardcoded categories.")
        sys.exit(1)

    # 5. Initialize model and "hydrate" it with the loaded parameters
    print("Loading parameters into new model...")
    model = LogisticRegression()
    
    # Manually set the learned parameters
    model.coef_ = model_coefs
    model.intercept_ = model_intercept
    
    # CRITICAL: We must also tell the model what classes it's predicting
    model.classes_ = np.array([0, 1])
    
    # --- FIX ---
    # Removed the line that caused the error, as it's not present
    # in older scikit-learn versions.
    # print(f"Model loaded. Features: {model.n_features_in_}")
    print("Model loaded successfully.")
    # --- END OF FIX ---

    # 6. Run predictions on the test set
    print("Running predictions on test data...")
    y_pred = model.predict(X_test_processed)

    # 7. Calculate and display results
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Late (0)', 'On Time (1)'])
    matrix = confusion_matrix(y_test, y_pred)

    print("\n--- Final Model Evaluation Results ---")
    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
    
    print("Classification Report:")
    print(report)
    
    print("Confusion Matrix:")
    print(matrix)
    print("  (Rows: Actual, Cols: Predicted)")

if __name__ == "__main__":
    main()


