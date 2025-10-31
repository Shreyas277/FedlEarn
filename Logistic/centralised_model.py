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
# We use the full 'train.csv' for training
TRAIN_DATA_FILE = "train.csv"
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

# --- 2. Main Training & Evaluation Function ---
def main():
    print("--- Centralized Model Training & Evaluation Script ---")

    # 1. Load the training dataset
    print(f"Loading training data from '{TRAIN_DATA_FILE}'...")
    try:
        train_df = pd.read_csv(TRAIN_DATA_FILE)
        y_train = train_df['Reached.on.Time_Y.N']
        X_train_raw = train_df.drop('Reached.on.Time_Y.N', axis=1)
    except FileNotFoundError:
        print(f"Error: Training data file '{TRAIN_DATA_FILE}' not found.")
        sys.exit(1)
    except KeyError:
        print("Error: Training CSV is missing required columns.")
        sys.exit(1)

    # 2. Load the test dataset
    print(f"Loading test data from '{TEST_DATA_FILE}'...")
    try:
        test_df = pd.read_csv(TEST_DATA_FILE)
        y_test = test_df['Reached.on.Time_Y.N']
        X_test_raw = test_df.drop('Reached.on.Time_Y.N', axis=1)
    except FileNotFoundError:
        print(f"Error: Test data file '{TEST_DATA_FILE}' not found.")
        sys.exit(1)
    except KeyError:
        print("Error: Test CSV is missing required columns.")
        sys.exit(1)

    # 3. Preprocess the data
    print("Preprocessing data...")
    preprocessor = get_preprocessing_pipeline()
    
    # Fit the preprocessor on the TRAINING data
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    
    # ONLY transform the TEST data (to prevent data leakage)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    print(f"Data preprocessed. Number of features: {X_train_processed.shape[1]}")

    # 4. Initialize and Train the model
    print("Training centralized model on 'train.csv'...")
    model = LogisticRegression()
    model.fit(X_train_processed, y_train)
    print("Model training complete.")

    # 5. Run predictions on the test set
    print("Evaluating model on 'test.csv'...")
    y_pred = model.predict(X_test_processed)

    # 6. Calculate and display results
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Late (0)', 'On Time (1)'])
    matrix = confusion_matrix(y_test, y_pred)

    print("\n--- Centralized Model Evaluation Results ---")
    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
    
    print("Classification Report:")
    print(report)
    
    print("Confusion Matrix:")
    print(matrix)
    print("  (Rows: Actual, Cols: Predicted)")

if __name__ == "__main__":
    main()

