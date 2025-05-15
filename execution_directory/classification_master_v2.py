import pandas as pd
import numpy as np
import os
from joblib import load
import pickle
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
# Configuration - hardcoded paths
INPUT_CSV_PATH = BASE_DIR / "13_samples_input.csv"
OUTPUT_CSV_PATH = BASE_DIR / "7_labels_output.csv"
# Model paths - hardcoded
# Note: These should now be paths to SVM or other scikit-learn models (.joblib or .pkl)
MODEL_PATHS = {
    "model1": BASE_DIR / "rf_to_svm_models_zipped" / "knee_flexion_svm_model.pkl",
    "model2": BASE_DIR / "rf_to_svm_models_zipped" / "saddle_height_svm_model.pkl",
    "model3": BASE_DIR / "rf_to_svm_models_zipped" / "saddle_position_svm_model.pkl",
    "model4": BASE_DIR / "rf_to_svm_models_zipped" / "reach_label_svm_model.pkl",
    "model5": BASE_DIR / "rf_to_svm_models_zipped" / "hip_rocking_svm_model.pkl",
    "model6": BASE_DIR / "rf_to_svm_models_zipped" / "hip_asymmetry_svm_model.pkl",
    "model7": BASE_DIR / "rf_to_svm_models_zipped" / "spine_asymmetry_svm_model.pkl"
}

# Feature configuration for each model (0-indexed)
MODEL_FEATURES = {
    "model1": [1, 0],
    "model2": [0, 1],
    "model3": [6, 12, 10, 11],
    "model4": [5, 2, 4, 11, 10, 12],
    "model5": [8],
    "model6": [3],
    "model7": [9]
}

# Class labels for each model (if needed for interpretation)
MODEL_CLASSES = {
    "model1": ["under-flexion", "normal", "over-flexion"],
    "model2": ["too low", "normal", "too high"],
    "model3": ["too forward", "normal", "too back"],
    "model4": ["underreach", "normal", "overreach"],
    "model5": ["unstable", "stable"],
    "model6": ["right-lean", "normal", "left-lean"],
    "model7": ["right-lean", "normal", "left-lean"]
}


def load_model(model_path):
    """
    Load a model from a given path, supporting scikit-learn model formats (.joblib, .pkl)
    """
    file_extension = os.path.splitext(model_path)[1].lower()

    try:
        if file_extension == '.joblib':
            return load(model_path)
        elif file_extension in ['.pkl', '.pickle']:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported model format: {file_extension}")
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return None


def get_model_prediction(model, features, model_name):
    """
    Get predictions from a scikit-learn model
    """
    try:
        # Reshape for scikit-learn models
        features_array = np.array([features])

        # Check if the model has predict_proba method (most classifiers do)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_array)
            class_idx = np.argmax(probs[0])
        else:
            # Regular prediction
            class_idx = model.predict(features_array)[0]

        # Map to class label if available
        if model_name in MODEL_CLASSES and class_idx < len(MODEL_CLASSES[model_name]):
            return MODEL_CLASSES[model_name][class_idx]
        else:
            return str(class_idx)

    except Exception as e:
        print(f"Error making prediction with {model_name}: {str(e)}")
        return "error"


def main():
    try:
        # Load input data
        print(f"Loading input data from {INPUT_CSV_PATH}")
        df = pd.read_csv(INPUT_CSV_PATH)

        # Check if we have all 13 features
        if df.shape[1] < 13:
            raise ValueError(f"Input CSV should have at least 13 features, but has {df.shape[1]}")

        # Load all models
        models = {}
        for model_name, model_path in MODEL_PATHS.items():
            print(f"Loading model: {model_name}")
            model = load_model(model_path)
            if model is not None:
                models[model_name] = model
            else:
                print(f"Failed to load {model_name}, skipping...")

        # Process each row and collect predictions
        results = []
        row_count = len(df)

        print(f"Processing {row_count} rows of data...")
        for i, row in df.iterrows():
            if i % 100 == 0:
                print(f"Processing row {i}/{row_count}")

            row_data = row.values
            row_result = {"row_id": i}

            # Apply each model with its specific features
            for model_name, model in models.items():
                if model_name in MODEL_FEATURES:
                        try:
                            selected_features = [row_data[idx] for idx in MODEL_FEATURES[model_name]]
                            print(f"{model_name} input features: {selected_features}")
                            prediction = get_model_prediction(model, selected_features, model_name)
                            print(f"{model_name} prediction: {prediction}")
                            row_result[f"{model_name}_prediction"] = prediction
                        except Exception as e:
                            print(f"Error processing model {model_name} on row {i}: {e}")

            results.append(row_result)

        # Create and save output DataFrame
        results_df = pd.DataFrame(results)
        print(f"Saving results to {OUTPUT_CSV_PATH}")
        results_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print("Processing complete!")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()