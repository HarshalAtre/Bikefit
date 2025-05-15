import pandas as pd
import numpy as np
import os
from tensorflow import keras
from joblib import load
import pickle

# Configuration - hardcoded paths
INPUT_CSV_PATH = "13_samples_input.csv"
OUTPUT_CSV_PATH = "predicted_single_cyclist.csv"

# Model paths - hardcoded
MODEL_PATHS = {
    "model1": "new_models/hip_asymmetry_model.h5",
    "model2": "new_models/hip_rocking_model.h5",
    "model3": "new_models/knee_flexion_model.h5",
    "model4": "new_models/reach_label_model.h5",
    "model5": "new_models/saddle_height_model.h5",
    "model6": "new_models/saddle_position_model.h5",
    "model7": "new_models/spine_asymmetry_model.h5",
}

# Feature configuration for each model (0-indexed)
MODEL_FEATURES = {
    "model1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "model2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "model3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "model4": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "model5": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "model6": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],         
    "model7": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
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
    Load a model from a given path, supporting various model formats (.h5, .joblib, .pkl)
    """
    file_extension = os.path.splitext(model_path)[1].lower()
    
    try:
        if file_extension == '.h5':
            return keras.models.load_model(model_path)
        elif file_extension == '.joblib':
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
    Get predictions from a model with appropriate handling for different model types
    """
    try:
        # Reshape features for keras models which expect batch dimension
        if isinstance(model, keras.Model):
            # Check if the model expects a specific input shape
            input_shape = model.input_shape
            
            # If input shape has more than 2 dimensions, reshape accordingly
            if len(input_shape) > 2:
                # For CNNs or models with specific input shapes
                sample_reshaped = np.reshape(features, input_shape[1:])
                features_array = np.array([sample_reshaped])
            else:
                # Standard reshape for regular models
                features_array = np.array([features])
            
            # Get raw predictions
            raw_preds = model.predict(features_array, verbose=0)
            
            # Handle different output formats
            if isinstance(raw_preds, list):
                preds = raw_preds[0]  # Some models return a list of arrays
            else:
                preds = raw_preds
            
            # Get class index for classification models
            if len(preds.shape) > 1 and preds.shape[1] > 1:
                # Multi-class classification
                class_idx = np.argmax(preds[0])
            else:
                # Binary classification with threshold 0.5
                class_idx = 1 if preds[0] > 0.5 else 0
                
            # Map to class label if available
            if model_name in MODEL_CLASSES and class_idx < len(MODEL_CLASSES[model_name]):
                return MODEL_CLASSES[model_name][class_idx]
            else:
                return str(class_idx)
            
        # Handle scikit-learn models (joblib or pickle loaded)
        else:
            # Reshape for scikit-learn models
            features_array = np.array([features])
            
            # Check if the model has predict_proba method
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
                    # Select only the features needed for this model
                    selected_features = [row_data[idx] for idx in MODEL_FEATURES[model_name]]
                    # Get prediction
                    prediction = get_model_prediction(model, selected_features, model_name)
                    row_result[f"{model_name}_prediction"] = prediction
            
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
