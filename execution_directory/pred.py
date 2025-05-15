import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os

# Path to models and encoders
model_dir = "cyclist_dysfunction_models"

# Define target dysfunctions and class labels
target_variables = {
    'knee_flexion': ['normal', 'under-flexion', 'over-flexion'],
    'saddle_height': ['normal', 'too low', 'too high'],
    'saddle_position': ['normal', 'too forward', 'too back'],
    'reach_label': ['normal', 'overreach', 'underreach'],
    'hip_rocking': ['stable', 'unstable'],
    'hip_asymmetry': ['normal', 'left-lean', 'right-lean'],
    'spine_asymmetry': ['normal', 'left lean', 'right lean']
}

# Input file (1 row, 13 columns), Output file
input_csv_path = '13_samples_input.csv'
output_csv_path = 'predicted_single_cyclist.csv'

# Read single row of input features
input_data = pd.read_csv(input_csv_path)
feature_columns = input_data.columns.tolist()
input_array = input_data[feature_columns].values

# Store results
results = []

# Loop through each dysfunction prediction
for target in target_variables.keys():
    model_path = os.path.join(model_dir, f"{target}_model.h5")
    encoder_path = os.path.join(model_dir, f"{target}_encoder.pkl")

    # Load model and encoder
    model = tf.keras.models.load_model(model_path)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    # Predict
    if len(target_variables[target]) == 2:  # Binary
        prob = float(model.predict(input_array)[0][0])
        pred_class = 1 if prob > 0.5 else 0
        confidence = prob if pred_class == 1 else 1 - prob
    else:  # Multi-class
        probs = model.predict(input_array)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    label = encoder.inverse_transform([pred_class])[0]

    results.append({
        'dysfunction': target,
        'prediction': label,
        'confidence': round(confidence, 4)
    })

# Save prediction results to CSV
pred_df = pd.DataFrame(results)
pred_df.to_csv(output_csv_path, index=False)
print(f"Prediction saved to {output_csv_path}")
