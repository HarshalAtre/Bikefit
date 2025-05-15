# Cycling Posture Classifier

This project provides a comprehensive pipeline for analyzing cycling posture data using multiple trained models. The system processes input data through 7 different models to identify various biomechanical issues in cycling posture.

## Features

- Processes cycling biomechanical data through 7 different models
- Handles multiple model formats (TensorFlow, scikit-learn, etc.)
- Generates a consolidated CSV with all predictions


## Models and Predictions

The pipeline evaluates 7 different aspects of cycling posture:

1. *Knee Flexion*: Identifies under-flexion, normal, or over-flexion
2. *Saddle Height*: Determines if the saddle is too low, normal, or too high
3. *Saddle Position*: Evaluates if the saddle is too forward, normal, or too back
4. *Reach Label*: Assesses if the rider has underreach, normal reach, or overreach
5. *Hip Rocking*: Determines if hip movement is stable or unstable
6. *Hip Asymmetry*: Identifies right-leaning, normal, or left-leaning hip position
7. *Spine Asymmetry*: Evaluates right-leaning, normal, or left-leaning spine position

## Installation

bash
# Clone the repository
git clone https://github.com/HarshalAtre/bikefit.git
cd bikefit

# Install dependencies
pip install -r requirements.txt

## Input Data Format

The input CSV should contain the following columns:

-min_knee_angle,
-max_knee_angle,
-avg_elbow_angle,
-avg_neck_angle,
-avg_back_angle,
-reach,
-horizontal_distance,
-mean_pelvis_angle,
-pelvis_angle_deviation,
-mean_spine_angle,
-rider_arm_length,
-rider_torso_length,
-rider_inseam_length

## Output labels
The output csv should contain all these 7 labels:

-knee_flexion,
-saddle_height,
-saddle_position,
-reach_label,
-hip_rocking,
-hip_asymmetry,
-spine_asymmetry


## Model Formats

The pipeline supports multiple model formats:
- TensorFlow/Keras models (.h5)
- Scikit-learn models saved with joblib (.joblib)
- Pickle-serialized models (.pkl)

## Requirements
-absl-py==2.2.2
-astunparse==1.6.3
-blinker==1.9.0
-certifi==2025.4.26
-charset-normalizer==3.4.2
-click==8.2.0
-colorama==0.4.6
-Flask==3.1.0
-flask-cors==5.0.1
-flatbuffers==25.2.10
-gast==0.6.0
-google-pasta==0.2.0
-grpcio==1.71.0
-gunicorn==23.0.0
-h5py==3.13.0
-idna==3.10
-itsdangerous==2.2.0
-Jinja2==3.1.6
-joblib==1.5.0
-keras==3.9.2
-libclang==18.1.1
-Markdown==3.8
-markdown-it-py==3.0.0
-MarkupSafe==3.0.2
-mdurl==0.1.2
-ml_dtypes==0.5.1
-namex==0.0.9
-numpy==2.1.3
-opencv-contrib-python==4.11.0.86
-opt_einsum==3.4.0
-optree==0.15.0
-packaging==25.0
-pandas==2.2.3
-protobuf==5.29.4
-Pygments==2.19.1
-python-dateutil==2.9.0.post0
-pytz==2025.2
-requests==2.32.3
-rich==14.0.0
-scikit-learn==1.6.1
-scipy==1.15.3
-setuptools==80.4.0
-six==1.17.0
-tensorboard==2.19.0
-tensorboard-data-server==0.7.2
-tensorflow==2.19.0
-termcolor==3.1.0
-threadpoolctl==3.6.0
-typing_extensions==4.13.2
-tzdata==2025.2
-urllib3==2.4.0
-Werkzeug==3.1.3
-wheel==0.45.1
-wrapt==1.17.2

## Files
CYCLOPEDIA:
Contains scripts and data for Android Frontend Integration. For more details refer to the Readme in the CYCLOPEDIA Directory
Takes 3 inputs, 8 photos and 2 videos from the user, and sends it to the backend using FlaskAPI endpoint.

Execution Directory:
    App.py: This script runs on server startup and keeps listening to port 7860, and when the backend receives a request from the client to process incoming data, it calls the following scripts.
    side_final_v3.py: This script processes the incoming calibration images, and side.mp4 to produce the Aruco calibration matrix and side_features.csv
    back_final.py: This script is called by side_final_v3.py, this processes the rear.mp4 and produces the back_features.csv
    merge.py: This script combines the side_features.csv and back_features.csv with user_input.csv to create the 13_input_features.csv which acts as the final input source for the classifiers.
    rf_to_svm_models_zipped: Contains all working models for 7 dysfunctions in .pkl alongwith their best parameters, confusion matrices, encoder pickel files and relevant svm features after rf classification.
    feature_analysis: Contains feature importance information for each label in .png and .csv formats
    new_models: contains pre-trained DNN models in .h5 and .pkl formats
    classification_master_v2.py: the main pipeline which calls each model and create decision for feature selection and making predictions for each models and storing the merged final predictions in a single csv.
    13_samples_input.csv: csv to be taken as initial input for prediction pipeline.
    7_labels_outputs.csv: csv generated containing prediction of labels for each model.

ML_PROJECT_FINAL.ipynb: The required jupyter notebook for training and validation of all models showcasing their results and necessary output graphs.



