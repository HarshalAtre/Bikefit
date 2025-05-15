from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import subprocess
from pathlib import Path
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

BASE_DIR = Path(__file__).resolve().parent
CALIB_IMG_DIR = BASE_DIR / "calibration_images"

os.makedirs(CALIB_IMG_DIR, exist_ok=True)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'message': 'Server is running'}), 200

@app.route('/upload', methods=['POST'])
def upload_and_process():
    try:
        # 1. Get form inputs
        rider_arm_length = float(request.form['armLength'])
        rider_torso_length = float(request.form['height'])
        rider_inseam_length = float(request.form['inseamLength'])

        # 2. Save input CSV
        pd.DataFrame([{
            "rider_arm_length": rider_arm_length,
            "rider_torso_length": rider_torso_length,
            "rider_inseam_length": rider_inseam_length
        }]).to_csv(BASE_DIR / "user_input.csv", index=False)

        # 3. Save videos
        rear_video = request.files.get('rearVideo')
        side_video = request.files.get('sideVideo')
        if rear_video:
            rear_video.save(BASE_DIR / "back.mp4")
        if side_video:
            side_video.save(BASE_DIR / "side.mp4")

        # 4. Save photos
        photos = request.files.getlist('photos')
        for photo in photos:
            photo.save(CALIB_IMG_DIR / photo.filename)

        # 5. Run pipeline
        subprocess.run(["python", "side_final_v3.py"], check=True, cwd=BASE_DIR)
        subprocess.run(["python", "merge.py"], check=True, cwd=BASE_DIR)
        subprocess.run([sys.executable, "classification_master_v2.py"], check=True)

        # 6. Return JSON output
        final_csv = pd.read_csv(BASE_DIR / "7_labels_output.csv")
        result = final_csv.iloc[0, 1:].to_dict()
        print(jsonify(result))
        key_map = {
                "model1_prediction": "Knee Flexion",
                "model2_prediction": "Saddle Height",
                "model3_prediction": "Saddle Position",
                "model4_prediction": "Reach",
                "model5_prediction": "Hip Rocking",
                "model6_prediction": "Hip Asymmetry",
                "model7_prediction": "Spine Asymmetry"
            }


        renamed = {key_map.get(k, k): v for k, v in result.items()}
        return jsonify(renamed)


    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 7860))  # Hugging Face expects port 7860
    app.run(host='0.0.0.0', port=port)


