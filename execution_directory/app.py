from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import subprocess
from pathlib import Path
import sys
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity,decode_token
from pymongo import MongoClient
from datetime import datetime,timezone
from datetime import timedelta

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

BASE_DIR = Path(__file__).resolve().parent
CALIB_IMG_DIR = BASE_DIR / "calibration_images"

os.makedirs(CALIB_IMG_DIR, exist_ok=True)

bcrypt = Bcrypt(app)
app.config['JWT_SECRET_KEY'] = 'your_secret_key'  
jwt = JWTManager(app)
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=36)  # Set the token expiration time to 1 hour
client = MongoClient("mongodb+srv://Harshal:3YJfLzhmv9RBiF6j@cluster0.50encmi.mongodb.net/")
db = client["cycling_app"]
users = db["users"]
history = db["history"]

# TESTING
@app.route('/ping', methods=['GET', 'HEAD'])
def ping():
    return jsonify({'status': 'ok', 'message': 'Server is running'}), 200

#AUTH
# Register
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    if users.find_one({"email": data['email']}):
        return jsonify({"msg": "User already exists"}), 409

    hashed_pw = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    users.insert_one({
        "name": data['name'],
        "email": data['email'],
        "password": hashed_pw,
        "created_at": datetime.utcnow()
    })
    token = create_access_token(identity=data['email'])
    return jsonify({"token": token,"msg": "Registered successfully"}), 201

# Login
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user = users.find_one({"email": data['email']})
    if not user or not bcrypt.check_password_hash(user['password'], data['password']):
        return jsonify({"msg": "Invalid credentials"}), 401

    token = create_access_token(identity=user['email'])
    return jsonify({"token": token, "name": user['name']}), 200

# Save history (must be logged in)
@app.route('/api/history', methods=['POST'])

@jwt_required()

def save_history():
    print("got")
    user_email = get_jwt_identity()
    data = request.json
    print(user_email)
    print(data)
    history.insert_one({
        "user_email": user_email,
    
        "results": data["results"],
        "timestamp": datetime.utcnow()
    })
    return jsonify({"msg": "History saved"}), 201

# Fetch history
@app.route('/api/history', methods=['GET'])
@jwt_required()
def get_history():
    user_email = get_jwt_identity()
    result = list(history.find({"user_email": user_email}, {"_id": 0}))
    return jsonify(result), 200

#CHECK 
@app.route('/check', methods=['POST'])
def check_token():
    token = request.json.get('token', None)
    if not token:
        return jsonify({'valid': False, 'error': 'Token not provided'}), 400

    try:
        decoded = decode_token(token)
        exp_timestamp = decoded.get("exp", None)

        if not exp_timestamp:
            return jsonify({'valid': False, 'error': 'No expiration in token'}), 400

        now = datetime.now(timezone.utc).timestamp()

        if exp_timestamp > now:
            return jsonify({'valid': True}), 200
        else:
            return jsonify({'valid': False}), 200

    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)}), 400


#PREDICTION
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


