import requests

# Server endpoint
url = "http://192.168.29.130:5000/upload"

# Form data
data = {
    'height': '175',
    'armLength': '65',
    'inseamLength': '80'
}

# Files to upload
files = []

# Add 8 photos
for i in range(1, 9):
    file_path = f'calibration_images/{i}.jpg'
    files.append(('photos', (f'{i}.jpg', open(file_path, 'rb'), 'image/jpeg')))

# Add rear and side videos
files.append(('rearVideo', ('rear.mp4', open('back.mp4', 'rb'), 'video/mp4')))
files.append(('sideVideo', ('side.mp4', open('side.mp4', 'rb'), 'video/mp4')))

# Send POST request
response = requests.post(url, files=files, data=data)

# Output response
print("Status code:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Response Text:", response.text)
