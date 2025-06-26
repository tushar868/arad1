from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)
model = YOLO("runs/detect/train/weights/best.pt")

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json.get('image')
    img_data = base64.b64decode(data.split(',')[1])
    np_img = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(frame)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1})

    return jsonify(detections)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
