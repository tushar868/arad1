from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load your trained YOLOv8 model (adjust the path as needed)
model = YOLO("runs/detect/train/weights/best.pt")

def readb64(base64_string):
    try:
        base64_data = base64_string.split(',')[1] if ',' in base64_string else base64_string
        decoded_data = base64.b64decode(base64_data)
        nparr = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("Error decoding base64:", str(e))
        return None

@app.route('/detect', methods=['POST'])
def detect_billboard():
    try:
        data = request.get_json()
        base64_img = data.get('image')

        if not base64_img:
            return jsonify({"error": "No image provided"}), 400

        image = readb64(base64_img)
        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400

        results = model(image)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return jsonify({"message": "No billboard found", "success": False}), 200

        # Optional: You can include bounding box coordinates in response if needed
        response_boxes = []
        for box in boxes.xyxy.tolist():
            response_boxes.append({
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3])
            })

        return jsonify({
            "message": "Billboard detected",
            "success": True,
            "boxes": response_boxes
        }), 200

    except Exception as e:
        print("Exception in /detect:", str(e))
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
