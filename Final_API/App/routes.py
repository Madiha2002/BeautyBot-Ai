from flask import Blueprint, render_template, request, jsonify
from .sockets import socketio
from .utils import *
import base64

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_message(data):

    try:
        # Decode the incoming Base64-encoded image bytes
        raw_bytes = base64.b64decode(data["body"])
        newData = data["features"]
        # print(newData)
        # Process the image bytes
        
        makeup_features  = [key for key, value in data["features"].items() if value != [0, 0, 0]]
        processed_image = apply_makeup_on_bytes(
            raw_bytes, 
            data["features"],
            is_stream=False, 
            features=makeup_features, 
            show_landmarks=True
        )
        
        # # Encode the processed image back to Base64 for transmission
        result = base64.b64encode(processed_image).decode('utf-8')
        
        # Emit the processed image back to the client
        socketio.emit("image", {"body": result})
        pass
    except Exception as e:
        print(f"Error processing image: {e}")




@main.route('/process-image', methods=['POST'])
def process_image():
    try:
        # Check if an image file is in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        # Retrieve the image file and additional features from the request
        image_file = request.files['image']
        features = request.json.get('features', {})

        # Read the image bytes
        image_bytes = image_file.read()

        # Extract features to apply makeup
        makeup_features = [key for key, value in features.items() if value != [0, 0, 0]]

        # Process the image
        processed_image = apply_makeup_on_bytes(
            image_bytes,
            features,
            is_stream=False,
            features=makeup_features,
            show_landmarks=True
        )

        # Encode the processed image to Base64 for the response
        processed_image_base64 = base64.b64encode(processed_image).decode('utf-8')

        return jsonify({"processed_image": processed_image_base64}), 200

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500



