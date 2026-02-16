from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import base64
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from deepface import DeepFace
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
EMBEDDINGS_FILE = "face_embeddings.json"
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
THRESHOLD = 0.68  # VGG-Face threshold

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load face embeddings
face_embeddings_dict = {}

def load_embeddings():
    """Load face embeddings from JSON file"""
    global face_embeddings_dict
    try:
        if not os.path.exists(EMBEDDINGS_FILE):
            print(f"⚠ Warning: {EMBEDDINGS_FILE} not found. Creating empty embeddings file.")
            face_embeddings_dict = {}
            return False
            
        with open(EMBEDDINGS_FILE, 'r') as f:
            data = json.load(f)
            
        if not data:
            print(f"⚠ Warning: {EMBEDDINGS_FILE} is empty.")
            face_embeddings_dict = {}
            return False
            
        face_embeddings_dict = {}
        # Convert lists to numpy arrays
        for name, embedding_list in data.items():
            if isinstance(embedding_list, list) and len(embedding_list) > 0:
                face_embeddings_dict[name] = np.array(embedding_list)
            else:
                print(f"⚠ Warning: Invalid embedding for {name}")
                
        print(f"✓ Loaded {len(face_embeddings_dict)} face embeddings from {EMBEDDINGS_FILE}")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON in {EMBEDDINGS_FILE}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error loading embeddings: {e}")
        return False

# Load embeddings on startup
load_embeddings()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove header if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
            
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

def verify_face(image_path):
    """Verify face against known embeddings"""
    try:
        if not face_embeddings_dict:
            return {
                'success': False,
                'error': 'No face embeddings loaded',
                'message': 'Database is empty. Please contact administrator.'
            }
        
        # Check if image exists
        if not os.path.exists(image_path):
            return {
                'success': False,
                'error': 'Image file not found',
                'message': 'Could not process the image'
            }
        
        # Extract embeddings from the test image
        test_embeddings = DeepFace.represent(
            img_path=image_path,
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=True
        )
        
        if not test_embeddings:
            return {
                'success': False,
                'error': 'No face detected',
                'message': 'Please ensure your face is clearly visible'
            }
        
        # Use first detected face
        test_embedding = np.array(test_embeddings[0]['embedding'])
        
        # Find closest match
        closest_match = None
        min_distance = float('inf')
        
        for name, known_embedding in face_embeddings_dict.items():
            try:
                distance = cosine(known_embedding, test_embedding)
                if distance < min_distance:
                    min_distance = distance
                    closest_match = name
            except Exception as e:
                print(f"Error comparing with {name}: {e}")
                continue
        
        # Check if match is within threshold
        if closest_match and min_distance < THRESHOLD:
            confidence = round((1 - min_distance) * 100, 2)
            return {
                'success': True,
                'name': closest_match,
                'confidence': confidence,
                'distance': round(min_distance, 4),
                'message': f'Welcome, {closest_match}!'
            }
        elif closest_match:
            return {
                'success': False,
                'error': 'Face not recognized',
                'closest_match': closest_match,
                'distance': round(min_distance, 4),
                'threshold': THRESHOLD,
                'message': 'Face not recognized. Please try again or contact administrator.'
            }
        else:
            return {
                'success': False,
                'error': 'No matches found',
                'message': 'No matching face found in database'
            }
            
    except ValueError as e:
        print(f"Face detection error: {e}")
        return {
            'success': False,
            'error': 'No face detected',
            'message': 'Could not detect a face in the image. Please ensure your face is clearly visible.'
        }
    except Exception as e:
        print(f"Error during face verification: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'An error occurred during verification. Please try again.'
        }

@app.route('/')
def index():
    """Serve the main page (index.html)"""
    return render_template('index.html')

@app.route('/welcome')
def welcome():
    """Serve the welcome page (welcome.html)"""
    # Note: The welcome.html uses a JS parameter 'name' for personalization
    return render_template('welcome.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/api/verify', methods=['POST'])
def verify():
    """API endpoint for face verification"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided',
                'message': 'Please capture an image first'
            }), 400
        
        # Convert base64 to image
        image_base64 = data['image']
        img = base64_to_image(image_base64)
        
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image format',
                'message': 'Could not process the image. Please try again.'
            }), 400
        
        # Save temporary image
        temp_filename = f'temp_face_{os.getpid()}.jpg'
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        try:
            cv2.imwrite(temp_path, img)
            
            # Verify face
            result = verify_face(temp_path)
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_path}: {e}")
        
    except Exception as e:
        print(f"Error in verify endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Server error occurred. Please try again.'
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check server and embeddings status"""
    return jsonify({
        'status': 'online',
        'embeddings_loaded': len(face_embeddings_dict) > 0,
        'total_faces': len(face_embeddings_dict),
        'faces': list(face_embeddings_dict.keys()) if face_embeddings_dict else [],
        'threshold': THRESHOLD
    })

@app.route('/api/reload-embeddings', methods=['POST'])
def reload_embeddings():
    """Reload face embeddings from file"""
    success = load_embeddings()
    return jsonify({
        'success': success,
        'total_faces': len(face_embeddings_dict),
        'faces': list(face_embeddings_dict.keys()) if face_embeddings_dict else [],
        'message': 'Embeddings reloaded successfully' if success else 'Failed to reload embeddings'
    })

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Server error',
        'message': 'An internal server error occurred'
    }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Association Day 2025 Face Recognition Server")
    print("="*60)
    
    if face_embeddings_dict:
        print(f"\n✓ Server ready with {len(face_embeddings_dict)} registered faces:")
        for name in face_embeddings_dict.keys():
            print(f"  • {name}")
    else:
        print("\n⚠ Warning: No face embeddings loaded!")
        print("  Please ensure 'face_embeddings.json' exists in the project directory.")
        print("  The server will run but face verification will fail.")
    
    print(f"\n📊 Configuration:")
    print(f"  • Threshold: {THRESHOLD}")
    print(f"  • Model: VGG-Face")
    print(f"  • Detector: OpenCV")
    print(f"  • Upload folder: {UPLOAD_FOLDER}")
    
    print("\n🔗 Server running on: http://localhost:5000")
    print("   - Main page: http://localhost:5000/")
    print("   - API status: http://localhost:5000/api/status")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)