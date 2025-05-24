from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import io
from PIL import Image
import re
import onnxruntime as ort
import json
import logging
from datetime import datetime
import os
import gc
from collections import defaultdict

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Reduced logging to save memory
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Only console logging
)
logger = logging.getLogger(__name__)

# Memory-optimized configuration
CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.1,
    'DEBUG_MODE': False,  # Disabled debug mode
    'MAX_IMAGE_SIZE': 640,  # Limit image size
    'BATCH_SIZE': 1,
    'CACHE_ENABLED': False,  # Disabled caching to save memory
    'ADAPTIVE_MAPPING': True,
    'LEARNING_MODE': False  # Disabled learning mode
}

# Load ONNX model with memory optimization
try:
    logger.warning("Loading ONNX model...")
    model_path = os.path.join(BASE_DIR, "best.onnx")
    
    # Optimize ONNX runtime for memory
    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.enable_mem_pattern = False  # Reduce memory usage
    session_options.enable_cpu_mem_arena = False  # Disable memory arena
    session_options.arena_extend_strategy = 'kSameAsRequested'
    
    session = ort.InferenceSession(
        model_path, 
        sess_options=session_options,
        providers=providers
    )
    
    model_loaded = True
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    logger.warning("ONNX model loaded successfully!")
    
except Exception as e:
    logger.error(f"Error loading ONNX model: {e}")
    session = None
    model_loaded = False

# Simplified class mapping system
class SimpleClassMapper:
    def __init__(self):
        # Pre-defined mappings only - no learning to save memory
        self.mappings = {
            # Clean detections
            "8323": "clean", "8345": "clean",
            # Layer shift detections  
            "4313": "layer", "4473": "layer", "4713": "layer", "4553": "layer",
            "4233": "layer", "4153": "layer", "8322": "layer", "4633": "layer",
            "2713": "layer", "4393": "layer",
            # Spaghetti detections
            "8267": "spaghetti", "8266": "spaghetti", "8270": "spaghetti",
            "8344": "spaghetti", "8269": "spaghetti", "8347": "spaghetti",
            "8264": "spaghetti", "8263": "spaghetti", "6341": "spaghetti",
            "6378": "spaghetti", "6335": "spaghetti", "8101": "spaghetti",
            "6346": "spaghetti", "6332": "spaghetti", "6334": "spaghetti",
            "6377": "spaghetti", "6356": "spaghetti"
        }
        
    def get_mapping(self, class_id, confidence=None):
        class_id_str = str(class_id)
        
        if class_id_str in self.mappings:
            return self.mappings[class_id_str]
        
        # Simple fallback logic
        class_id_int = int(class_id)
        if 4000 <= class_id_int < 5000:
            return 'layer'
        elif 6000 <= class_id_int < 7000:
            return 'spaghetti'
        elif 8000 <= class_id_int < 8350:
            if 8260 <= class_id_int < 8300:
                return 'spaghetti'
            return 'clean'
        return 'clean'

# Initialize simplified mapper
mapper = SimpleClassMapper()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'version': '2.1.0'
    })

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        if not data or 'image' not in data:
            raise ValueError("No image data provided")
        
        image_b64 = data['image']
        
        # Remove data URL prefix if present
        if 'data:image' in image_b64:
            image_b64 = re.sub('^data:image/.+;base64,', '', image_b64)
        
        # Decode and preprocess image with memory optimization
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        
        # Limit image size to reduce memory usage
        max_size = CONFIG['MAX_IMAGE_SIZE']
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Process image
        img_array = preprocess_image_optimized(image)
        
        # Run inference
        outputs = session.run(output_names, {input_name: img_array})
        
        # Process detections
        detections = process_detections_optimized(outputs)
        
        # Clean up memory
        del img_array, outputs, image, image_data
        gc.collect()
        
        return jsonify(detections)
    
    except Exception as e:
        logger.error(f"Error in detection: {e}")
        # Clean up memory on error
        gc.collect()
        return jsonify({'error': str(e)}), 500

def preprocess_image_optimized(image, target_size=(640, 640)):
    """Memory-optimized image preprocessing"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Simple resize without padding to save memory
        resized = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(resized, dtype=np.float32)
        
        # Normalize
        img_array = img_array / 255.0
        
        # Transpose and add batch dimension
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

def process_detections_optimized(outputs):
    """Memory-optimized detection processing"""
    detections = []
    
    try:
        output = outputs[0]
        
        if len(output.shape) == 3:
            # Process only first N detections to save memory
            max_detections = min(100, len(output[0]))
            
            for i in range(max_detections):
                detection = output[0][i]
                
                if len(detection) < 5:
                    continue
                
                x, y, w, h = detection[0:4]
                confidence = detection[4]
                
                # Get class scores
                if len(detection) > 5:
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    class_score = class_scores[class_id]
                else:
                    class_id = 0
                    class_score = 1.0
                
                # Normalize confidence values
                if confidence > 1.0:
                    confidence = confidence / 100.0
                if class_score > 1.0:
                    class_score = class_score / 100.0
                
                combined_confidence = confidence * class_score
                
                if combined_confidence < CONFIG['CONFIDENCE_THRESHOLD']:
                    continue
                
                # Get defect type
                defect_type = mapper.get_mapping(class_id, combined_confidence)
                
                detections.append({
                    'type': defect_type,
                    'confidence': float(combined_confidence),
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'class_id': int(class_id)
                })
        
        # Keep only best detection per type
        type_groups = {}
        for det in detections:
            det_type = det['type']
            if det_type not in type_groups or det['confidence'] > type_groups[det_type]['confidence']:
                type_groups[det_type] = det
        
        detections = list(type_groups.values())
        
        if not detections:
            detections = [{
                'type': 'clean',
                'confidence': 0.95,
                'bbox': [0.5, 0.5, 0.1, 0.1],
                'class_id': 0
            }]
        
        return detections
    
    except Exception as e:
        logger.error(f"Error processing detections: {e}")
        return [{
            'type': 'clean',
            'confidence': 0.8,
            'bbox': [0.5, 0.5, 0.1, 0.1],
            'class_id': 0
        }]

# Remove memory-intensive endpoints to save memory
# Keeping only essential endpoints

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)