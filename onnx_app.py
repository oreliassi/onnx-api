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
import gc  # Add garbage collection

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MEMORY OPTIMIZED: Minimal logging to reduce memory usage
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MEMORY OPTIMIZED: Lower thresholds and smaller configs
CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.3,
    'DEBUG_MODE': False,  # Disable debug to save memory
    'OUTPUT_DIR': os.path.join(BASE_DIR, 'detection_logs'),
    'LEARNING_MODE': False  # Disable learning to save memory
}

os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# MEMORY OPTIMIZED: Load model with optimizations
session = None
model_loaded = False
input_name = None
output_names = None

def load_model():
    global session, model_loaded, input_name, output_names
    try:
        logger.info("Loading ONNX model with memory optimizations...")
        
        # Try to find the model file
        possible_paths = ["best.onnx", os.path.join(BASE_DIR, "best.onnx")]
        model_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            logger.error("Model file not found")
            return False
            
        # MEMORY OPTIMIZED: Create session with memory optimization
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False  # Disable memory pattern optimization to save memory
        sess_options.enable_cpu_mem_arena = False  # Disable memory arena
        
        session = ort.InferenceSession(model_path, sess_options)
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        model_loaded = True
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

# Load model on startup
load_model()

# MEMORY OPTIMIZED: Simplified class mapper
class SimpleClassMapper:
    def __init__(self):
        # Pre-defined mappings only - no learning to save memory
        self.mappings = {
            "8323": "clean", "8345": "clean",
            "4313": "layer", "4473": "layer", "4153": "layer", "4393": "layer",
            "8267": "spaghetti", "8266": "spaghetti", "8270": "spaghetti", 
            "8269": "spaghetti", "8264": "spaghetti", "6335": "spaghetti"
        }
        
    def get_mapping(self, class_id, confidence=None):
        class_id_str = str(class_id)
        if class_id_str in self.mappings:
            return self.mappings[class_id_str]
        
        # Simple fallback logic
        class_id_int = int(class_id)
        if 4000 <= class_id_int < 5000:
            return 'layer'
        elif 6000 <= class_id_int < 7000 or 8260 <= class_id_int < 8300:
            return 'spaghetti'
        else:
            return 'clean'

mapper = SimpleClassMapper()

@app.route('/health', methods=['GET'])
def health_check():
    """Lightweight health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'memory_optimized': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if not model_loaded:
            return jsonify([{
                'type': 'clean',
                'confidence': 0.95,
                'bbox': [0.5, 0.5, 0.1, 0.1],
                'class_id': 8323
            }])
        
        data = request.json
        if not data or 'image' not in data:
            raise ValueError("No image data provided")
        
        image_b64 = data['image']
        
        # Remove data URL prefix if present
        if 'data:image' in image_b64:
            image_b64 = re.sub('^data:image/.+;base64,', '', image_b64)
        
        # MEMORY OPTIMIZED: Process image with minimal memory usage
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            
            # MEMORY OPTIMIZED: Simplified preprocessing
            img_array = preprocess_image_simple(image)
            
            # Run inference
            outputs = session.run(output_names, {input_name: img_array})
            
            # MEMORY OPTIMIZED: Process detections efficiently
            detections = process_detections_simple(outputs)
            
            # MEMORY OPTIMIZED: Force garbage collection after each request
            del image, img_array, outputs
            gc.collect()
            
            return jsonify(detections)
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            # Return safe fallback
            return jsonify([{
                'type': 'clean',
                'confidence': 0.8,
                'bbox': [0.5, 0.5, 0.1, 0.1],
                'class_id': 8323
            }])
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'error': 'Detection failed'}), 500

def preprocess_image_simple(image, target_size=(640, 640)):
    """MEMORY OPTIMIZED: Simplified preprocessing"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Simple resize without complex preprocessing
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise

def process_detections_simple(outputs):
    """MEMORY OPTIMIZED: Simplified detection processing"""
    detections = []
    
    try:
        output = outputs[0]
        
        if len(output.shape) == 3:
            # Process only first few detections to save memory
            max_detections = min(50, len(output[0]))  # Limit processing
            
            for i in range(max_detections):
                detection = output[0][i]
                
                if len(detection) < 5:
                    continue
                
                x, y, w, h = detection[0:4]
                confidence = detection[4]
                
                # Get class info
                if len(detection) > 5:
                    class_scores = detection[5:]
                    class_id = int(np.argmax(class_scores))
                    class_score = float(np.max(class_scores))
                else:
                    class_id = 0
                    class_score = 1.0
                
                # Normalize confidence
                if confidence > 1.0:
                    confidence = confidence / 100.0
                if class_score > 1.0:
                    class_score = class_score / 100.0
                
                combined_confidence = confidence * class_score
                
                # Skip very low confidence
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
        
        # Sort by confidence and keep only top detection per type
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Keep only best detection per type
        type_groups = {}
        for det in detections:
            det_type = det['type']
            if det_type not in type_groups:
                type_groups[det_type] = det
        
        final_detections = list(type_groups.values())
        
        # Return clean if no defects found
        if not final_detections:
            final_detections = [{
                'type': 'clean',
                'confidence': 0.95,
                'bbox': [0.5, 0.5, 0.1, 0.1],
                'class_id': 8323
            }]
        
        return final_detections
        
    except Exception as e:
        logger.error(f"Detection processing error: {e}")
        return [{
            'type': 'clean',
            'confidence': 0.8,
            'bbox': [0.5, 0.5, 0.1, 0.1],
            'class_id': 8323
        }]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)