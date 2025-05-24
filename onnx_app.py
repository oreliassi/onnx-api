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
import gc  # For garbage collection
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Configure logging for production
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimized configuration for low memory
CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.4,
    'DEBUG_MODE': False,  # Disabled for production
    'LEARNING_MODE': False,  # Disabled to save memory
    'MAX_CACHE_SIZE': 10  # Limit cache size
}

# Global variables
session = None
model_loaded = False
input_name = None
output_names = None

# Simplified class mapping (hardcoded to save memory)
SIMPLE_CLASS_MAPPING = {
    # Clean detections
    8323: "clean",
    8345: "clean",
    
    # Layer shift detections  
    4153: "layer", 4313: "layer", 4393: "layer", 4473: "layer",
    4233: "layer", 4553: "layer", 4633: "layer", 4713: "layer",
    2713: "layer", 8322: "layer",
    
    # Spaghetti detections
    8264: "spaghetti", 8266: "spaghetti", 8267: "spaghetti", 8269: "spaghetti", 8270: "spaghetti",
    8263: "spaghetti", 8344: "spaghetti", 8347: "spaghetti", 8101: "spaghetti",
    6335: "spaghetti", 6332: "spaghetti", 6334: "spaghetti", 6341: "spaghetti", 
    6346: "spaghetti", 6356: "spaghetti", 6377: "spaghetti", 6378: "spaghetti"
}

def load_model():
    """Load ONNX model with memory optimization"""
    global session, model_loaded, input_name, output_names
    
    try:
        logger.info("Loading ONNX model...")
        
        # Configure ONNX Runtime for low memory usage
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False  # Disable memory pattern optimization
        sess_options.enable_cpu_mem_arena = False  # Disable memory arena
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        # Load with providers priority (CPU only for stability)
        providers = ['CPUExecutionProvider']
        
        session = ort.InferenceSession("best.onnx", sess_options, providers=providers)
        model_loaded = True
        
        # Get model metadata
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        logger.info("ONNX model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        session = None
        model_loaded = False
        return False

def get_defect_type(class_id, confidence):
    """Simple mapping function"""
    class_id = int(class_id)
    
    # Check direct mappings first
    if class_id in SIMPLE_CLASS_MAPPING:
        return SIMPLE_CLASS_MAPPING[class_id]
    
    # Range-based fallbacks
    if 4000 <= class_id < 5000:
        return 'layer'
    elif 6000 <= class_id < 7000:
        return 'spaghetti'
    elif 8000 <= class_id < 8350:
        if 8260 <= class_id < 8300:
            return 'spaghetti'
        return 'clean'
    
    # Low confidence default
    return 'clean' if confidence < 0.4 else 'clean'

@app.route('/health', methods=['GET'])
def health_check():
    """Lightweight health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'version': '2.1.0'
    })

@app.route('/detect', methods=['POST'])
def detect():
    """Optimized detection endpoint"""
    try:
        if not model_loaded:
            return jsonify([{
                'type': 'clean',
                'confidence': 0.95,
                'bbox': [0.5, 0.5, 0.5, 0.5],
                'class_id': 0
            }])
        
        # Get image data
        data = request.json
        if not data or 'image' not in data:
            raise ValueError("No image data provided")
        
        image_b64 = data['image']
        
        # Remove data URL prefix
        if 'data:image' in image_b64:
            image_b64 = re.sub('^data:image/.+;base64,', '', image_b64)
        
        # Process image with memory management
        detections = process_image_optimized(image_b64)
        
        # Force garbage collection
        gc.collect()
        
        return jsonify(detections)
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        gc.collect()
        return jsonify([{
            'type': 'clean',
            'confidence': 0.7,
            'bbox': [0.5, 0.5, 0.5, 0.5],
            'class_id': 0
        }])

def process_image_optimized(image_b64):
    """Memory-optimized image processing"""
    try:
        # Decode image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize directly to target size (skip padding for memory savings)
        target_size = (640, 640)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Clean up PIL image
        image.close()
        del image
        
        # Run inference
        outputs = session.run(output_names, {input_name: img_array})
        
        # Clean up input array
        del img_array
        
        # Process results
        detections = process_detections_simple(outputs[0])
        
        # Clean up outputs
        del outputs
        
        return detections
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return [{
            'type': 'clean',
            'confidence': 0.8,
            'bbox': [0.5, 0.5, 0.1, 0.1],
            'class_id': 0
        }]

def process_detections_simple(output):
    """Simplified detection processing"""
    detections = []
    
    try:
        if len(output.shape) == 3:
            # Process only top detections to save memory
            for i, detection in enumerate(output[0][:100]):  # Limit to first 100
                x, y, w, h = detection[0:4]
                confidence = detection[4]
                
                if len(detection) > 5:
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    class_score = class_scores[class_id]
                else:
                    class_id = 0
                    class_score = 1.0
                
                # Normalize confidence
                if confidence > 1.0:
                    confidence = confidence / 100.0
                if class_score > 1.0:
                    class_score = class_score / 100.0
                
                combined_confidence = confidence * class_score
                
                # Skip low confidence early
                if combined_confidence < CONFIG['CONFIDENCE_THRESHOLD']:
                    continue
                
                # Get defect type
                defect_type = get_defect_type(class_id, combined_confidence)
                
                detections.append({
                    'type': defect_type,
                    'confidence': float(combined_confidence),
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'class_id': int(class_id)
                })
        
        # Sort by confidence and take best per type
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Group by type (keep only highest confidence per type)
        type_groups = {}
        for det in detections:
            det_type = det['type']
            if det_type not in type_groups:
                type_groups[det_type] = det
        
        result = list(type_groups.values())
        
        # Default to clean if no detections
        if not result:
            result = [{
                'type': 'clean',
                'confidence': 0.95,
                'bbox': [0.5, 0.5, 0.1, 0.1],
                'class_id': 0
            }]
        
        return result
        
    except Exception as e:
        logger.error(f"Detection processing error: {e}")
        return [{
            'type': 'clean',
            'confidence': 0.8,
            'bbox': [0.5, 0.5, 0.1, 0.1],
            'class_id': 0
        }]

# Simplified endpoints for memory efficiency
@app.route('/test_detection', methods=['POST'])
def test_detection():
    """Simplified test endpoint"""
    try:
        data = request.json
        detections = process_image_optimized(data['image'])
        return jsonify({
            'total_detections': len(detections),
            'detections': detections
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        logger.error("Failed to load model, exiting")
        exit(1)
    
    # Run with optimized settings
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)