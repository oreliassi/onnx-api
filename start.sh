#!/bin/bash
echo "Starting deployment..."

# Set memory-efficient environment variables
export ONNXRUNTIME_MEMORY_PATTERN=0
export OMP_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=""
export PYTHONUNBUFFERED=1

echo "Environment configured for memory efficiency"
echo "Starting application..."

# Start the application
python onnx_app.py