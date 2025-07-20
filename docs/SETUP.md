# FarmSentry Setup Guide

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for dependencies and models
- **CPU**: Multi-core processor recommended

### GPU Requirements (Optional)

- **NVIDIA GPU**: CUDA-compatible GPU with Compute Capability 3.5+
- **CUDA**: Version 11.8 or compatible
- **cuDNN**: Compatible version with CUDA
- **VRAM**: 4GB minimum, 8GB+ recommended for optimal performance

## Installation Methods

### Method 1: Standard Installation

#### Step 1: Clone Repository

```bash
git clone <repository-url>
cd farmsentry
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv farmsentry-env
source farmsentry-env/bin/activate  # Linux/macOS
# or
farmsentry-env\Scripts\activate     # Windows

# Using conda
conda create -n farmsentry python=3.9
conda activate farmsentry
```

#### Step 3: Install Dependencies

**For CPU-only deployment:**
```bash
pip install -r requirements/requirements-cpu.txt
```

**For GPU deployment:**
```bash
pip install -r requirements/requirements-gpu.txt
```

### Method 2: Manual Installation

If you prefer to install dependencies manually:

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python
pip install numpy

# For GPU support, install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verification

### Test Basic Installation

```bash
# Test Python imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; print('YOLOv8: OK')"
```

### Test GPU Installation (if applicable)

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Check NVIDIA driver
nvidia-smi
```

### Test FarmSentry Modules

```bash
# Test import structure
python -c "from src.core.detection_module import AnimalDetector; print('Detection module: OK')"
python -c "from src.core.alert_module import AlertHandler; print('Alert module: OK')"
```

## Configuration Setup

### Basic Configuration

1. **Video Source Configuration**:
   ```python
   # Edit src/config/config_cpu.py or src/config/config_gpu.py
   VIDEO_PATH = "data/sample_videos/sample_video_4.mp4"  # Use sample video
   # or
   VIDEO_PATH = "/path/to/your/video.mp4"  # Use custom video
   # or
   VIDEO_PATH = 0  # Use webcam (device 0)
   ```

2. **Performance Tuning**:
   ```python
   # For faster processing (lower quality)
   DETECTION_SIZE = 320
   FRAME_SKIP = 2
   
   # For better accuracy (slower processing)
   DETECTION_SIZE = 1280
   FRAME_SKIP = 0
   ```

3. **Detection Sensitivity**:
   ```python
   # More sensitive (more detections, possible false positives)
   CONFIDENCE_THRESHOLD = 0.25
   
   # Less sensitive (fewer detections, more reliable)
   CONFIDENCE_THRESHOLD = 0.60
   ```

### Advanced Configuration

1. **Custom Object Classes**:
   ```python
   # Add classes to ignore
   OMIT_CLASSES = [
       "person", "bird", "car", "chair", "bicycle", 
       "motorcycle", "suitcase", "bench", "bottle", 
       "vase", "truck", "plant", "tv", "laptop"
   ]
   ```

2. **Alert Customization**:
   ```python
   ALERT_DURATION = 15    # Keep alerts visible for 15 seconds
   ALERT_COOLDOWN = 3     # Wait 3 seconds before re-alerting
   ```

3. **Hardware Optimization**:
   ```python
   # CPU configuration
   DEVICE = "cpu"
   FRAME_SKIP = 2         # Process every 3rd frame
   
   # GPU configuration  
   DEVICE = "cuda"
   FRAME_SKIP = 0         # Process all frames
   ```

## First Run

### Using Sample Videos

```bash
# Run with CPU
python main_cpu.py

# Run with GPU (if available)
python main_gpu.py
```

The application will:
1. Load the YOLOv8 model (download if first time)
2. Open the configured video file
3. Display a window with real-time detection results
4. Show FPS and processing statistics

### Controls

- **Exit**: Press 'q' to quit the application
- **Window**: Click the video window to focus it before using keyboard controls

## Troubleshooting Setup Issues

### Common Installation Problems

1. **"No module named 'torch'" Error**:
   ```bash
   # Reinstall PyTorch
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio
   ```

2. **"No module named 'cv2'" Error**:
   ```bash
   # Install OpenCV
   pip install opencv-python
   # If issues persist, try:
   pip install opencv-python-headless
   ```

3. **CUDA Not Available**:
   ```bash
   # Check NVIDIA driver
   nvidia-smi
   
   # Reinstall CUDA-enabled PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **"Could not open video" Error**:
   - Check video file path in configuration
   - Ensure video format is supported (MP4, AVI, MOV)
   - Test with sample videos first

### Performance Issues

1. **Low FPS on CPU**:
   - Increase `FRAME_SKIP` value
   - Reduce `DETECTION_SIZE` to 320
   - Close other applications

2. **GPU Not Being Used**:
   ```python
   # Verify in configuration file
   DEVICE = "cuda"  # Not "gpu"
   ```

3. **High Memory Usage**:
   - Reduce `DETECTION_SIZE`
   - Increase `FRAME_SKIP`
   - Use CPU version for lighter workload

### Video Source Issues

1. **Webcam Not Working**:
   ```python
   # Try different camera indices
   VIDEO_PATH = 0  # Default camera
   VIDEO_PATH = 1  # Secondary camera
   ```

2. **Video Format Not Supported**:
   ```bash
   # Convert video using ffmpeg
   ffmpeg -i input_video.avi -c:v libx264 -c:a aac output_video.mp4
   ```

3. **Network Camera/RTSP Stream**:
   ```python
   VIDEO_PATH = "rtsp://username:password@camera_ip:554/stream"
   ```

## Performance Optimization

### CPU Optimization

```python
# config_cpu.py optimizations
DETECTION_SIZE = 320      # Smaller input size
FRAME_SKIP = 2           # Skip frames
CONFIDENCE_THRESHOLD = 0.5  # Higher threshold
```

### GPU Optimization

```python
# config_gpu.py optimizations
DETECTION_SIZE = 640     # Balanced size
FRAME_SKIP = 0          # Process all frames
CONFIDENCE_THRESHOLD = 0.4  # Lower threshold for better detection
```

### System-Level Optimization

1. **CPU Usage**:
   - Close unnecessary applications
   - Use Task Manager/Activity Monitor to monitor CPU usage
   - Consider upgrading to multi-core processor

2. **Memory Management**:
   - Monitor RAM usage during execution
   - Close browser tabs and other memory-intensive applications
   - Consider increasing virtual memory/swap space

3. **Storage Optimization**:
   - Use SSD for better I/O performance
   - Ensure sufficient free space (>10% of drive)

## Next Steps

After successful setup:

1. **Test with Your Data**:
   - Add your own video files to `data/sample_videos/`
   - Update configuration to use your videos
   - Test detection accuracy with your specific use case

2. **Customize for Your Needs**:
   - Modify `OMIT_CLASSES` for your environment
   - Adjust detection thresholds based on testing
   - Configure alert timing for your workflow

3. **Integration**:
   - Review the API documentation for custom integrations
   - Consider building custom alert handlers
   - Explore automation possibilities

4. **Production Deployment**:
   - Set up proper logging
   - Configure error handling
   - Plan for system monitoring and maintenance

For additional help, refer to the main README.md file or create an issue in the repository.