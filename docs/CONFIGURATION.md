# FarmSentry Configuration Guide

## Overview

FarmSentry uses separate configuration files for CPU and GPU deployments, allowing fine-tuned optimization for different hardware setups. This guide covers all configuration options and provides recommendations for various use cases.

## Configuration Files

- `src/config/config_cpu.py` - CPU-specific settings
- `src/config/config_gpu.py` - GPU-specific settings

Both files share the same structure but with different default values optimized for their respective hardware.

## Configuration Parameters

### Video Input Settings

#### VIDEO_PATH
**Type**: `str`  
**Default**: `"data/sample_videos/sample_video_4.mp4"`  
**Description**: Path to the input video file or camera device.

**Options**:
```python
# Local video file
VIDEO_PATH = "data/sample_videos/sample_video_1.mp4"
VIDEO_PATH = "/absolute/path/to/video.mp4"
VIDEO_PATH = "relative/path/to/video.mp4"

# Webcam (device index)
VIDEO_PATH = 0    # Default camera
VIDEO_PATH = 1    # Secondary camera

# Network camera/RTSP stream
VIDEO_PATH = "rtsp://user:pass@192.168.1.100:554/stream"
VIDEO_PATH = "http://192.168.1.100:8080/video"
```

**Recommendations**:
- Use MP4 format for best compatibility
- Test with sample videos before using custom content
- For webcam, start with device 0 and increment if needed

### Detection Parameters

#### CONFIDENCE_THRESHOLD
**Type**: `float`  
**Range**: `0.0 - 1.0`  
**Default**: `0.40`  
**Description**: Minimum confidence score for a detection to be considered valid.

**Impact**:
- **Lower values** (0.2-0.4): More detections, higher chance of false positives
- **Higher values** (0.5-0.8): Fewer detections, more reliable results

**Use Cases**:
```python
CONFIDENCE_THRESHOLD = 0.25  # High sensitivity for security monitoring
CONFIDENCE_THRESHOLD = 0.40  # Balanced for general farm monitoring  
CONFIDENCE_THRESHOLD = 0.60  # Conservative for critical applications
```

#### IOU_THRESHOLD
**Type**: `float`  
**Range**: `0.0 - 1.0`  
**Default**: `0.50`  
**Description**: Intersection over Union threshold for non-maximum suppression.

**Purpose**: Eliminates duplicate detections of the same object.

**Guidelines**:
```python
IOU_THRESHOLD = 0.30  # More aggressive suppression (fewer overlapping boxes)
IOU_THRESHOLD = 0.50  # Standard suppression (recommended)
IOU_THRESHOLD = 0.70  # Conservative suppression (allows more overlaps)
```

#### DETECTION_SIZE
**Type**: `int`  
**Options**: `320, 640, 1280` (or other valid YOLO input sizes)  
**Default**: `640`  
**Description**: Input image size for the YOLOv8 model.

**Performance vs. Accuracy Trade-off**:
```python
DETECTION_SIZE = 320   # Fast processing, lower accuracy
DETECTION_SIZE = 640   # Balanced performance and accuracy (recommended)
DETECTION_SIZE = 1280  # High accuracy, slower processing
```

**Hardware Recommendations**:
- **CPU**: 320-640 for real-time processing
- **GPU**: 640-1280 depending on GPU memory and performance requirements

#### MIN_ANIMAL_AREA
**Type**: `int`  
**Units**: pixels  
**Default**: `1500`  
**Description**: Minimum bounding box area to filter out small detections.

**Calculation**: `area = (x_max - x_min) * (y_max - y_min)`

**Adjustment Guidelines**:
```python
MIN_ANIMAL_AREA = 500   # Detect smaller objects (birds, small animals)
MIN_ANIMAL_AREA = 1500  # Standard threshold (medium-sized animals)
MIN_ANIMAL_AREA = 5000  # Large animals only (livestock, large wildlife)
```

### System Performance Settings

#### DEVICE
**Type**: `str`  
**Options**: `"cpu"`, `"cuda"`  
**Default**: 
- CPU config: `"cpu"`
- GPU config: `"cuda"`

**Important**: Ensure CUDA is available before using `"cuda"`.

#### FRAME_SKIP
**Type**: `int`  
**Default**: `1`  
**Description**: Process every nth frame. Value of 1 means process every other frame.

**Performance Impact**:
```python
FRAME_SKIP = 0   # Process all frames (highest accuracy, slowest)
FRAME_SKIP = 1   # Process every 2nd frame (50% processing load)
FRAME_SKIP = 2   # Process every 3rd frame (33% processing load)
FRAME_SKIP = 4   # Process every 5th frame (20% processing load)
```

**Use Cases**:
- **Real-time monitoring**: FRAME_SKIP = 1-2
- **Resource-constrained systems**: FRAME_SKIP = 3-5
- **High-accuracy analysis**: FRAME_SKIP = 0

### Alert System Configuration

#### ALERT_DURATION
**Type**: `int`  
**Units**: seconds  
**Default**: `10`  
**Description**: How long to keep an alert active after the last detection of an object class.

**Considerations**:
```python
ALERT_DURATION = 5   # Quick alerts for fast-moving animals
ALERT_DURATION = 10  # Standard duration (recommended)
ALERT_DURATION = 30  # Extended alerts for important detections
```

#### ALERT_COOLDOWN
**Type**: `int`  
**Units**: seconds  
**Default**: `5`  
**Description**: Minimum time between re-triggering visual alerts for the same class.

**Purpose**: Prevents alert fatigue from continuous detections.

```python
ALERT_COOLDOWN = 0   # No cooldown (continuous alerts)
ALERT_COOLDOWN = 5   # Standard cooldown (recommended)
ALERT_COOLDOWN = 30  # Long cooldown for non-critical monitoring
```

### Object Filtering

#### OMIT_CLASSES
**Type**: `list[str]`  
**Default**: 
```python
[
    "person", "bird", "car", "chair", "bicycle", "motorcycle", 
    "suitcase", "bench", "bottle", "vase", "truck", "plant"
]
```
**Description**: List of YOLO class names to ignore during detection.

**YOLO Object Classes** (commonly relevant for farms):
```python
# Animals (typically keep these)
"cow", "sheep", "horse", "dog", "cat", "bird"

# Vehicles (may want to omit or keep)
"car", "truck", "motorcycle", "bicycle"

# People (usually omit for animal-focused monitoring)
"person"

# Objects (usually omit)
"chair", "bench", "bottle", "vase", "suitcase", "backpack"
```

**Customization Examples**:
```python
# Livestock monitoring (keep only farm animals)
OMIT_CLASSES = [
    "person", "bird", "car", "chair", "bicycle", "motorcycle",
    "airplane", "bus", "train", "truck", "boat", "traffic light",
    "bench", "backpack", "umbrella", "handbag", "tie", "suitcase"
]

# Security monitoring (keep people and vehicles)
OMIT_CLASSES = [
    "chair", "bench", "bottle", "vase", "plant", "book",
    "clock", "scissors", "teddy bear", "hair drier"
]

# Wildlife monitoring (keep all animals, omit human objects)
OMIT_CLASSES = [
    "person", "car", "truck", "chair", "table", "laptop",
    "mouse", "remote", "keyboard", "cell phone"
]
```

## Configuration Profiles

### High Performance Profile (GPU Required)

```python
# config_gpu.py settings
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.50
DETECTION_SIZE = 1280
MIN_ANIMAL_AREA = 1000
DEVICE = "cuda"
FRAME_SKIP = 0
ALERT_DURATION = 15
ALERT_COOLDOWN = 3
```

**Use Case**: High-accuracy monitoring with powerful GPU hardware.

### Balanced Profile (Default)

```python
# Balanced settings for CPU or GPU
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.50
DETECTION_SIZE = 640
MIN_ANIMAL_AREA = 1500
FRAME_SKIP = 1
ALERT_DURATION = 10
ALERT_COOLDOWN = 5
```

**Use Case**: General-purpose farm monitoring with good performance-accuracy balance.

### Resource-Efficient Profile (CPU/Low-End Hardware)

```python
# config_cpu.py optimized settings
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.60
DETECTION_SIZE = 320
MIN_ANIMAL_AREA = 2000
DEVICE = "cpu"
FRAME_SKIP = 3
ALERT_DURATION = 8
ALERT_COOLDOWN = 10
```

**Use Case**: Older hardware or battery-powered deployments.

### Security Profile (High Sensitivity)

```python
# High-sensitivity security monitoring
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DETECTION_SIZE = 640
MIN_ANIMAL_AREA = 800
FRAME_SKIP = 0
ALERT_DURATION = 20
ALERT_COOLDOWN = 2
OMIT_CLASSES = ["chair", "bench", "bottle", "vase", "plant"]  # Keep people and vehicles
```

**Use Case**: Security applications where missing detections is more costly than false positives.

## Advanced Configuration

### Custom Video Sources

#### Multiple Camera Setup
```python
# Create separate config files for each camera
# config_camera1.py
VIDEO_PATH = 0
DEVICE = "cuda:0"

# config_camera2.py  
VIDEO_PATH = 1
DEVICE = "cuda:1"
```

#### Network Cameras
```python
# RTSP camera configuration
VIDEO_PATH = "rtsp://admin:password@192.168.1.100:554/h264/ch1/main/av_stream"

# HTTP camera configuration
VIDEO_PATH = "http://192.168.1.100:8080/video"

# For network cameras, consider increasing frame skip due to latency
FRAME_SKIP = 2
```

### Environment-Specific Tuning

#### Indoor Monitoring
```python
# Indoor barns, stables
CONFIDENCE_THRESHOLD = 0.45  # Higher due to consistent lighting
DETECTION_SIZE = 640
MIN_ANIMAL_AREA = 1200      # Animals typically closer to camera
```

#### Outdoor Monitoring  
```python
# Pastures, fields
CONFIDENCE_THRESHOLD = 0.35  # Lower due to variable lighting/weather
DETECTION_SIZE = 1280       # Higher for distant objects
MIN_ANIMAL_AREA = 2000      # Animals may be farther away
```

#### Night Vision/IR Monitoring
```python
# Infrared/night vision cameras
CONFIDENCE_THRESHOLD = 0.30  # Lower due to different image characteristics
IOU_THRESHOLD = 0.60        # More conservative overlap handling
MIN_ANIMAL_AREA = 1800      # Account for IR image quality
```

## Dynamic Configuration

### Runtime Configuration Changes

While not built into the current system, you can implement dynamic configuration:

```python
# Example: Modify configuration at runtime
import src.config.config_cpu as config

def update_sensitivity(new_threshold):
    config.CONFIDENCE_THRESHOLD = new_threshold
    print(f"Updated confidence threshold to {new_threshold}")

def toggle_frame_processing():
    config.FRAME_SKIP = 0 if config.FRAME_SKIP > 0 else 1
    print(f"Frame skip set to {config.FRAME_SKIP}")
```

### Configuration Validation

```python
def validate_config():
    """Validate configuration parameters."""
    assert 0.0 <= CONFIDENCE_THRESHOLD <= 1.0, "Invalid confidence threshold"
    assert 0.0 <= IOU_THRESHOLD <= 1.0, "Invalid IoU threshold"
    assert DETECTION_SIZE in [320, 640, 1280], "Invalid detection size"
    assert MIN_ANIMAL_AREA > 0, "Invalid minimum area"
    assert FRAME_SKIP >= 0, "Invalid frame skip value"
    assert DEVICE in ["cpu", "cuda"], "Invalid device"
```

## Configuration Best Practices

1. **Start with Defaults**: Begin with the provided configurations and adjust based on testing.

2. **Test Incrementally**: Change one parameter at a time to understand its impact.

3. **Monitor Performance**: Watch FPS and detection quality when adjusting settings.

4. **Document Changes**: Keep notes on why specific configurations were chosen.

5. **Environment-Specific Configs**: Create separate configurations for different deployment environments.

6. **Version Control**: Track configuration changes alongside code changes.

7. **Validation**: Always validate configuration parameters before deployment.

## Troubleshooting Configuration Issues

### Poor Detection Quality
- Lower `CONFIDENCE_THRESHOLD`
- Increase `DETECTION_SIZE`
- Reduce `MIN_ANIMAL_AREA`
- Review and adjust `OMIT_CLASSES`

### Too Many False Positives
- Raise `CONFIDENCE_THRESHOLD`
- Increase `MIN_ANIMAL_AREA`
- Add problematic classes to `OMIT_CLASSES`

### Performance Issues
- Increase `FRAME_SKIP`
- Reduce `DETECTION_SIZE`
- Switch from GPU to CPU (or vice versa)

### Alert Problems
- Adjust `ALERT_DURATION` and `ALERT_COOLDOWN`
- Check if detection quality issues are causing alert problems

This configuration guide should help you optimize FarmSentry for your specific use case and hardware setup.