# FarmSentry 🐄

**Real-time Animal Detection and Alert System for Agricultural Settings**

FarmSentry is an intelligent monitoring system that uses YOLOv8 object detection to identify animals in farm environments and provide real-time visual alerts. Built for both CPU and GPU deployment, it's designed to help farmers monitor livestock and detect potential threats or intrusions.

## 🚀 Features

- **Real-time Detection**: Process video feeds with YOLOv8 for accurate animal identification
- **Flexible Deployment**: Support for both CPU and GPU execution
- **Smart Alerts**: Time-based alert system with cooldown periods
- **Configurable Filtering**: Ignore specific object classes and set minimum detection thresholds
- **Performance Optimized**: Frame skipping and detection size tuning for optimal performance
- **Visual Feedback**: Clear bounding boxes, labels, and alert indicators

## 📁 Project Structure

```
farmsentry/
├── main_cpu.py                    # CPU entry point
├── main_gpu.py                    # GPU entry point
├── README.md                      # This file
├── CLAUDE.md                      # Development guidelines
├── src/
│   ├── core/
│   │   ├── detection_module.py    # AnimalDetector class
│   │   └── alert_module.py        # AlertHandler class
│   ├── config/
│   │   ├── config_cpu.py          # CPU configuration
│   │   └── config_gpu.py          # GPU configuration
│   └── main/
│       ├── main_cpu.py            # CPU main application
│       └── main_gpu.py            # GPU main application
├── models/
│   └── yolov8m.pt                 # YOLOv8 medium model
├── data/
│   └── sample_videos/             # Test video files
│       ├── sample_video_1.mp4
│       ├── sample_video_2.mp4
│       ├── sample_video_3.mp4
│       └── sample_video_4.mp4
└── requirements/
    ├── requirements-cpu.txt       # CPU dependencies
    └── requirements-gpu.txt       # GPU dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- For GPU version: CUDA 11.8+ compatible GPU

### CPU Version

1. Clone the repository:
```bash
git clone <repository-url>
cd farmsentry
```

2. Install CPU dependencies:
```bash
pip install -r requirements/requirements-cpu.txt
```

3. Run the application:
```bash
python main_cpu.py
```

### GPU Version

1. Install GPU dependencies (requires CUDA 11.8):
```bash
pip install -r requirements/requirements-gpu.txt
```

2. Run the application:
```bash
python main_gpu.py
```

## ⚙️ Configuration

### CPU Configuration (`src/config/config_cpu.py`)

```python
# Video Input
VIDEO_PATH = "data/sample_videos/sample_video_4.mp4"

# Detection Parameters
CONFIDENCE_THRESHOLD = 0.40  # Detection confidence threshold
IOU_THRESHOLD = 0.50         # Non-maximum suppression threshold
DETECTION_SIZE = 640         # Input image size for detection
MIN_ANIMAL_AREA = 1500       # Minimum bounding box area

# System Settings
DEVICE = "cpu"               # Processing device
FRAME_SKIP = 1              # Process every nth frame

# Alert Settings
ALERT_DURATION = 10         # Alert duration in seconds
ALERT_COOLDOWN = 5          # Cooldown between alerts

# Classes to Ignore
OMIT_CLASSES = [
    "person", "bird", "car", "chair", "bicycle", 
    "motorcycle", "suitcase", "bench", "bottle", 
    "vase", "truck", "plant"
]
```

### GPU Configuration (`src/config/config_gpu.py`)

Similar to CPU configuration but with `DEVICE = "cuda"`.

## 🎮 Usage

### Basic Operation

1. **Start the application**:
   ```bash
   python main_cpu.py  # For CPU version
   python main_gpu.py  # For GPU version
   ```

2. **Monitor the video feed**: The application will display a window showing the processed video with:
   - Red bounding boxes around detected animals
   - Class labels with confidence scores
   - Visual alerts (red borders and flashing text)

3. **Exit**: Press 'q' to quit the application

### Customizing Detection

1. **Change video source**: Modify `VIDEO_PATH` in the configuration file
2. **Adjust sensitivity**: Lower `CONFIDENCE_THRESHOLD` for more detections
3. **Filter objects**: Add unwanted classes to `OMIT_CLASSES`
4. **Optimize performance**: Increase `FRAME_SKIP` or reduce `DETECTION_SIZE`

## 🔧 Code Usage

### Using Detection Module

```python
from src.core.detection_module import AnimalDetector
import cv2

detector = AnimalDetector()
cap = cv2.VideoCapture("your_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame, detections = detector.process_frame(frame)
    cv2.imshow('Detection', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Using Alert System

```python
from src.core.alert_module import AlertHandler

alert_handler = AlertHandler()
alert_handler.update_alerts(detections)
alert_frame = alert_handler.add_visual_alerts(processed_frame)
```

## 🐛 Troubleshooting

### Common Issues

1. **"Could not open video" error**:
   - Check that the video file exists at the specified path
   - Ensure the video format is supported by OpenCV
   - Verify file permissions

2. **Poor detection performance**:
   - Increase `CONFIDENCE_THRESHOLD` to reduce false positives
   - Adjust `DETECTION_SIZE` (higher = more accurate, slower)
   - Check lighting conditions in the video

3. **GPU version not working**:
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
   - Ensure CUDA version compatibility

4. **Low FPS**:
   - Increase `FRAME_SKIP` to process fewer frames
   - Reduce `DETECTION_SIZE` for faster processing
   - Consider switching to CPU version for lighter workloads

### Performance Optimization

- **For real-time processing**: Use `FRAME_SKIP = 2-3` and `DETECTION_SIZE = 320-640`
- **For accuracy**: Use `FRAME_SKIP = 0` and `DETECTION_SIZE = 1280`
- **For resource-constrained systems**: Use CPU version with `DETECTION_SIZE = 320`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection model
- [OpenCV](https://opencv.org/) for video processing capabilities
- [PyTorch](https://pytorch.org/) for deep learning framework

## 📞 Support

For questions or issues, please:
1. Check the troubleshooting section above
2. Review existing issues in the repository
3. Create a new issue with detailed information about your problem

---

**Happy Farming! 🌾**

This is an added line for demonstration purposes.