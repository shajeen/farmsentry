"""
Configuration for FarmSentry: Farmland Animal Alert System (GPU Version)
"""

# -- Video Input --
VIDEO_PATH = "data/sample_videos/sample_video_4.mp4"  # Path to the video file for processing

# -- Detection Parameters --
CONFIDENCE_THRESHOLD = 0.40  # Minimum confidence for a detection to be considered valid
IOU_THRESHOLD = 0.50         # Intersection Over Union threshold for non-maximum suppression
DETECTION_SIZE = 640         # Image size for detection (e.g., 320, 640, 1280)
MIN_ANIMAL_AREA = 1500       # Minimum bounding box area (in pixels) to filter out small objects

# -- Classes to Ignore --
# A list of object classes that should be ignored by the detection system.
OMIT_CLASSES = [
    "person", "bird", "car", "chair", "bicycle", "motorcycle", 
    "suitcase", "bench", "bottle", "vase", "truck", "plant"
]

# -- System and Performance --
DEVICE = "cuda"  # "cpu" or "cuda" for GPU
FRAME_SKIP = 1  # Process every nth frame (e.g., 1 for all, 2 for every other)

# -- Alerting --
ALERT_DURATION = 10  # Seconds to keep an alert active after the last detection
ALERT_COOLDOWN = 5   # Seconds to wait before re-triggering a visual alert