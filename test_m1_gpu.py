import cv2

# === Detection Settings ===
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
AGNOSTIC_NMS = False  # Non-Maximum Suppression mode

# === Tracking Settings (DeepSORT) ===
MAX_DISAPPEARED = 30
MAX_DISTANCE = 50
TRACKER_MAX_IOU_DISTANCE = 0.7
TRACKER_MAX_AGE = 70
TRACKER_N_INIT = 3
TRACKER_MAX_COSINE_DISTANCE = 0.2

# === Counting Settings ===
COUNTING_LINE_THICKNESS = 3
COUNTING_LINE_COLOR = (0, 255, 255)  # Yellow in BGR
COUNTING_REGION_COLOR = (255, 0, 255)  # Magenta in BGR
ENABLE_BIDIRECTIONAL_COUNTING = True
COUNT_CONFIDENCE_THRESHOLD = 0.5

# === Visualization Settings ===
BBOX_THICKNESS = 2
BBOX_COLOR = (0, 255, 0)  # Green in BGR
TEXT_COLOR = (255, 255, 255)  # White in BGR
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
TRAIL_LENGTH = 30
TRAIL_THICKNESS = 2
TRAIL_COLOR = (0, 0, 255)  # Red in BGR

# === Performance Optimization Settings ===
RESIZE_FRAME = False
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480
TARGET_FPS = 30
SKIP_FRAMES = 0  # Skip every N frames for performance (0 = no skipping)
ENABLE_FRAME_BUFFER = True
BUFFER_SIZE = 5

# === Input/Output Settings ===
SAVE_OUTPUT = False
OUTPUT_DIR = "output"
PROCESSED_VIDEO_DIR = "output/processed_videos"
SCREENSHOT_DIR = "output/screenshots"
LOG_DIR = "logs"
OUTPUT_FORMAT = "mp4v"  # Codec for video output
OUTPUT_QUALITY = 90  # JPEG quality for screenshots

# === UI and Display Settings ===
WINDOW_NAME = "Object Tracking & Counting"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
STATS_POSITION = (10, 30)
SHOW_FPS = True
SHOW_COUNT = True
SHOW_IDS = True
SHOW_TRAILS = True
SHOW_CONFIDENCE = True
FULLSCREEN = False

# === Memory Management ===
ENABLE_MEMORY_CLEANUP = True
MEMORY_CLEANUP_INTERVAL = 100  # Frames
MAX_MEMORY_USAGE_GB = 8  # Maximum memory usage before cleanup

# === Apple Silicon (M1/M2) Optimizations ===
MPS_OPTIMIZATIONS = {
    'empty_cache_interval': 50,  # Empty MPS cache every N frames
    'use_amp': True,  # Automatic Mixed Precision
    'compile_model': False,  # Torch compile (experimental)
    'memory_fraction': 0.8  # Fraction of GPU memory to use
}

# === CUDA Optimizations ===
CUDA_OPTIMIZATIONS = {
    'benchmark': True,  # Enable cuDNN benchmark
    'deterministic': False,  # Deterministic algorithms
    'empty_cache_interval': 100,
    'memory_fraction': 0.9
}

# === Logging Configuration ===
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "logs/tracking.log"
ENABLE_LOGGING = True
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
MAX_LOG_SIZE_MB = 10
LOG_BACKUP_COUNT = 3

# === Classes and Detection ===
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# === Common Object Classes for Tracking ===
COMMON_CLASSES = {
    'person': 0,
    'car': 2,
    'motorcycle': 3,
    'bus': 5,
    'truck': 7,
    'bicycle': 1
}

# === Counting Zones Presets ===
PRESET_COUNTING_LINES = {
    'horizontal_center': lambda w, h: [(0, h // 2), (w, h // 2)],
    'vertical_center': lambda w, h: [(w // 2, 0), (w // 2, h)],
    'diagonal_tl_br': lambda w, h: [(0, 0), (w, h)],
    'diagonal_tr_bl': lambda w, h: [(w, 0), (0, h)]
}

# === Advanced Features ===
ENABLE_HEATMAP = False
HEATMAP_ALPHA = 0.3
ENABLE_TRAIL_HISTORY = True
ENABLE_ZONE_ANALYTICS = False
ENABLE_SPEED_ESTIMATION = False
SPEED_ESTIMATION_PIXELS_PER_METER = 10

# === Export Settings ===
EXPORT_STATISTICS = True
STATISTICS_FORMAT = "json"  # json, csv, txt
EXPORT_INTERVAL = 1000  # Export stats every N frames
AUTO_SAVE_SCREENSHOTS = False
SCREENSHOT_INTERVAL = 500  # Auto screenshot every N frames

# === Development and Debug ===
DEBUG_MODE = False
SHOW_DEBUG_INFO = False
PROFILE_PERFORMANCE = False
ENABLE_TENSORBOARD = False
TENSORBOARD_LOG_DIR = "runs"

# === Platform-Specific Settings ===
PLATFORM_SETTINGS = {
    'macos': {
        'use_metal': True,
        'optimize_for_m1': True,
        'preferred_device': 'mps'
    },
    'windows': {
        'use_directml': False,
        'preferred_device': 'cuda'
    },
    'linux': {
        'preferred_device': 'cuda'
    }
}

# === Validation and Constraints ===
MIN_DETECTION_SIZE = 10  # Minimum bounding box size in pixels
MAX_DETECTION_SIZE = 1000  # Maximum bounding box size in pixels
MIN_CONFIDENCE_FOR_TRACKING = 0.3
MAX_TRACKS_PER_FRAME = 100

# === Quality Control ===
ENABLE_QUALITY_FILTER = True
BLUR_THRESHOLD = 100  # Laplacian variance threshold for blur detection
MIN_OBJECT_LIFETIME = 5  # Minimum frames an object should exist

# === Color Schemes ===
COLOR_SCHEMES = {
    'default': {
        'bbox': (0, 255, 0),
        'text': (255, 255, 255),
        'line': (0, 255, 255),
        'trail': (0, 0, 255)
    },
    'dark': {
        'bbox': (100, 255, 100),
        'text': (200, 200, 200),
        'line': (100, 255, 255),
        'trail': (100, 100, 255)
    },
    'high_contrast': {
        'bbox': (0, 255, 0),
        'text': (255, 255, 0),
        'line': (255, 0, 255),
        'trail': (255, 165, 0)
    }
}

CURRENT_COLOR_SCHEME = 'default'


# === Auto-Configuration Based on Hardware ===
def auto_configure_for_hardware():
    """Auto-configure settings based on detected hardware"""
    import torch
    import platform

    # Detect platform
    system = platform.system().lower()

    # Configure device
    if system == 'darwin':  # macOS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            globals()['DEVICE'] = 'mps'
            globals()['HALF_PRECISION'] = True
        else:
            globals()['DEVICE'] = 'cpu'
    elif torch.cuda.is_available():
        globals()['DEVICE'] = 'cuda'
        globals()['HALF_PRECISION'] = True
    else:
        globals()['DEVICE'] = 'cpu'
        globals()['HALF_PRECISION'] = False

    # Adjust batch size based on device
    if globals()['DEVICE'] == 'cpu':
        globals()['BATCH_SIZE'] = 1
        globals()['TARGET_FPS'] = 15
    elif globals()['DEVICE'] == 'mps':
        globals()['BATCH_SIZE'] = 1  # Keep at 1 for MPS stability
        globals()['TARGET_FPS'] = 30
    else:  # CUDA
        globals()['BATCH_SIZE'] = 2
        globals()['TARGET_FPS'] = 30


# Auto-configure on import
auto_configure_for_hardware()