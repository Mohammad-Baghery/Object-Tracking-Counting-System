
# YOLOv8 Detection Settings
YOLO_MODEL = "yolov8n.pt"  # Model size: n(ano), s(mall), m(edium), l(arge), x(tra-large)
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
DEVICE = "auto"  # "auto", "cpu", "cuda", "mps"

# Object Classes to Track (COCO Dataset)
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
TRACK_CLASSES = [0, 1, 2, 3, 5, 7]

# DeepSORT Tracking Settings
MAX_DISAPPEARED = 50
MAX_DISTANCE = 100
TRACKER_CONFIG = {
    'max_age': 50,
    'n_init': 3,
    'nn_budget': None,
    'nms_max_overlap': 1.0,
    'max_cosine_distance': 0.4,
}

# Counting Settings
COUNTING_LINE_POSITION = 400  # Y coordinate for horizontal counting line
COUNTING_LINE_THICKNESS = 3
COUNTING_LINE_COLOR = (0, 255, 255)  # Yellow in BGR

# Visualization Settings
BBOX_COLOR = (0, 255, 0)  # Green in BGR
BBOX_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White
TEXT_THICKNESS = 2
TEXT_SIZE = 0.7

# Output Settings
SAVE_OUTPUT = False
OUTPUT_PATH = "output/result.mp4"
OUTPUT_FPS = 30

# Performance Settings
SKIP_FRAMES = 0  # Skip every N frames to improve performance
RESIZE_FRAME = None  # (width, height) or None for original size

# COCO Class Names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
