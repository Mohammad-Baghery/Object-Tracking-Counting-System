
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import config


def draw_bounding_box(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                      track_id: int, class_name: str, confidence: float) -> np.ndarray:
    """
    Draw bounding box with tracking ID and class information

    Args:
        frame: Input frame
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        track_id: Tracking ID
        class_name: Object class name
        confidence: Detection confidence

    Returns:
        Frame with drawn bounding box
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), config.BBOX_COLOR, config.BBOX_THICKNESS)

    # Prepare label text
    label = f"ID:{track_id} {class_name} {confidence:.2f}"

    # Calculate label size and position
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SIZE, config.TEXT_THICKNESS
    )

    # Draw label background
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - 10),
        (x1 + text_width, y1),
        config.BBOX_COLOR,
        -1
    )

    # Draw label text
    cv2.putText(
        frame, label, (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.TEXT_SIZE,
        config.TEXT_COLOR,
        config.TEXT_THICKNESS
    )

    return frame


def draw_counting_line(frame: np.ndarray, line_position: int) -> np.ndarray:
    """
    Draw counting line on frame

    Args:
        frame: Input frame
        line_position: Y coordinate of counting line

    Returns:
        Frame with counting line
    """
    height, width = frame.shape[:2]

    # Draw counting line
    cv2.line(
        frame,
        (0, line_position),
        (width, line_position),
        config.COUNTING_LINE_COLOR,
        config.COUNTING_LINE_THICKNESS
    )

    # Add line label
    cv2.putText(
        frame,
        "COUNTING LINE",
        (10, line_position - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        config.COUNTING_LINE_COLOR,
        2
    )

    return frame


def draw_statistics(frame: np.ndarray, stats: Dict[str, int]) -> np.ndarray:
    """
    Draw counting statistics on frame

    Args:
        frame: Input frame
        stats: Dictionary with counting statistics

    Returns:
        Frame with statistics
    """
    height, width = frame.shape[:2]

    # Background for statistics
    overlay = frame.copy()
    cv2.rectangle(overlay, (width - 250, 10), (width - 10, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Draw statistics text
    y_offset = 35
    cv2.putText(frame, "STATISTICS", (width - 240, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    y_offset += 25
    for class_name, count in stats.items():
        cv2.putText(frame, f"{class_name}: {count}", (width - 240, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

    return frame


def get_center_point(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate center point of bounding box

    Args:
        bbox: Bounding box coordinates (x1, y1, x2, y2)

    Returns:
        Center point coordinates (cx, cy)
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm

    Args:
        point: Point coordinates (x, y)
        polygon: List of polygon vertices [(x1, y1), (x2, y2), ...]

    Returns:
        True if point is inside polygon, False otherwise
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize frame to target size while maintaining aspect ratio

    Args:
        frame: Input frame
        target_size: Target size (width, height)

    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    target_width, target_height = target_size

    # Calculate scaling factor
    scale = min(target_width / width, target_height / height)

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize frame
    resized = cv2.resize(frame, (new_width, new_height))

    # Create padded frame if needed
    if new_width != target_width or new_height != target_height:
        padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        return padded

    return resized


def calculate_fps(frame_count: int, elapsed_time: float) -> float:
    """
    Calculate frames per second

    Args:
        frame_count: Number of processed frames
        elapsed_time: Elapsed time in seconds

    Returns:
        FPS value
    """
    if elapsed_time == 0:
        return 0.0
    return frame_count / elapsed_time


def format_time(seconds: float) -> str:
    """
    Format time in seconds to HH:MM:SS format

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
