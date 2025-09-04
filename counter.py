
import numpy as np
from typing import Dict, List, Tuple, Set
import config
from utils import get_center_point, point_in_polygon


class ObjectCounter:
    """Object counter using virtual counting line or region"""

    def __init__(self, counting_line_position: int = None, counting_region: List[Tuple[int, int]] = None):
        """
        Initialize object counter

        Args:
            counting_line_position: Y coordinate of horizontal counting line
            counting_region: List of polygon vertices for counting region
        """
        self.counting_line_position = counting_line_position or config.COUNTING_LINE_POSITION
        self.counting_region = counting_region

        # Tracking data
        self.tracked_objects = {}  # {track_id: {"class": class_name, "positions": [positions]}}
        self.counted_objects = set()  # Track IDs that have been counted
        self.crossing_objects = {}  # {track_id: "direction"} for line crossing detection

        # Statistics
        self.count_stats = {}  # {class_name: count}
        self.total_count = 0

    def update(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Update counter with new detections

        Args:
            detections: List of detection dictionaries with keys:
                       'bbox', 'track_id', 'class_name', 'confidence'

        Returns:
            Updated counting statistics
        """
        current_frame_ids = set()

        for detection in detections:
            track_id = detection['track_id']
            bbox = detection['bbox']
            class_name = detection['class_name']

            current_frame_ids.add(track_id)

            # Get center point of bounding box
            center_x, center_y = get_center_point(bbox)

            # Initialize tracking data for new objects
            if track_id not in self.tracked_objects:
                self.tracked_objects[track_id] = {
                    'class': class_name,
                    'positions': [],
                    'counted': False
                }

            # Update position history
            self.tracked_objects[track_id]['positions'].append((center_x, center_y))

            # Keep only recent positions (last 10 frames)
            if len(self.tracked_objects[track_id]['positions']) > 10:
                self.tracked_objects[track_id]['positions'] = \
                    self.tracked_objects[track_id]['positions'][-10:]

            # Check for counting conditions
            if not self.tracked_objects[track_id]['counted']:
                if self._should_count_object(track_id, center_x, center_y):
                    self._count_object(track_id, class_name)

        # Clean up disappeared objects
        self._cleanup_disappeared_objects(current_frame_ids)

        return self.count_stats.copy()

    def _should_count_object(self, track_id: int, center_x: int, center_y: int) -> bool:
        """
        Determine if object should be counted based on counting method

        Args:
            track_id: Object tracking ID
            center_x: Object center X coordinate
            center_y: Object center Y coordinate

        Returns:
            True if object should be counted
        """
        positions = self.tracked_objects[track_id]['positions']

        if len(positions) < 2:
            return False

        if self.counting_region:
            # Region-based counting
            return self._check_region_crossing(track_id, center_x, center_y)
        else:
            # Line-based counting
            return self._check_line_crossing(track_id, center_x, center_y)

    def _check_line_crossing(self, track_id: int, center_x: int, center_y: int) -> bool:
        """
        Check if object crosses the counting line

        Args:
            track_id: Object tracking ID
            center_x: Current center X coordinate
            center_y: Current center Y coordinate

        Returns:
            True if line crossing detected
        """
        positions = self.tracked_objects[track_id]['positions']

        if len(positions) < 2:
            return False

        # Get previous and current Y positions
        prev_y = positions[-2][1]
        curr_y = center_y

        # Check if object crossed the counting line
        line_y = self.counting_line_position

        # Crossing from top to bottom or bottom to top
        if (prev_y < line_y and curr_y >= line_y) or (prev_y > line_y and curr_y <= line_y):
            # Determine crossing direction
            if prev_y < line_y and curr_y >= line_y:
                direction = "down"
            else:
                direction = "up"

            self.crossing_objects[track_id] = direction
            return True

        return False

    def _check_region_crossing(self, track_id: int, center_x: int, center_y: int) -> bool:
        """
        Check if object enters the counting region

        Args:
            track_id: Object tracking ID
            center_x: Current center X coordinate
            center_y: Current center Y coordinate

        Returns:
            True if region entry detected
        """
        if not self.counting_region:
            return False

        positions = self.tracked_objects[track_id]['positions']

        if len(positions) < 2:
            return False

        # Check if object entered the region
        prev_point = positions[-2]
        curr_point = (center_x, center_y)

        prev_inside = point_in_polygon(prev_point, self.counting_region)
        curr_inside = point_in_polygon(curr_point, self.counting_region)

        # Count when object enters the region
        return not prev_inside and curr_inside

    def _count_object(self, track_id: int, class_name: str):
        """
        Count the object and update statistics

        Args:
            track_id: Object tracking ID
            class_name: Object class name
        """
        # Mark as counted
        self.tracked_objects[track_id]['counted'] = True
        self.counted_objects.add(track_id)

        # Update statistics
        if class_name not in self.count_stats:
            self.count_stats[class_name] = 0

        self.count_stats[class_name] += 1
        self.total_count += 1

        print(f"✅ Counted: {class_name} (ID: {track_id}) - Total: {self.total_count}")

    def _cleanup_disappeared_objects(self, current_frame_ids: Set[int]):
        """
        Remove tracking data for objects that are no longer detected

        Args:
            current_frame_ids: Set of track IDs in current frame
        """
        disappeared_ids = set(self.tracked_objects.keys()) - current_frame_ids

        for track_id in disappeared_ids:
            del self.tracked_objects[track_id]
            if track_id in self.crossing_objects:
                del self.crossing_objects[track_id]

    def reset_counter(self):
        """Reset all counting statistics and tracking data"""
        self.tracked_objects.clear()
        self.counted_objects.clear()
        self.crossing_objects.clear()
        self.count_stats.clear()
        self.total_count = 0
        print("🔄 Counter reset")

    def get_statistics(self) -> Dict[str, any]:
        """
        Get detailed counting statistics

        Returns:
            Dictionary with detailed statistics
        """
        return {
            'total_count': self.total_count,
            'class_counts': self.count_stats.copy(),
            'active_tracks': len(self.tracked_objects),
            'counted_objects': len(self.counted_objects)
        }

    def export_statistics(self) -> str:
        """
        Export statistics as formatted string

        Returns:
            Formatted statistics string
        """
        stats = self.get_statistics()

        output = "=== COUNTING STATISTICS ===\n"
        output += f"Total Objects Counted: {stats['total_count']}\n"
        output += f"Active Tracks: {stats['active_tracks']}\n"
        output += "\nClass Breakdown:\n"

        for class_name, count in stats['class_counts'].items():
            output += f"  {class_name}: {count}\n"

        return output
