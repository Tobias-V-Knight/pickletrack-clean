import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
import math

class BallTracker:
    """Tracks ball trajectory and identifies bounce points."""
    
    def __init__(self, max_trajectory_length: int = 30, min_trajectory_length: int = 5,
                 bounce_threshold: float = 0.3, velocity_threshold: float = 5.0):
        """
        Initialize ball tracker.
        
        Args:
            max_trajectory_length (int): Maximum number of points to track
            min_trajectory_length (int): Minimum trajectory length for analysis
            bounce_threshold (float): Threshold for bounce detection (velocity change ratio)
            velocity_threshold (float): Minimum velocity for valid trajectory
        """
        self.max_trajectory_length = max_trajectory_length
        self.min_trajectory_length = min_trajectory_length
        self.bounce_threshold = bounce_threshold
        self.velocity_threshold = velocity_threshold
        
        self.trajectory = deque(maxlen=max_trajectory_length)
        self.velocities = deque(maxlen=max_trajectory_length)
        self.bounce_points = []
        self.current_trajectory_id = 0
        
    def add_detection(self, frame_number: int, bbox: List[float], confidence: float) -> bool:
        """
        Add a ball detection to the trajectory.
        
        Args:
            frame_number (int): Current frame number
            bbox (list): Bounding box [x1, y1, x2, y2]
            confidence (float): Detection confidence
            
        Returns:
            bool: True if detection was added successfully
        """
        # Calculate center point
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Add to trajectory
        point = {
            'frame': frame_number,
            'x': center_x,
            'y': center_y,
            'bbox': bbox,
            'confidence': confidence
        }
        
        self.trajectory.append(point)
        
        # Calculate velocity if we have at least 2 points
        if len(self.trajectory) >= 2:
            velocity = self._calculate_velocity()
            self.velocities.append(velocity)
            
            # Check for bounce
            if self._detect_bounce():
                self._record_bounce_point()
        
        return True
    
    def _calculate_velocity(self) -> float:
        """Calculate velocity between last two points."""
        if len(self.trajectory) < 2:
            return 0.0
        
        p1 = self.trajectory[-2]
        p2 = self.trajectory[-1]
        
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        
        return math.sqrt(dx*dx + dy*dy)
    
    def _detect_bounce(self) -> bool:
        """
        Detect if a bounce occurred based on velocity changes.
        
        Returns:
            bool: True if bounce detected
        """
        if len(self.velocities) < 3:
            return False
        
        # Look for significant velocity change (bounce)
        current_vel = self.velocities[-1]
        prev_vel = self.velocities[-2]
        
        if prev_vel > self.velocity_threshold and current_vel > self.velocity_threshold:
            velocity_ratio = abs(current_vel - prev_vel) / max(prev_vel, 1e-6)
            return velocity_ratio > self.bounce_threshold
        
        return False
    
    def _record_bounce_point(self):
        """Record the current point as a bounce point."""
        if len(self.trajectory) > 0:
            bounce_point = self.trajectory[-1].copy()
            bounce_point['trajectory_id'] = self.current_trajectory_id
            self.bounce_points.append(bounce_point)
    
    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get current trajectory."""
        return list(self.trajectory)
    
    def get_bounce_points(self) -> List[Dict[str, Any]]:
        """Get all recorded bounce points."""
        return self.bounce_points.copy()
    
    def clear_trajectory(self):
        """Clear current trajectory and start new one."""
        self.trajectory.clear()
        self.velocities.clear()
        self.current_trajectory_id += 1
    
    def is_trajectory_valid(self) -> bool:
        """Check if current trajectory is valid for analysis."""
        return len(self.trajectory) >= self.min_trajectory_length
    
    def get_trajectory_direction(self) -> Optional[str]:
        """
        Determine the general direction of the trajectory.
        
        Returns:
            str: 'left', 'right', 'up', 'down', or None if insufficient data
        """
        if len(self.trajectory) < 2:
            return None
        
        start_point = self.trajectory[0]
        end_point = self.trajectory[-1]
        
        dx = end_point['x'] - start_point['x']
        dy = end_point['y'] - start_point['y']
        
        # Determine primary direction
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'
    
    def get_trajectory_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Get bounding box of trajectory.
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y) or None if no trajectory
        """
        if not self.trajectory:
            return None
        
        x_coords = [p['x'] for p in self.trajectory]
        y_coords = [p['y'] for p in self.trajectory]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def get_trajectory_length(self) -> float:
        """
        Calculate total length of trajectory.
        
        Returns:
            float: Total trajectory length in pixels
        """
        if len(self.trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(self.trajectory)):
            p1 = self.trajectory[i-1]
            p2 = self.trajectory[i]
            
            dx = p2['x'] - p1['x']
            dy = p2['y'] - p1['y']
            total_length += math.sqrt(dx*dx + dy*dy)
        
        return total_length

class MultiBallTracker:
    """Tracks multiple balls simultaneously."""
    
    def __init__(self, max_distance: float = 50.0, max_frames_lost: int = 5):
        """
        Initialize multi-ball tracker.
        
        Args:
            max_distance (float): Maximum distance to associate detections with tracks
            max_frames_lost (int): Maximum frames a track can be lost before removal
        """
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
        self.tracks = {}  # track_id -> BallTracker
        self.next_track_id = 0
        self.frame_count = 0
    
    def update(self, frame_number: int, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracks with new detections.
        
        Args:
            frame_number (int): Current frame number
            detections (list): List of ball detections
            
        Returns:
            List of tracked balls with track IDs
        """
        self.frame_count = frame_number
        
        # Update existing tracks
        updated_tracks = []
        for track_id, tracker in self.tracks.items():
            best_detection = self._find_best_detection(tracker, detections)
            if best_detection:
                tracker.add_detection(frame_number, best_detection['bbox'], best_detection['confidence'])
                updated_tracks.append({
                    'track_id': track_id,
                    'bbox': best_detection['bbox'],
                    'confidence': best_detection['confidence'],
                    'trajectory': tracker.get_trajectory()
                })
                # Remove matched detection
                detections.remove(best_detection)
            else:
                # Track lost - increment lost frames
                tracker.lost_frames = getattr(tracker, 'lost_frames', 0) + 1
        
        # Create new tracks for unmatched detections
        for detection in detections:
            new_tracker = BallTracker()
            new_tracker.add_detection(frame_number, detection['bbox'], detection['confidence'])
            new_tracker.lost_frames = 0
            
            self.tracks[self.next_track_id] = new_tracker
            updated_tracks.append({
                'track_id': self.next_track_id,
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'trajectory': new_tracker.get_trajectory()
            })
            self.next_track_id += 1
        
        # Remove old tracks
        self._remove_old_tracks()
        
        return updated_tracks
    
    def _find_best_detection(self, tracker: BallTracker, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the best matching detection for a track."""
        if not detections or not tracker.trajectory:
            return None
        
        best_detection = None
        min_distance = float('inf')
        
        current_pos = tracker.trajectory[-1]
        
        for detection in detections:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            distance = math.sqrt((center_x - current_pos['x'])**2 + (center_y - current_pos['y'])**2)
            
            if distance < min_distance and distance < self.max_distance:
                min_distance = distance
                best_detection = detection
        
        return best_detection
    
    def _remove_old_tracks(self):
        """Remove tracks that have been lost for too many frames."""
        tracks_to_remove = []
        for track_id, tracker in self.tracks.items():
            if getattr(tracker, 'lost_frames', 0) > self.max_frames_lost:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_all_bounce_points(self) -> List[Dict[str, Any]]:
        """Get bounce points from all tracks."""
        all_bounces = []
        for track_id, tracker in self.tracks.items():
            bounces = tracker.get_bounce_points()
            for bounce in bounces:
                bounce['track_id'] = track_id
                all_bounces.append(bounce)
        return all_bounces
    
    def get_active_tracks(self) -> Dict[int, BallTracker]:
        """Get currently active tracks."""
        return self.tracks.copy() 