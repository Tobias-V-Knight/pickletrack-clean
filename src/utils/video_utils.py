import cv2
import numpy as np
from typing import Tuple, Optional, Generator
from pathlib import Path
import json
import csv
from datetime import datetime

class VideoProcessor:
    """Utility class for video processing operations."""
    
    def __init__(self, video_path: str):
        """
        Initialize video processor.
        
        Args:
            video_path (str): Path to the video file
        """
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video."""
        return self.cap.read()
    
    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Get frame at specific timestamp.
        
        Args:
            timestamp (float): Time in seconds
            
        Returns:
            Frame at timestamp or None if not found
        """
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def frame_generator(self, start_frame: int = 0, end_frame: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generate frames from video.
        
        Args:
            start_frame (int): Starting frame number
            end_frame (int, optional): Ending frame number
            
        Yields:
            Tuple of (frame_number, frame)
        """
        if end_frame is None:
            end_frame = self.frame_count
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_num in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_num, frame
    
    def timestamp_to_frame(self, timestamp: float) -> int:
        """Convert timestamp to frame number."""
        return int(timestamp * self.fps)
    
    def frame_to_timestamp(self, frame_number: int) -> float:
        """Convert frame number to timestamp."""
        return frame_number / self.fps
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS format."""
        return str(datetime.utcfromtimestamp(seconds).strftime('%H:%M:%S'))

class VideoWriter:
    """Utility class for writing video files."""
    
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int], 
                 codec: str = 'mp4v'):
        """
        Initialize video writer.
        
        Args:
            output_path (str): Output video path
            fps (float): Frames per second
            frame_size (tuple): (width, height)
            codec (str): Video codec
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, frame_size)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def write_frame(self, frame: np.ndarray):
        """Write a frame to the video."""
        self.writer.write(frame)
    
    def release(self):
        """Release video writer resources."""
        if self.writer:
            self.writer.release()

def save_results_csv(results: list, output_path: str):
    """
    Save shot results to CSV file.
    
    Args:
        results (list): List of shot result dictionaries
        output_path (str): Output CSV file path
    """
    if not results:
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = results[0].keys()
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def save_summary_json(summary: dict, output_path: str):
    """
    Save performance summary to JSON file.
    
    Args:
        summary (dict): Performance summary dictionary
        output_path (str): Output JSON file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as jsonfile:
        json.dump(summary, jsonfile, indent=2)

def draw_detection_box(image: np.ndarray, bbox: list, label: str = "", 
                      color: Tuple[int, int, int] = (0, 255, 0), 
                      thickness: int = 2) -> np.ndarray:
    """
    Draw detection bounding box on image.
    
    Args:
        image (np.ndarray): Input image
        bbox (list): Bounding box [x1, y1, x2, y2]
        label (str): Label text
        color (tuple): BGR color
        thickness (int): Line thickness
        
    Returns:
        Image with drawn bounding box
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        # Add label background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # Add label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return image 