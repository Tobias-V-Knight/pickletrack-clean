import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json

class VideoTrimmer:
    """
    Video trimming utility for PickleTrack sessions.
    Automatically detects active segments and provides manual trimming options.
    """
    
    def __init__(self, motion_threshold: float = 200, 
                 min_segment_duration: int = 5,
                 max_gap_duration: int = 10):
        """
        Initialize video trimmer.
        
        Args:
            motion_threshold (float): Minimum motion to consider as activity
            min_segment_duration (int): Minimum duration for valid segment (seconds)
            max_gap_duration (int): Maximum gap to merge segments (seconds)
        """
        self.motion_threshold = motion_threshold
        self.min_segment_duration = min_segment_duration
        self.max_gap_duration = max_gap_duration
        
    def detect_activity_segments(self, video_path: str, 
                                sample_interval: int = 30) -> List[Tuple[float, float]]:
        """
        Automatically detect active segments in video using motion detection.
        
        Args:
            video_path (str): Path to input video
            sample_interval (int): Analyze every N frames for speed
            
        Returns:
            List of (start_time, end_time) tuples for active segments
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Analyzing video for activity... ({total_frames} frames at {fps:.1f} FPS)")
        
        # Motion detection setup
        motion_scores = []
        timestamps = []
        prev_frame = None
        
        frame_count = 0
        while frame_count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            timestamp = frame_count / fps
            timestamps.append(timestamp)
            
            # Convert to grayscale and blur
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_frame is not None:
                # Calculate frame difference
                frame_diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.sum(frame_diff)
                motion_scores.append(motion_score)
            else:
                motion_scores.append(0)
            
            prev_frame = gray
            frame_count += sample_interval
        
        cap.release()
        
        # Find active segments based on motion
        active_segments = self._find_active_segments(motion_scores, timestamps)
        
        print(f"Found {len(active_segments)} active segments:")
        for i, (start, end) in enumerate(active_segments):
            duration = end - start
            print(f"  Segment {i+1}: {start:.1f}s - {end:.1f}s ({duration:.1f}s)")
        
        return active_segments
    
    def _find_active_segments(self, motion_scores: List[float], 
                             timestamps: List[float]) -> List[Tuple[float, float]]:
        """Find continuous segments of activity from motion scores."""
        
        if not motion_scores:
            return []
        
        # Determine activity threshold (adaptive based on video content)
        motion_array = np.array(motion_scores)
        activity_threshold = max(
            self.motion_threshold,
            np.percentile(motion_array, 60)  # Use 60th percentile as baseline (more sensitive)
        )
        
        # Find active periods
        active_mask = motion_array > activity_threshold
        segments = []
        
        in_segment = False
        segment_start = 0
        
        for i, is_active in enumerate(active_mask):
            timestamp = timestamps[i]
            
            if is_active and not in_segment:
                # Start new segment
                segment_start = timestamp
                in_segment = True
            elif not is_active and in_segment:
                # End current segment
                segment_end = timestamp
                duration = segment_end - segment_start
                
                if duration >= self.min_segment_duration:
                    segments.append((segment_start, segment_end))
                
                in_segment = False
        
        # Handle case where video ends during active segment
        if in_segment and timestamps:
            segment_end = timestamps[-1]
            duration = segment_end - segment_start
            if duration >= self.min_segment_duration:
                segments.append((segment_start, segment_end))
        
        # Merge nearby segments
        segments = self._merge_close_segments(segments)
        
        return segments
    
    def _merge_close_segments(self, segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge segments that are close together."""
        if len(segments) <= 1:
            return segments
        
        merged = []
        current_start, current_end = segments[0]
        
        for start, end in segments[1:]:
            gap = start - current_end
            
            if gap <= self.max_gap_duration:
                # Merge with current segment
                current_end = end
            else:
                # Save current segment and start new one
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Add final segment
        merged.append((current_start, current_end))
        
        return merged
    
    def trim_video(self, input_path: str, output_path: str, 
                   start_time: float, end_time: float) -> bool:
        """
        Trim video to specified time range.
        
        Args:
            input_path (str): Input video path
            output_path (str): Output video path
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            
        Returns:
            bool: Success status
        """
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate frame range
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Trim video
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            
            print(f"Trimming video: {start_time:.1f}s - {end_time:.1f}s")
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                current_frame += 1
                
                # Progress update
                if current_frame % 100 == 0:
                    progress = (current_frame - start_frame) / (end_frame - start_frame) * 100
                    print(f"  Progress: {progress:.1f}%")
            
            cap.release()
            out.release()
            
            print(f"✅ Video trimmed successfully: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error trimming video: {e}")
            return False
    
    def create_activity_summary(self, video_path: str, segments: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Create summary of video activity analysis."""
        
        cap = cv2.VideoCapture(video_path)
        total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        active_duration = sum(end - start for start, end in segments)
        inactive_duration = total_duration - active_duration
        
        summary = {
            "video_path": video_path,
            "total_duration": total_duration,
            "active_duration": active_duration,
            "inactive_duration": inactive_duration,
            "activity_percentage": (active_duration / total_duration) * 100 if total_duration > 0 else 0,
            "segments": [
                {
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start
                }
                for start, end in segments
            ],
            "recommendations": self._generate_trim_recommendations(segments, total_duration)
        }
        
        return summary
    
    def _generate_trim_recommendations(self, segments: List[Tuple[float, float]], 
                                     total_duration: float) -> List[str]:
        """Generate recommendations for video trimming."""
        recommendations = []
        
        if not segments:
            recommendations.append("No significant activity detected. Check motion detection settings.")
            return recommendations
        
        active_duration = sum(end - start for start, end in segments)
        activity_percentage = (active_duration / total_duration) * 100
        
        if activity_percentage < 30:
            recommendations.append(f"Low activity ({activity_percentage:.1f}%). Consider trimming to active segments only.")
        
        if len(segments) > 5:
            recommendations.append(f"Multiple segments detected ({len(segments)}). Consider merging nearby segments.")
        
        longest_segment = max(segments, key=lambda x: x[1] - x[0])
        longest_duration = longest_segment[1] - longest_segment[0]
        
        if longest_duration > 600:  # 10 minutes
            recommendations.append(f"Longest segment is {longest_duration/60:.1f} minutes. Consider splitting for faster processing.")
        
        total_recommended = sum(min(end - start, 300) for start, end in segments)  # Cap at 5 min per segment
        if total_recommended < active_duration:
            recommendations.append(f"Recommend processing {total_recommended/60:.1f} minutes of most active content.")
        
        return recommendations
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get basic video information."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Cannot open video"}
        
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
            "file_size": Path(video_path).stat().st_size if Path(video_path).exists() else 0
        }
        
        cap.release()
        return info

def demo_video_trimming():
    """Demo function to show video trimming capabilities."""
    
    # Example usage
    video_path = "data/videos/session1.mp4.mov"  # Update with actual path
    
    if not Path(video_path).exists():
        print(f"Demo video not found: {video_path}")
        return
    
    trimmer = VideoTrimmer()
    
    # Get video info
    info = trimmer.get_video_info(video_path)
    print("Video Info:")
    print(f"  Duration: {info['duration']:.1f} seconds ({info['duration']/60:.1f} minutes)")
    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.1f}")
    
    # Detect activity
    segments = trimmer.detect_activity_segments(video_path)
    
    # Create summary
    summary = trimmer.create_activity_summary(video_path, segments)
    
    print(f"\nActivity Summary:")
    print(f"  Active time: {summary['active_duration']:.1f}s ({summary['activity_percentage']:.1f}%)")
    print(f"  Inactive time: {summary['inactive_duration']:.1f}s")
    
    print(f"\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  • {rec}")

if __name__ == "__main__":
    demo_video_trimming()