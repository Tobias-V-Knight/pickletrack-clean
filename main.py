#!/usr/bin/env python3
"""
PickleTrack Main Processing Pipeline
Processes pickleball practice videos to track shot accuracy using cone-defined zones.
"""

import argparse
import cv2
import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add src to Python path
current_dir = Path(__file__).parent
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

from detection.yolo_cone_detector import YOLOConeDetector
from detection.yolo_detector import YOLODetector
from mapping.court_mapper import CourtMapper
from mapping.zone_constructor import ZoneConstructor
from tracking.ball_tracker import MultiBallTracker
from analysis.shot_analyzer import ShotAnalyzer
from utils.video_utils import VideoProcessor, VideoWriter, draw_detection_box

class PickleTrackPipeline:
    """Main processing pipeline for PickleTrack."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the PickleTrack pipeline.
        
        Args:
            config_path (str): Path to model configuration file
        """
        self.config_path = config_path
        self.load_config()
        
        # Initialize components
        self.cone_detector = YOLOConeDetector(config_path)
        self.ball_detector = YOLODetector(self.config['ball_detection']['model'])
        self.court_mapper = CourtMapper()
        self.zone_constructor = ZoneConstructor(self.court_mapper)
        self.ball_tracker = MultiBallTracker()
        self.shot_analyzer = ShotAnalyzer(self.court_mapper)
        
        # Pipeline state
        self.zones_established = False
        self.frame_count = 0
        self.processed_shots = []
        
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using defaults.")
            self.config = {
                'ball_detection': {'model': 'yolov8n.pt', 'confidence': 0.5},
                'cone_detection': {'model': 'yolov8n.pt', 'confidence': 0.6}
            }
    
    def process_video(self, video_path: str, output_dir: str, 
                     use_mock_zones: bool = False, max_frames: int = None) -> Dict[str, Any]:
        """
        Process a video file to detect shots and analyze performance.
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory for output files
            use_mock_zones (bool): Use mock zones instead of detecting cones
            max_frames (int): Maximum frames to process (None for all)
            
        Returns:
            Processing results summary
        """
        print(f"[PickleTrack] Starting processing...")
        print(f"[Video] {video_path}")
        print(f"[Output] {output_dir}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize video processor
        with VideoProcessor(video_path) as video:
            print(f"[Video Info] {video.width}x{video.height}, {video.fps:.1f} FPS, {video.duration:.1f}s")
            
            # Setup video writer for annotated output
            output_video_path = output_path / f"{Path(video_path).stem}_annotated.mp4"
            video_writer = VideoWriter(
                str(output_video_path), 
                video.fps, 
                (video.width, video.height)
            )
            
            # Phase 1: Establish zones from cones
            if not use_mock_zones:
                self.establish_zones_from_video(video, max_frames=100)  # Check first 100 frames
            else:
                print("[Zones] Using mock zones for testing")
                self.zone_constructor.create_mock_zones()
                self.zones_established = True
            
            # Phase 2: Process full video for ball tracking and shot analysis
            if self.zones_established:
                results = self.process_frames_for_shots(video, video_writer, max_frames)
            else:
                print("[Error] No zones established. Cannot analyze shots.")
                results = {'error': 'No zones detected'}
            
            video_writer.release()
        
        # Phase 3: Generate final results
        if self.zones_established and not results.get('error'):
            self.generate_final_results(output_path, Path(video_path).stem)
        
        print("[Complete] Processing finished!")
        return results
    
    def establish_zones_from_video(self, video: VideoProcessor, max_frames: int = 100):
        """
        Establish target zones by detecting cones in the first part of the video.
        
        Args:
            video: VideoProcessor instance
            max_frames (int): Maximum frames to check for cones
        """
        print("[Phase 1] Detecting cones to establish zones...")
        
        cone_detections_all = []
        frames_processed = 0
        
        for frame_num, frame in video.frame_generator(0, max_frames):
            # Detect cones
            cone_detections = self.cone_detector.detect_cones(frame)
            
            if cone_detections:
                print(f"   Frame {frame_num}: Found {len(cone_detections)} cones")
                cone_detections_all.extend(cone_detections)
                
                # Add to zone constructor
                self.zone_constructor.add_cone_detections(cone_detections, frame_num)
            
            frames_processed += 1
            
            # Stop early if we have enough cone detections
            if len(cone_detections_all) >= 12:  # Assume 3-4 cones, multiple detections each
                break
        
        print(f"[Cone Detection] Complete: {len(cone_detections_all)} total detections from {frames_processed} frames")
        
        # Construct zones from cone detections
        if len(self.zone_constructor.cone_positions) >= 3:
            zones = self.zone_constructor.construct_zones_auto("target_zone")
            if zones:
                print(f"[Zones] Established: {list(zones.keys())}")
                self.zones_established = True
                
                # Print zone info
                for zone_name, zone_data in zones.items():
                    print(f"   {zone_name}: {zone_data['cone_count']} cones, {zone_data['area']:.1f} sq ft")
            else:
                print("[Error] Failed to construct zones from cone detections")
        else:
            print(f"[Error] Insufficient cones detected ({len(self.zone_constructor.cone_positions)} < 3)")
    
    def process_frames_for_shots(self, video: VideoProcessor, video_writer: VideoWriter, 
                               max_frames: int = None) -> Dict[str, Any]:
        """
        Process video frames to detect and track ball shots.
        
        Args:
            video: VideoProcessor instance  
            video_writer: VideoWriter for annotated output
            max_frames (int): Maximum frames to process
            
        Returns:
            Processing results
        """
        print("[Phase 2] Processing frames for ball tracking and shot analysis...")
        
        total_frames = min(video.frame_count, max_frames) if max_frames else video.frame_count
        frames_processed = 0
        shots_detected = 0
        
        for frame_num, frame in video.frame_generator():
            if max_frames and frames_processed >= max_frames:
                break
            
            # Create annotated frame copy
            annotated_frame = frame.copy()
            
            # Detect balls
            ball_detections = self.ball_detector.detect(frame)
            ball_detections = [d for d in ball_detections if 'ball' in d['class_name'].lower()]
            
            # Update ball tracker
            tracked_balls = self.ball_tracker.update(frame_num, ball_detections)
            
            # Analyze completed trajectories for shots
            for track_id, tracker in self.ball_tracker.get_active_tracks().items():
                if tracker.is_trajectory_valid():
                    trajectory = tracker.get_trajectory()
                    shot_result = self.shot_analyzer.analyze_trajectory(trajectory, track_id)
                    
                    if shot_result:
                        # Check if shot lands in target zone
                        end_court = shot_result.get('end_court_coords')
                        if end_court:
                            zone_name = self.zone_constructor.get_zone_at_point(end_court)
                            shot_result['target_zone'] = zone_name
                            shot_result['result'] = 'hit' if zone_name else 'miss'
                        
                        self.shot_analyzer.add_shot(shot_result)
                        shots_detected += 1
                        
                        print(f"   Shot detected: {shot_result['result']} (Track {track_id})")
            
            # Draw annotations on frame
            annotated_frame = self.draw_frame_annotations(annotated_frame, tracked_balls)
            
            # Write annotated frame
            video_writer.write_frame(annotated_frame)
            
            frames_processed += 1
            
            # Progress update
            if frames_processed % 100 == 0:
                progress = (frames_processed / total_frames) * 100
                print(f"   Progress: {progress:.1f}% ({frames_processed}/{total_frames} frames)")
        
        print(f"[Shot Tracking] Complete: {shots_detected} shots detected from {frames_processed} frames")
        
        return {
            'frames_processed': frames_processed,
            'shots_detected': shots_detected,
            'success': True
        }
    
    def draw_frame_annotations(self, frame, tracked_balls: List[Dict[str, Any]]):
        """Draw annotations on frame."""
        # Draw zones
        if self.zones_established:
            frame = self.court_mapper.draw_court_overlay(frame, draw_zones=True)
            
            # Draw zone polygons from constructor
            for zone_name, zone_data in self.zone_constructor.get_zones().items():
                color = tuple(zone_data['color'])
                coords = zone_data['coordinates']
                
                # Transform to image coordinates and draw
                image_coords = []
                for coord in coords:
                    img_coord = self.court_mapper.court_to_image_coords(coord)
                    if img_coord:
                        image_coords.append(img_coord)
                
                if len(image_coords) >= 3:
                    points = np.array(image_coords, dtype=np.int32)
                    cv2.polylines(frame, [points], True, color, 3)
                    
                    # Add zone label
                    center_x = int(sum(p[0] for p in image_coords) / len(image_coords))
                    center_y = int(sum(p[1] for p in image_coords) / len(image_coords))
                    cv2.putText(frame, zone_name, (center_x, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw tracked balls
        for ball in tracked_balls:
            bbox = ball['bbox']
            track_id = ball['track_id']
            label = f"Ball {track_id}"
            frame = draw_detection_box(frame, bbox, label, (0, 255, 0))
            
            # Draw trajectory
            trajectory = ball.get('trajectory', [])
            if len(trajectory) > 1:
                points = [(int(p['x']), int(p['y'])) for p in trajectory]
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (255, 0, 0), 2)
        
        return frame
    
    def generate_final_results(self, output_path: Path, session_name: str):
        """Generate final analysis results."""
        print("[Phase 3] Generating final results...")
        
        # Generate session summary
        session_info = {
            'session_name': session_name,
            'total_frames': self.frame_count,
            'zones_detected': len(self.zone_constructor.get_zones()),
            'processing_date': str(Path().cwd())
        }
        
        summary = self.shot_analyzer.generate_session_summary(session_info)
        
        # Export results
        self.shot_analyzer.export_results(str(output_path), session_name)
        
        # Print summary
        overall_stats = summary.get('overall_stats', {})
        print(f"[Session Summary]")
        print(f"   Total shots: {overall_stats.get('total_shots', 0)}")
        print(f"   Hits: {overall_stats.get('hits', 0)}")
        print(f"   Misses: {overall_stats.get('misses', 0)}")
        print(f"   Out of bounds: {overall_stats.get('out_of_bounds', 0)}")
        print(f"   Accuracy: {overall_stats.get('accuracy', 0):.1%}")
        
        # Print recommendations
        recommendations = self.shot_analyzer.get_recommendations()
        if recommendations:
            print("[Recommendations]")
            for rec in recommendations:
                print(f"   • {rec}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PickleTrack - Pickleball Shot Tracking")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default="data/outputs", help="Output directory")
    parser.add_argument("--config", default="config/model_config.yaml", help="Model config file")
    parser.add_argument("--mock-zones", action="store_true", help="Use mock zones for testing")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    
    args = parser.parse_args()
    
    # Validate input video
    if not Path(args.video).exists():
        print(f"[Error] Video file not found: {args.video}")
        return 1
    
    try:
        # Initialize and run pipeline
        pipeline = PickleTrackPipeline(args.config)
        results = pipeline.process_video(
            args.video, 
            args.output, 
            use_mock_zones=args.mock_zones,
            max_frames=args.max_frames
        )
        
        if results.get('error'):
            print(f"[Error] Processing failed: {results['error']}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"[Error] Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())