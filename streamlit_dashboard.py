#!/usr/bin/env python3
"""
PickleTrack Streamlit Analytics Dashboard
Interactive web dashboard for shot analysis with court mapping
"""

import streamlit as st
import cv2
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

from detection.yolo_cone_detector import YOLOConeDetector
from tracking.ball_tracker import BallTracker
from analysis.shot_analyzer import ShotAnalyzer
from mapping.court_mapper import CourtMapper
from utils.video_trimmer import VideoTrimmer

class PickleTrackStreamlitDashboard:
    def __init__(self):
        """Initialize the Streamlit dashboard."""
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'shot_data' not in st.session_state:
            st.session_state.shot_data = []
        if 'stats' not in st.session_state:
            st.session_state.stats = {}
        if 'court_corners' not in st.session_state:
            st.session_state.court_corners = None
        if 'zone_performance' not in st.session_state:
            st.session_state.zone_performance = {}
        if 'zones_defined' not in st.session_state:
            st.session_state.zones_defined = {}
        if 'video_info' not in st.session_state:
            st.session_state.video_info = {}
        if 'activity_segments' not in st.session_state:
            st.session_state.activity_segments = []
    
    def load_demo_data(self):
        """Load demo data for immediate showcase."""
        try:
            demo_path = Path("data/outputs/demo_session_analysis.json")
            if demo_path.exists():
                with open(demo_path, 'r') as f:
                    demo_data = json.load(f)
                
                # Load demo data into session state
                st.session_state.shot_data = demo_data['shots']
                st.session_state.stats = demo_data['statistics']
                st.session_state.zone_performance = demo_data['zone_performance']
                st.session_state.zones_defined = demo_data['zones_defined']
                st.session_state.analysis_complete = True
                
                st.success("✅ Demo data loaded! Explore the zones and shot accuracy below.")
            else:
                st.error("Demo data file not found")
        except Exception as e:
            st.error(f"Error loading demo data: {e}")
    
    def analyze_video_activity(self, video_path: str):
        """Analyze video for activity segments using motion detection."""
        try:
            trimmer = VideoTrimmer()
            
            # Get video info
            video_info = trimmer.get_video_info(video_path)
            st.session_state.video_info = video_info
            
            # Detect activity segments
            segments = trimmer.detect_activity_segments(video_path, sample_interval=60)  # Sample every 60 frames for speed
            st.session_state.activity_segments = segments
            
            # Create activity summary
            summary = trimmer.create_activity_summary(video_path, segments)
            
            st.success(f"Activity analysis complete!")
            st.info(f"Video: {video_info['duration']/60:.1f} minutes total")
            st.info(f"Active segments: {len(segments)} ({summary['activity_percentage']:.1f}% active)")
            
            # Show recommendations
            if summary['recommendations']:
                st.write("**Recommendations:**")
                for rec in summary['recommendations']:
                    st.write(f"• {rec}")
                    
        except Exception as e:
            st.error(f"Error analyzing video activity: {e}")
    
    def setup_sidebar(self):
        """Set up the sidebar with controls."""
        st.sidebar.title("🏓 PickleTrack Controls")
        
        # Demo mode option
        demo_mode = st.sidebar.checkbox("🎯 Try Demo Mode", value=True, help="Use pre-loaded sample data to explore features instantly")
        
        if demo_mode:
            st.sidebar.success("Demo mode active - showing sample analysis")
            if st.sidebar.button("Load Demo Data", type="primary"):
                self.load_demo_data()
            return {'demo_mode': True}
        
        # Video selection
        video_files = list(Path("data/videos").glob("*.mp4*"))
        if video_files:
            video_names = [f.name for f in video_files]
            selected_video = st.sidebar.selectbox("Select Video", video_names)
            video_path = Path("data/videos") / selected_video
        else:
            st.sidebar.error("No video files found in data/videos/")
            return None
        
        # Video Trimming Section
        st.sidebar.subheader("📹 Video Trimming")
        enable_trimming = st.sidebar.checkbox("Enable Smart Trimming", value=True, 
                                            help="Automatically detect and trim to active segments")
        
        if enable_trimming:
            if st.sidebar.button("🔍 Analyze Video Activity"):
                with st.spinner("Analyzing video for activity..."):
                    self.analyze_video_activity(str(video_path))
            
            # Show activity analysis results
            if st.session_state.activity_segments:
                segments = st.session_state.activity_segments
                total_active = sum(end - start for start, end in segments)
                
                st.sidebar.success(f"Found {len(segments)} active segments")
                st.sidebar.info(f"Active time: {total_active/60:.1f} minutes")
                
                # Manual trim controls
                if len(segments) > 0:
                    st.sidebar.write("**Manual Trim (Optional):**")
                    video_info = st.session_state.get('video_info', {})
                    max_duration = video_info.get('duration', 3600)
                    
                    trim_start = st.sidebar.slider("Start (seconds)", 0.0, max_duration, 0.0)
                    trim_end = st.sidebar.slider("End (seconds)", trim_start, max_duration, min(300.0, max_duration))
                    
                    if st.sidebar.button("✂️ Apply Trim"):
                        st.sidebar.info(f"Will process {trim_end - trim_start:.1f} seconds")
        
        # Analysis settings
        st.sidebar.subheader("Analysis Settings")
        max_frames = st.sidebar.slider("Max Frames to Process", 50, 1000, 300)
        confidence_threshold = st.sidebar.slider("Ball Detection Confidence", 0.1, 0.9, 0.5)
        
        # Run analysis button
        if st.sidebar.button("Run Analysis", type="primary"):
            with st.spinner("Running shot analysis..."):
                self.run_analysis(str(video_path), max_frames, confidence_threshold)
        
        return {
            'video_path': str(video_path),
            'max_frames': max_frames,
            'confidence_threshold': confidence_threshold
        }
    
    def run_analysis(self, video_path: str, max_frames: int, confidence_threshold: float):
        """Run the complete shot analysis pipeline."""
        try:
            # Initialize components
            ball_tracker = BallTracker()
            
            # Process video
            shot_data, stats = self.process_video_streamlined(
                video_path, max_frames, confidence_threshold
            )
            
            # Store results in session state
            st.session_state.shot_data = shot_data
            st.session_state.stats = stats
            st.session_state.analysis_complete = True
            
            st.success(f"Analysis complete! Found {len(shot_data)} shots.")
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
    
    def process_video_streamlined(self, video_path: str, max_frames: int, confidence_threshold: float):
        """Streamlined video processing for dashboard."""
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = min(total_frames, max_frames)
        
        # Set up court mapping (simplified)
        ret, first_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        if not ret:
            raise ValueError("Could not read first frame")
        
        h, w = first_frame.shape[:2]
        court_corners = [
            (int(w * 0.15), int(h * 0.3)),   # Top-left
            (int(w * 0.85), int(h * 0.3)),   # Top-right  
            (int(w * 0.85), int(h * 0.85)),  # Bottom-right
            (int(w * 0.15), int(h * 0.85))   # Bottom-left
        ]
        
        court_mapper = CourtMapper()
        court_mapper.set_court_corners(court_corners)
        st.session_state.court_corners = court_corners
        
        # Initialize ball tracker
        ball_tracker = BallTracker()
        
        # Process frames
        shot_data = []
        current_trajectory = []
        frame_num = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while frame_num < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            progress = frame_num / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_num}/{total_frames}")
            
            # Detect balls (simplified detection)
            ball_detections = self.detect_balls_simple(frame, confidence_threshold)
            
            # Process detections
            for detection in ball_detections:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Convert to court coordinates
                court_coords = court_mapper.image_to_court_coords((center_x, center_y))
                
                if court_coords:
                    trajectory_point = {
                        'frame': frame_num,
                        'timestamp': frame_num / fps,
                        'image_coords': (center_x, center_y),
                        'court_coords': court_coords,
                        'confidence': detection['confidence']
                    }
                    current_trajectory.append(trajectory_point)
            
            # Detect shot boundaries (every 2 seconds or end of video)
            if len(current_trajectory) > 0 and (
                frame_num % 60 == 0 or frame_num == total_frames - 1
            ):
                if len(current_trajectory) >= 3:
                    shot_analysis = self.analyze_trajectory_simple(current_trajectory)
                    if shot_analysis:
                        shot_data.append(shot_analysis)
                current_trajectory = []
            
            frame_num += 1
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        # Calculate statistics
        stats = self.calculate_stats_simple(shot_data)
        
        return shot_data, stats
    
    def detect_balls_simple(self, frame, confidence_threshold):
        """Simplified ball detection for demo purposes."""
        # This is a placeholder - in reality, you'd use the YOLO ball detector
        # For demo, we'll create some mock detections
        
        # Simulate ball detections (replace with actual YOLO detection)
        detections = []
        
        # Simple color-based detection for yellow/green objects (balls)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Yellow range for tennis balls
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                detection = {
                    'bbox': [x, y, x+w, y+h],
                    'confidence': min(0.9, area / 1000),  # Mock confidence
                    'class_name': 'sports ball'
                }
                if detection['confidence'] >= confidence_threshold:
                    detections.append(detection)
        
        return detections
    
    def analyze_trajectory_simple(self, trajectory):
        """Simplified trajectory analysis."""
        if len(trajectory) < 3:
            return None
        
        court_coords = [point['court_coords'] for point in trajectory]
        timestamps = [point['timestamp'] for point in trajectory]
        
        start_pos = court_coords[0]
        end_pos = court_coords[-1]
        
        # Calculate metrics
        distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        duration = timestamps[-1] - timestamps[0]
        avg_speed = distance / duration if duration > 0 else 0
        
        # Simple bounce detection
        bounces = max(0, len(trajectory) // 10)  # Rough estimate
        
        # Determine outcome
        court_width, court_height = 44, 20
        in_bounds = all(
            0 <= pos[0] <= court_width and 0 <= pos[1] <= court_height 
            for pos in court_coords
        )
        
        if not in_bounds:
            outcome = 'out_of_bounds'
        elif bounces > 0:
            outcome = 'successful_hit'
        else:
            outcome = 'missed_shot'
        
        return {
            'start_position': start_pos,
            'end_position': end_pos,
            'distance': distance,
            'duration': duration,
            'avg_speed': avg_speed,
            'bounces': bounces,
            'in_bounds': in_bounds,
            'outcome': outcome,
            'trajectory': trajectory
        }
    
    def calculate_stats_simple(self, shot_data):
        """Calculate basic statistics."""
        stats = {
            'total_shots': len(shot_data),
            'shots_in_bounds': 0,
            'shots_out_of_bounds': 0,
            'missed_shots': 0,
            'successful_hits': 0,
            'total_bounces': 0,
            'avg_ball_speed': 0,
            'shot_accuracy': 0
        }
        
        if not shot_data:
            return stats
        
        for shot in shot_data:
            if shot['outcome'] == 'out_of_bounds':
                stats['shots_out_of_bounds'] += 1
            elif shot['outcome'] == 'successful_hit':
                stats['successful_hits'] += 1
                stats['shots_in_bounds'] += 1
            else:
                stats['missed_shots'] += 1
                stats['shots_in_bounds'] += 1
            
            stats['total_bounces'] += shot['bounces']
        
        # Calculate averages
        total_speed = sum(shot['avg_speed'] for shot in shot_data)
        stats['avg_ball_speed'] = total_speed / len(shot_data)
        
        if stats['total_shots'] > 0:
            stats['shot_accuracy'] = (stats['successful_hits'] / stats['total_shots']) * 100
        
        return stats
    
    def display_video_trimming_interface(self):
        """Display the video trimming and preprocessing interface."""
        
        # Demo notice
        if not st.session_state.get('analysis_complete', False):
            st.info("📋 This feature works with uploaded videos. Try demo mode to see other features!")
        
        # Video upload section
        st.subheader("📤 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a pickleball session video", 
            type=['mp4', 'mov', 'avi', 'mkv'],
            help="Upload your full practice session - we'll automatically detect the active segments"
        )
        
        if uploaded_file is not None:
            st.success(f"Video uploaded: {uploaded_file.name}")
            
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Analyze button
            if st.button("🔍 Analyze Video Activity", type="primary"):
                with st.spinner("Analyzing video for ball-hitting activity..."):
                    self.analyze_video_activity(temp_path)
        
        # Show video info if available
        video_info = st.session_state.get('video_info', {})
        if video_info:
            st.subheader("📊 Video Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Duration", f"{video_info['duration']/60:.1f} min")
            with col2:
                st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
            with col3:
                st.metric("Frame Rate", f"{video_info['fps']:.1f} FPS")
        
        # Show activity analysis results
        activity_segments = st.session_state.get('activity_segments', [])
        if activity_segments:
            st.subheader("🎯 Activity Detection Results")
            
            total_duration = video_info.get('duration', 0)
            active_duration = sum(end - start for start, end in activity_segments)
            activity_percentage = (active_duration / total_duration) * 100 if total_duration > 0 else 0
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Active Segments", len(activity_segments))
            with col2:
                st.metric("Active Time", f"{active_duration/60:.1f} min")
            with col3:
                st.metric("Activity %", f"{activity_percentage:.1f}%")
            
            # Activity timeline visualization
            if len(activity_segments) > 0:
                st.subheader("📈 Activity Timeline")
                
                # Create timeline data for visualization
                timeline_data = []
                for i, (start, end) in enumerate(activity_segments):
                    timeline_data.append({
                        'Segment': f'Segment {i+1}',
                        'Start': start,
                        'End': end,
                        'Duration': end - start
                    })
                
                # Create Gantt-like chart
                import plotly.figure_factory as ff
                
                fig = ff.create_gantt(
                    timeline_data,
                    colors=['#2ecc71'],
                    index_col='Segment',
                    title='Active Segments Timeline',
                    show_colorbar=True,
                    bar_width=0.5,
                    showgrid_x=True,
                    showgrid_y=True
                )
                
                fig.update_layout(
                    xaxis_title="Time (seconds)",
                    height=200 + len(activity_segments) * 30
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Trimming recommendations
            st.subheader("💡 Recommendations")
            
            if activity_percentage < 30:
                st.warning(f"⚠️ Low activity detected ({activity_percentage:.1f}%). Consider trimming to active segments only.")
            elif activity_percentage > 80:
                st.success(f"✅ High activity video ({activity_percentage:.1f}%). Minimal trimming needed.")
            else:
                st.info(f"ℹ️ Moderate activity ({activity_percentage:.1f}%). Some trimming recommended.")
            
            # Manual trimming controls
            st.subheader("✂️ Trim Video")
            st.markdown("*Choose which segments to process for faster analysis*")
            
            # Option 1: Process all active segments
            if st.button("🎯 Process All Active Segments"):
                with st.spinner("Processing all active segments..."):
                    try:
                        trimmer = VideoTrimmer()
                        
                        # Get video info
                        video_info = st.session_state.video_info
                        segments = st.session_state.activity_segments
                        
                        # Process each segment
                        all_results = []
                        progress_bar = st.progress(0)
                        
                        for i, segment in enumerate(segments):
                            start_frame, end_frame = segment
                            progress_bar.progress((i + 1) / len(segments))
                            
                            # Create segment summary
                            start_time = start_frame / video_info['fps']
                            end_time = end_frame / video_info['fps']
                            
                            segment_result = {
                                'segment': i + 1,
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': end_time - start_time,
                                'frames': end_frame - start_frame
                            }
                            all_results.append(segment_result)
                        
                        # Store results
                        st.session_state.processed_segments = all_results
                        
                        st.success(f"✅ Processed {len(segments)} active segments!")
                        st.info(f"Total active time: {sum(r['duration'] for r in all_results):.1f} seconds")
                        
                        # Show segment details
                        for result in all_results:
                            st.write(f"**Segment {result['segment']}**: {result['start_time']:.1f}s - {result['end_time']:.1f}s ({result['duration']:.1f}s)")
                        
                    except Exception as e:
                        st.error(f"Error processing segments: {e}")
            
            # Option 2: Custom time range
            with st.expander("🔧 Custom Time Range"):
                col1, col2 = st.columns(2)
                with col1:
                    start_time = st.number_input("Start (seconds)", 0.0, total_duration, 0.0)
                with col2:
                    end_time = st.number_input("End (seconds)", start_time, total_duration, min(300.0, total_duration))
                
                trim_duration = end_time - start_time
                st.info(f"Selected duration: {trim_duration:.1f} seconds ({trim_duration/60:.1f} minutes)")
                
                if st.button("✂️ Trim & Process"):
                    with st.spinner(f"Trimming video from {start_time:.1f}s to {end_time:.1f}s..."):
                        try:
                            trimmer = VideoTrimmer()
                            video_info = st.session_state.video_info
                            
                            # Convert time to frames
                            start_frame = int(start_time * video_info['fps'])
                            end_frame = int(end_time * video_info['fps'])
                            
                            # Create custom segment
                            custom_segment = {
                                'segment': 'custom',
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': trim_duration,
                                'frames': end_frame - start_frame,
                                'start_frame': start_frame,
                                'end_frame': end_frame
                            }
                            
                            # Store custom segment
                            st.session_state.custom_segment = custom_segment
                            
                            st.success(f"✅ Custom segment defined!")
                            st.info(f"**Custom Range**: {start_time:.1f}s - {end_time:.1f}s ({trim_duration:.1f}s, {end_frame - start_frame} frames)")
                            
                            # Show trimming benefits
                            original_duration = video_info['duration']
                            time_saved = original_duration - trim_duration
                            savings_percent = (time_saved / original_duration) * 100
                            
                            st.write(f"**Time Savings**: {time_saved:.1f}s ({savings_percent:.1f}% reduction)")
                            st.write("This segment is ready for analysis when you upload the video!")
                            
                        except Exception as e:
                            st.error(f"Error creating custom trim: {e}")
        
        # Benefits explanation
        st.subheader("🚀 Why Use Video Trimming?")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **⚡ Faster Processing**
            - Process only relevant segments
            - Reduce analysis time by 60-80%
            - Focus on actual ball-hitting activity
            """)
        
        with col2:
            st.markdown("""
            **🎯 Better Accuracy**
            - Eliminate warm-up/break periods
            - Reduce false detections
            - Improve shot tracking quality
            """)
    
    def display_zone_accuracy_highlights(self):
        """Display zone accuracy metrics - the core feature."""
        if not st.session_state.analysis_complete:
            st.info("🎯 Load demo data to see zone accuracy tracking!")
            return
        
        zone_performance = st.session_state.zone_performance
        if not zone_performance:
            st.warning("No zone data available")
            return
        
        st.markdown("""
        <div class="highlight-box">
            <h2 style="margin-top: 0; color: #1f77b4;">🎯 Target Zone Accuracy - Core Feature</h2>
            <p style="margin-bottom: 0; font-style: italic;">This is what makes PickleTrack unique: precise shot accuracy tracking in cone-defined zones</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create zone accuracy metrics
        cols = st.columns(len(zone_performance))
        
        for i, (zone_name, zone_data) in enumerate(zone_performance.items()):
            with cols[i]:
                accuracy = zone_data['accuracy']
                attempts = zone_data['attempts']
                hits = zone_data['hits']
                
                # Color code based on accuracy
                if accuracy >= 60:
                    color = "🟢"
                elif accuracy >= 40:
                    color = "🟡" 
                else:
                    color = "🔴"
                
                st.metric(
                    f"{color} {zone_name.replace('_', ' ').title()}", 
                    f"{accuracy:.1f}%",
                    delta=f"{hits}/{attempts} hits"
                )
        
        # Zone performance chart
        if zone_performance:
            zone_names = [name.replace('target_zone_', '').replace('_', ' ').title() for name in zone_performance.keys()]
            accuracies = [data['accuracy'] for data in zone_performance.values()]
            attempts = [data['attempts'] for data in zone_performance.values()]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=zone_names,
                y=accuracies,
                text=[f"{acc:.1f}%<br>({att} shots)" for acc, att in zip(accuracies, attempts)],
                textposition='auto',
                marker_color=['#2ecc71' if acc >= 60 else '#f39c12' if acc >= 40 else '#e74c3c' for acc in accuracies]
            ))
            
            fig.update_layout(
                title="Zone Accuracy Comparison",
                xaxis_title="Target Zones",
                yaxis_title="Accuracy (%)",
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def display_overview_metrics(self):
        """Display key metrics in the overview section."""
        if not st.session_state.analysis_complete:
            st.info("Run analysis to see metrics")
            return
        
        stats = st.session_state.stats
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Shots", stats['total_shots'])
        
        with col2:
            st.metric("Shot Accuracy", f"{stats['shot_accuracy']:.1f}%")
        
        with col3:
            st.metric("Avg Ball Speed", f"{stats['avg_ball_speed']:.1f} ft/s")
        
        with col4:
            st.metric("Total Bounces", stats['total_bounces'])
    
    def display_shot_outcomes_chart(self):
        """Display shot outcomes using Plotly pie chart."""
        if not st.session_state.analysis_complete:
            return
        
        stats = st.session_state.stats
        
        # Prepare data
        labels = ['Successful Hits', 'Missed Shots', 'Out of Bounds']
        values = [stats['successful_hits'], stats['missed_shots'], stats['shots_out_of_bounds']]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            marker_colors=colors,
            textinfo='label+percent',
            hovertemplate='%{label}: %{value}<br>%{percent}<extra></extra>'
        )])
        
        fig.update_layout(title="Shot Outcomes Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    def display_court_heatmap(self):
        """Display court heatmap of shot landing positions."""
        if not st.session_state.analysis_complete:
            return
        
        shot_data = st.session_state.shot_data
        if not shot_data:
            return
        
        # Prepare data
        end_positions = [shot['end_position'] for shot in shot_data]
        outcomes = [shot['outcome'] for shot in shot_data]
        
        x_coords = [pos[0] for pos in end_positions]
        y_coords = [pos[1] for pos in end_positions]
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': x_coords,
            'y': y_coords,
            'outcome': outcomes
        })
        
        # Create scatter plot with Plotly
        fig = px.scatter(
            df, x='x', y='y', color='outcome',
            color_discrete_map={
                'successful_hit': '#2ecc71',
                'missed_shot': '#f39c12',
                'out_of_bounds': '#e74c3c'
            },
            title="Shot Landing Positions on Court"
        )
        
        # Add court boundary
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=44, y1=20,
            line=dict(color="black", width=2),
            fillcolor="lightgreen",
            opacity=0.2
        )
        
        fig.update_layout(
            xaxis_title="Court Width (feet)",
            yaxis_title="Court Length (feet)",
            xaxis=dict(range=[-2, 46]),
            yaxis=dict(range=[-2, 22]),
            width=600,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_speed_distribution(self):
        """Display ball speed distribution using Seaborn."""
        if not st.session_state.analysis_complete:
            return
        
        shot_data = st.session_state.shot_data
        if not shot_data:
            return
        
        speeds = [shot['avg_speed'] for shot in shot_data]
        
        # Create histogram with Plotly
        fig = px.histogram(
            x=speeds,
            nbins=10,
            title="Ball Speed Distribution",
            labels={'x': 'Speed (ft/s)', 'y': 'Number of Shots'}
        )
        
        fig.update_layout(
            xaxis_title="Speed (ft/s)",
            yaxis_title="Number of Shots",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_shot_timeline(self):
        """Display shot timeline."""
        if not st.session_state.analysis_complete:
            return
        
        shot_data = st.session_state.shot_data
        if not shot_data:
            return
        
        # Prepare timeline data
        timeline_data = []
        for i, shot in enumerate(shot_data):
            if shot['trajectory']:
                timeline_data.append({
                    'shot_number': i + 1,
                    'timestamp': shot['trajectory'][0]['timestamp'],
                    'outcome': shot['outcome'],
                    'speed': shot['avg_speed']
                })
        
        df = pd.DataFrame(timeline_data)
        
        # Create timeline scatter plot
        fig = px.scatter(
            df, x='timestamp', y='shot_number', 
            color='outcome', size='speed',
            color_discrete_map={
                'successful_hit': '#2ecc71',
                'missed_shot': '#f39c12',
                'out_of_bounds': '#e74c3c'
            },
            title="Shot Timeline",
            labels={'timestamp': 'Time (seconds)', 'shot_number': 'Shot Number'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_data_table(self):
        """Display detailed shot data in a table."""
        if not st.session_state.analysis_complete:
            return
        
        shot_data = st.session_state.shot_data
        if not shot_data:
            return
        
        # Prepare table data
        table_data = []
        for i, shot in enumerate(shot_data):
            # Get zone info
            target_zone = shot.get('target_zone', 'No zone')
            zone_result = shot.get('result', shot['outcome'])
            
            table_data.append({
                'Shot #': i + 1,
                'Target Zone': target_zone.replace('target_zone_', '').replace('_', ' ').title() if target_zone != 'No zone' else 'None',
                'Zone Result': '🎯 HIT' if zone_result == 'hit' else '❌ MISS' if zone_result == 'miss' else '🚫 OUT',
                'Outcome': shot['outcome'].replace('_', ' ').title(),
                'Distance (ft)': round(shot['distance'], 2),
                'Speed (ft/s)': round(shot['avg_speed'], 2),
                'Bounces': shot['bounces'],
                'In Bounds': '✓' if shot['in_bounds'] else '✗'
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    
    def run_dashboard(self):
        """Main dashboard function."""
        st.set_page_config(
            page_title="PickleTrack Analytics",
            page_icon="🏓",
            layout="wide"
        )
        
        # Custom CSS for professional styling
        st.markdown("""
        <style>
        .main {
            padding-top: 1rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            border: 1px solid #e1e5e9;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .zone-accuracy-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .highlight-box {
            background-color: #e8f4fd;
            border-left: 4px solid #1f77b4;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.25rem;
        }
        .success-box {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.25rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header with branding
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #1f77b4; font-size: 3rem; margin-bottom: 0;">🏓 PickleTrack</h1>
            <h3 style="color: #666; font-weight: 300; margin-top: 0;">AI-Powered Pickleball Analytics</h3>
            <p style="color: #888; font-size: 1.1rem;">Track shot accuracy in cone-defined target zones</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        config = self.setup_sidebar()
        if not config:
            return
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Zone Accuracy", "📹 Video Trimming", "📊 Overview", "📈 Performance", "📋 Data"])
        
        with tab1:
            st.header("🎯 Zone Accuracy Tracking")
            st.markdown("**The core feature: Track shot accuracy in cone-defined target zones**")
            
            # Prominent zone accuracy display
            self.display_zone_accuracy_highlights()
            
            col1, col2 = st.columns(2)
            with col1:
                self.display_court_heatmap()
            with col2:
                self.display_shot_outcomes_chart()
        
        with tab2:
            st.header("📹 Video Trimming & Preprocessing")
            st.markdown("**Automatically detect active segments in long videos to focus processing on relevant content**")
            
            self.display_video_trimming_interface()
        
        with tab3:
            st.header("Session Overview")
            self.display_overview_metrics()
            
            col1, col2 = st.columns(2)
            with col1:
                self.display_shot_outcomes_chart()
            with col2:
                self.display_court_heatmap()
        
        with tab4:
            st.header("Shot Analysis")
            col1, col2 = st.columns(2)
            with col1:
                self.display_speed_distribution()
            with col2:
                self.display_shot_timeline()
        
        with tab5:
            st.header("Detailed Data")
            if st.session_state.analysis_complete:
                st.subheader("Shot Details")
                self.display_data_table()
                
                # Export options
                st.subheader("Export Data")
                if st.button("Export to CSV"):
                    shot_data = st.session_state.shot_data
                    if shot_data:
                        # Convert to DataFrame and download
                        table_data = []
                        for i, shot in enumerate(shot_data):
                            table_data.append({
                                'shot_id': i + 1,
                                'outcome': shot['outcome'],
                                'distance': shot['distance'],
                                'duration': shot['duration'],
                                'avg_speed': shot['avg_speed'],
                                'bounces': shot['bounces'],
                                'in_bounds': shot['in_bounds']
                            })
                        
                        df = pd.DataFrame(table_data)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="pickletrack_analysis.csv",
                            mime="text/csv"
                        )

def main():
    """Run the Streamlit dashboard."""
    dashboard = PickleTrackStreamlitDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()