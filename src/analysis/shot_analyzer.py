import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

class ShotAnalyzer:
    """Analyzes ball trajectories and classifies shots."""
    
    def __init__(self, court_mapper, min_trajectory_length: int = 10, 
                 bounce_detection_threshold: float = 0.3):
        """
        Initialize shot analyzer.
        
        Args:
            court_mapper: CourtMapper instance for coordinate transformation
            min_trajectory_length (int): Minimum trajectory length for analysis
            bounce_detection_threshold (float): Threshold for bounce detection
        """
        self.court_mapper = court_mapper
        self.min_trajectory_length = min_trajectory_length
        self.bounce_detection_threshold = bounce_detection_threshold
        self.shots = []
        self.session_summary = {
            'session_info': {},
            'zone_performance': {},
            'overall_stats': {}
        }
    
    def analyze_trajectory(self, trajectory: List[Dict[str, Any]], 
                          track_id: int) -> Optional[Dict[str, Any]]:
        """
        Analyze a ball trajectory and classify the shot.
        
        Args:
            trajectory (list): List of trajectory points
            track_id (int): Unique track identifier
            
        Returns:
            Shot analysis result or None if trajectory is invalid
        """
        if len(trajectory) < self.min_trajectory_length:
            return None
        
        # Extract key information
        start_point = trajectory[0]
        end_point = trajectory[-1]
        
        # Transform coordinates to court space
        start_court = self.court_mapper.image_to_court_coords((start_point['x'], start_point['y']))
        end_court = self.court_mapper.image_to_court_coords((end_point['x'], end_point['y']))
        
        if start_court is None or end_court is None:
            return None
        
        # Analyze shot
        shot_result = self._classify_shot(start_court, end_court, trajectory)
        
        # Add metadata
        shot_result.update({
            'track_id': track_id,
            'frame_start': start_point['frame'],
            'frame_end': end_point['frame'],
            'timestamp_start': start_point.get('timestamp', 0),
            'timestamp_end': end_point.get('timestamp', 0),
            'trajectory_length': len(trajectory),
            'start_court_coords': start_court,
            'end_court_coords': end_court
        })
        
        return shot_result
    
    def _classify_shot(self, start_court: Tuple[float, float], 
                      end_court: Tuple[float, float], 
                      trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify a shot based on start and end positions.
        
        Args:
            start_court (tuple): Start position in court coordinates
            end_court (tuple): End position in court coordinates
            trajectory (list): Full trajectory data
            
        Returns:
            Shot classification result
        """
        # Check if end point is in court
        in_court = self.court_mapper.is_point_in_court(end_court)
        
        # Determine target zone
        target_zone = self.court_mapper.get_zone_at_point(end_court)
        
        # Classify shot result
        if not in_court:
            result = 'out_of_bounds'
        elif target_zone:
            result = 'hit'
        else:
            result = 'miss'
        
        # Calculate shot statistics
        shot_stats = self._calculate_shot_statistics(trajectory, start_court, end_court)
        
        return {
            'result': result,
            'target_zone': target_zone,
            'start_x': start_court[0],
            'start_y': start_court[1],
            'end_x': end_court[0],
            'end_y': end_court[1],
            'distance': shot_stats['distance'],
            'angle': shot_stats['angle'],
            'max_height': shot_stats['max_height'],
            'bounce_count': shot_stats['bounce_count']
        }
    
    def _calculate_shot_statistics(self, trajectory: List[Dict[str, Any]], 
                                 start_court: Tuple[float, float], 
                                 end_court: Tuple[float, float]) -> Dict[str, Any]:
        """
        Calculate various shot statistics.
        
        Args:
            trajectory (list): Trajectory data
            start_court (tuple): Start position
            end_court (tuple): End position
            
        Returns:
            Dictionary of shot statistics
        """
        # Calculate distance
        dx = end_court[0] - start_court[0]
        dy = end_court[1] - start_court[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Calculate angle
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Calculate max height (approximate from image coordinates)
        max_height = 0
        for point in trajectory:
            if 'y' in point:
                max_height = max(max_height, point['y'])
        
        # Count bounces (simplified - could be enhanced with velocity analysis)
        bounce_count = 0
        if len(trajectory) > 2:
            # Simple bounce detection based on direction changes
            for i in range(1, len(trajectory) - 1):
                prev_y = trajectory[i-1]['y']
                curr_y = trajectory[i]['y']
                next_y = trajectory[i+1]['y']
                
                # Check for direction change in Y (bounce)
                if (curr_y < prev_y and curr_y < next_y) or (curr_y > prev_y and curr_y > next_y):
                    bounce_count += 1
        
        return {
            'distance': distance,
            'angle': angle,
            'max_height': max_height,
            'bounce_count': bounce_count
        }
    
    def add_shot(self, shot_result: Dict[str, Any]):
        """
        Add a shot result to the analysis.
        
        Args:
            shot_result (dict): Shot analysis result
        """
        self.shots.append(shot_result)
    
    def generate_session_summary(self, session_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive session summary.
        
        Args:
            session_info (dict): Session metadata
            
        Returns:
            Session summary dictionary
        """
        self.session_summary['session_info'] = session_info
        
        # Calculate overall statistics
        total_shots = len(self.shots)
        hits = sum(1 for shot in self.shots if shot['result'] == 'hit')
        misses = sum(1 for shot in self.shots if shot['result'] == 'miss')
        out_of_bounds = sum(1 for shot in self.shots if shot['result'] == 'out_of_bounds')
        
        overall_accuracy = hits / total_shots if total_shots > 0 else 0
        
        self.session_summary['overall_stats'] = {
            'total_shots': total_shots,
            'hits': hits,
            'misses': misses,
            'out_of_bounds': out_of_bounds,
            'accuracy': overall_accuracy
        }
        
        # Calculate zone performance
        zone_stats = {}
        for shot in self.shots:
            zone = shot.get('target_zone', 'none')
            if zone not in zone_stats:
                zone_stats[zone] = {'attempts': 0, 'hits': 0, 'misses': 0, 'out_of_bounds': 0}
            
            zone_stats[zone]['attempts'] += 1
            zone_stats[zone][shot['result'] + 's'] += 1
        
        # Calculate zone accuracies
        for zone, stats in zone_stats.items():
            if stats['attempts'] > 0:
                stats['accuracy'] = stats['hits'] / stats['attempts']
            else:
                stats['accuracy'] = 0
        
        self.session_summary['zone_performance'] = zone_stats
        
        return self.session_summary
    
    def get_zone_performance(self, zone_name: str) -> Optional[Dict[str, Any]]:
        """
        Get performance statistics for a specific zone.
        
        Args:
            zone_name (str): Name of the zone
            
        Returns:
            Zone performance statistics or None if zone not found
        """
        return self.session_summary.get('zone_performance', {}).get(zone_name)
    
    def get_shot_trends(self) -> Dict[str, Any]:
        """
        Analyze shot trends over time.
        
        Returns:
            Dictionary of trend analysis
        """
        if len(self.shots) < 2:
            return {}
        
        # Calculate accuracy over time (every 5 shots)
        accuracy_trend = []
        shot_window = 5
        
        for i in range(shot_window, len(self.shots) + 1, shot_window):
            window_shots = self.shots[i-shot_window:i]
            window_hits = sum(1 for shot in window_shots if shot['result'] == 'hit')
            accuracy = window_hits / len(window_shots)
            accuracy_trend.append({
                'shot_range': f"{i-shot_window+1}-{i}",
                'accuracy': accuracy
            })
        
        # Calculate distance trends
        distances = [shot['distance'] for shot in self.shots]
        avg_distance = np.mean(distances) if distances else 0
        
        # Calculate consistency (standard deviation of distances)
        distance_std = np.std(distances) if len(distances) > 1 else 0
        
        return {
            'accuracy_trend': accuracy_trend,
            'average_distance': avg_distance,
            'distance_consistency': distance_std,
            'total_shots': len(self.shots)
        }
    
    def export_results(self, output_dir: str, session_name: str):
        """
        Export analysis results to files.
        
        Args:
            output_dir (str): Output directory path
            session_name (str): Name of the session
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export shot results to CSV
        csv_path = output_path / f"{session_name}_shots.csv"
        self._export_shots_csv(csv_path)
        
        # Export summary to JSON
        json_path = output_path / f"{session_name}_summary.json"
        self._export_summary_json(json_path)
    
    def _export_shots_csv(self, csv_path: Path):
        """Export shot results to CSV file."""
        import csv
        
        if not self.shots:
            return
        
        fieldnames = self.shots[0].keys()
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.shots)
    
    def _export_summary_json(self, json_path: Path):
        """Export session summary to JSON file."""
        with open(json_path, 'w') as jsonfile:
            json.dump(self.session_summary, jsonfile, indent=2)
    
    def get_recommendations(self) -> List[str]:
        """
        Generate performance recommendations based on analysis.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not self.shots:
            return ["No shots analyzed yet."]
        
        overall_stats = self.session_summary.get('overall_stats', {})
        accuracy = overall_stats.get('accuracy', 0)
        
        # Accuracy-based recommendations
        if accuracy < 0.5:
            recommendations.append("Focus on improving accuracy - consider practicing with slower shots first.")
        elif accuracy < 0.7:
            recommendations.append("Good progress! Try increasing shot speed while maintaining accuracy.")
        else:
            recommendations.append("Excellent accuracy! Consider practicing more challenging shot patterns.")
        
        # Zone-specific recommendations
        zone_performance = self.session_summary.get('zone_performance', {})
        for zone, stats in zone_performance.items():
            if zone != 'none' and stats['attempts'] >= 5:
                zone_accuracy = stats.get('accuracy', 0)
                if zone_accuracy < 0.6:
                    recommendations.append(f"Focus on improving accuracy in {zone} zone.")
        
        # Distance-based recommendations
        distances = [shot['distance'] for shot in self.shots]
        avg_distance = np.mean(distances) if distances else 0
        
        if avg_distance < 20:
            recommendations.append("Consider practicing longer shots to improve court coverage.")
        elif avg_distance > 40:
            recommendations.append("Try practicing shorter, more controlled shots for better precision.")
        
        return recommendations

class ShotClassifier:
    """Advanced shot classification based on trajectory analysis."""
    
    def __init__(self):
        """Initialize shot classifier."""
        self.shot_types = {
            'forehand': 'Forehand shot',
            'backhand': 'Backhand shot',
            'volley': 'Volley shot',
            'drop_shot': 'Drop shot',
            'lob': 'Lob shot'
        }
    
    def classify_shot_type(self, trajectory: List[Dict[str, Any]], 
                          start_court: Tuple[float, float], 
                          end_court: Tuple[float, float]) -> str:
        """
        Classify the type of shot based on trajectory characteristics.
        
        Args:
            trajectory (list): Ball trajectory
            start_court (tuple): Start position
            end_court (tuple): End position
            
        Returns:
            Shot type classification
        """
        # Calculate trajectory characteristics
        distance = np.sqrt((end_court[0] - start_court[0])**2 + (end_court[1] - start_court[1])**2)
        
        # Calculate max height
        max_height = max(point.get('y', 0) for point in trajectory)
        min_height = min(point.get('y', 0) for point in trajectory)
        height_range = max_height - min_height
        
        # Calculate trajectory arc
        arc_ratio = height_range / distance if distance > 0 else 0
        
        # Classify based on characteristics
        if arc_ratio > 0.5:
            return 'lob'
        elif distance < 15:
            return 'drop_shot'
        elif arc_ratio < 0.2:
            return 'volley'
        else:
            # Default to forehand/backhand based on direction
            dx = end_court[0] - start_court[0]
            return 'backhand' if dx < 0 else 'forehand'
    
    def get_shot_difficulty(self, shot_result: Dict[str, Any]) -> str:
        """
        Assess shot difficulty based on various factors.
        
        Args:
            shot_result (dict): Shot analysis result
            
        Returns:
            Difficulty level: 'easy', 'medium', 'hard'
        """
        distance = shot_result.get('distance', 0)
        angle = abs(shot_result.get('angle', 0))
        bounce_count = shot_result.get('bounce_count', 0)
        
        # Calculate difficulty score
        difficulty_score = 0
        
        # Distance factor
        if distance > 30:
            difficulty_score += 2
        elif distance > 20:
            difficulty_score += 1
        
        # Angle factor (cross-court shots are harder)
        if angle > 45:
            difficulty_score += 2
        elif angle > 20:
            difficulty_score += 1
        
        # Bounce factor
        if bounce_count > 1:
            difficulty_score += 1
        
        # Classify difficulty
        if difficulty_score >= 4:
            return 'hard'
        elif difficulty_score >= 2:
            return 'medium'
        else:
            return 'easy' 