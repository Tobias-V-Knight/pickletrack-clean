import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import yaml
from pathlib import Path

class CourtMapper:
    """Handles court coordinate mapping and zone detection."""
    
    def __init__(self, court_config_path: str = "config/court_config.yaml"):
        """
        Initialize court mapper.
        
        Args:
            court_config_path (str): Path to court configuration file
        """
        self.court_config = self._load_court_config(court_config_path)
        self.homography_matrix = None
        self.court_corners = None
        self.zones = self.court_config.get('zones', {})
        
    def _load_court_config(self, config_path: str) -> dict:
        """Load court configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Court config file {config_path} not found. Using defaults.")
            return {
                'court_dimensions': {'width': 44, 'height': 20},
                'zones': {}
            }
    
    def set_court_corners(self, corners: List[Tuple[float, float]]):
        """
        Set the four corners of the court in image coordinates.
        
        Args:
            corners (list): List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        if len(corners) != 4:
            raise ValueError("Exactly 4 corner points required")
        
        self.court_corners = np.array(corners, dtype=np.float32)
        
        # Define target court coordinates (in feet)
        court_width = self.court_config['court_dimensions']['width']
        court_height = self.court_config['court_dimensions']['height']
        
        # Target corners: top-left, top-right, bottom-right, bottom-left
        target_corners = np.array([
            [0, 0],
            [court_width, 0],
            [court_width, court_height],
            [0, court_height]
        ], dtype=np.float32)
        
        # Calculate homography matrix
        self.homography_matrix = cv2.getPerspectiveTransform(
            self.court_corners, target_corners
        )
    
    def image_to_court_coords(self, image_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Transform image coordinates to court coordinates.
        
        Args:
            image_point (tuple): (x, y) in image coordinates
            
        Returns:
            Tuple of (x, y) in court coordinates (feet) or None if transformation failed
        """
        if self.homography_matrix is None:
            return None
        
        point = np.array([[image_point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        
        if transformed is not None and len(transformed) > 0:
            return (transformed[0][0], transformed[0][1])
        
        return None
    
    def court_to_image_coords(self, court_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Transform court coordinates to image coordinates.
        
        Args:
            court_point (tuple): (x, y) in court coordinates (feet)
            
        Returns:
            Tuple of (x, y) in image coordinates or None if transformation failed
        """
        if self.homography_matrix is None:
            return None
        
        # Use inverse homography
        inv_homography = np.linalg.inv(self.homography_matrix)
        point = np.array([[court_point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, inv_homography)
        
        if transformed is not None and len(transformed) > 0:
            return (transformed[0][0], transformed[0][1])
        
        return None
    
    def get_zone_at_point(self, court_point: Tuple[float, float]) -> Optional[str]:
        """
        Determine which zone a court point belongs to.
        
        Args:
            court_point (tuple): (x, y) in court coordinates
            
        Returns:
            Zone name or None if not in any zone
        """
        x, y = court_point
        
        for zone_name, zone_config in self.zones.items():
            coords = zone_config.get('coordinates', [])
            if len(coords) == 2:
                # Rectangle zone
                (x1, y1), (x2, y2) = coords
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return zone_name
            elif len(coords) > 2:
                # Polygon zone
                polygon = np.array(coords, dtype=np.float32)
                point = np.array([[x, y]], dtype=np.float32)
                if cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
                    return zone_name
        
        return None
    
    def get_zone_color(self, zone_name: str) -> Tuple[int, int, int]:
        """
        Get the color for a zone.
        
        Args:
            zone_name (str): Name of the zone
            
        Returns:
            BGR color tuple
        """
        if zone_name in self.zones:
            color = self.zones[zone_name].get('color', [0, 255, 0])
            return tuple(color)
        return (0, 255, 0)  # Default green
    
    def draw_court_overlay(self, image: np.ndarray, draw_zones: bool = True) -> np.ndarray:
        """
        Draw court overlay on image.
        
        Args:
            image (np.ndarray): Input image
            draw_zones (bool): Whether to draw zone boundaries
            
        Returns:
            Image with court overlay
        """
        if self.court_corners is None:
            return image
        
        # Draw court boundary
        corners_int = self.court_corners.astype(np.int32)
        cv2.polylines(image, [corners_int], True, (255, 255, 255), 2)
        
        if draw_zones and self.homography_matrix is not None:
            # Draw zones
            for zone_name, zone_config in self.zones.items():
                coords = zone_config.get('coordinates', [])
                color = self.get_zone_color(zone_name)
                
                if len(coords) == 2:
                    # Rectangle zone
                    (x1, y1), (x2, y2) = coords
                    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    self._draw_zone_polygon(image, corners, color, zone_name)
                elif len(coords) > 2:
                    # Polygon zone
                    self._draw_zone_polygon(image, coords, color, zone_name)
        
        return image
    
    def _draw_zone_polygon(self, image: np.ndarray, court_coords: List[Tuple[float, float]], 
                          color: Tuple[int, int, int], zone_name: str):
        """Draw a zone polygon on the image."""
        # Transform court coordinates to image coordinates
        image_coords = []
        for coord in court_coords:
            img_coord = self.court_to_image_coords(coord)
            if img_coord:
                image_coords.append(img_coord)
        
        if len(image_coords) >= 3:
            # Draw filled polygon with transparency
            overlay = image.copy()
            coords_array = np.array(image_coords, dtype=np.int32)
            cv2.fillPoly(overlay, [coords_array], color)
            
            # Blend with original image
            alpha = 0.3
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            # Draw boundary
            cv2.polylines(image, [coords_array], True, color, 2)
            
            # Add zone label
            if image_coords:
                center_x = sum(coord[0] for coord in image_coords) / len(image_coords)
                center_y = sum(coord[1] for coord in image_coords) / len(image_coords)
                
                cv2.putText(image, zone_name, (int(center_x), int(center_y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def is_point_in_court(self, court_point: Tuple[float, float]) -> bool:
        """
        Check if a point is within the court boundaries.
        
        Args:
            court_point (tuple): (x, y) in court coordinates
            
        Returns:
            bool: True if point is in court
        """
        x, y = court_point
        court_width = self.court_config['court_dimensions']['width']
        court_height = self.court_config['court_dimensions']['height']
        
        return 0 <= x <= court_width and 0 <= y <= court_height
    
    def get_court_dimensions(self) -> Tuple[float, float]:
        """Get court dimensions in feet."""
        dims = self.court_config['court_dimensions']
        return (dims['width'], dims['height'])
    
    def get_zone_coordinates(self, zone_name: str) -> Optional[List[Tuple[float, float]]]:
        """
        Get coordinates for a specific zone.
        
        Args:
            zone_name (str): Name of the zone
            
        Returns:
            List of coordinate tuples or None if zone not found
        """
        if zone_name in self.zones:
            return self.zones[zone_name].get('coordinates', [])
        return None

class AutomaticCourtDetector:
    """Automatically detect court corners using computer vision."""
    
    def __init__(self):
        """Initialize automatic court detector."""
        pass
    
    def detect_court_corners(self, image: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """
        Automatically detect court corners in image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List of 4 corner points or None if detection failed
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (likely the court)
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we have 4 points, we found a rectangle
        if len(approx) == 4:
            corners = []
            for point in approx:
                corners.append((point[0][0], point[0][1]))
            return corners
        
        return None
    
    def refine_corners(self, image: np.ndarray, initial_corners: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Refine corner detection using sub-pixel accuracy.
        
        Args:
            image (np.ndarray): Input image
            initial_corners (list): Initial corner estimates
            
        Returns:
            Refined corner points
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convert corners to numpy array
        corners = np.array(initial_corners, dtype=np.float32)
        
        # Refine corners
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        return [(float(x), float(y)) for x, y in refined_corners] 