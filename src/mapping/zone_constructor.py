import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
try:
    from shapely.geometry import Polygon, Point
    from shapely import convex_hull
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Shapely not available, using fallback geometry methods")
import math

class ZoneConstructor:
    """Constructs target zones from cone detections."""
    
    def __init__(self, court_mapper, min_cones: int = 3, max_cones: int = 6):
        """
        Initialize zone constructor.
        
        Args:
            court_mapper: CourtMapper instance for coordinate transformation
            min_cones (int): Minimum number of cones to form a zone
            max_cones (int): Maximum number of cones to consider for a zone
        """
        self.court_mapper = court_mapper
        self.min_cones = min_cones
        self.max_cones = max_cones
        self.zones = {}
        self.cone_positions = []
        
    def add_cone_detections(self, cone_detections: List[Dict[str, Any]], frame_number: int = 0):
        """
        Add cone detections to the zone constructor.
        
        Args:
            cone_detections (list): List of cone detection dictionaries
            frame_number (int): Frame number for tracking
        """
        for detection in cone_detections:
            # Get center point of bounding box
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Transform to court coordinates
            court_coords = self.court_mapper.image_to_court_coords((center_x, center_y))
            
            if court_coords:
                cone_position = {
                    'image_coords': (center_x, center_y),
                    'court_coords': court_coords,
                    'bbox': bbox,
                    'confidence': detection['confidence'],
                    'frame': frame_number,
                    'class_name': detection['class_name']
                }
                self.cone_positions.append(cone_position)
    
    def construct_zones_auto(self, zone_name: str = "auto_zone") -> Dict[str, Any]:
        """
        Automatically construct zones from all detected cones.
        
        Args:
            zone_name (str): Name for the constructed zone
            
        Returns:
            Dictionary of constructed zones
        """
        if len(self.cone_positions) < self.min_cones:
            return {}
        
        # Get court coordinates of all cones
        court_points = [pos['court_coords'] for pos in self.cone_positions]
        
        # Create zone polygon
        zone_polygon = self._create_zone_polygon(court_points)
        
        if zone_polygon:
            self.zones[zone_name] = {
                'coordinates': zone_polygon,
                'cone_count': len(self.cone_positions),
                'cone_positions': self.cone_positions.copy(),
                'area': self._calculate_polygon_area(zone_polygon),
                'color': [255, 0, 0]  # Red by default
            }
        
        return self.zones
    
    def construct_zones_clustered(self, max_distance: float = 5.0) -> Dict[str, Any]:
        """
        Construct multiple zones by clustering nearby cones.
        
        Args:
            max_distance (float): Maximum distance (in feet) to group cones
            
        Returns:
            Dictionary of constructed zones
        """
        if len(self.cone_positions) < self.min_cones:
            return {}
        
        # Cluster cones based on proximity
        clusters = self._cluster_cones_by_distance(max_distance)
        
        zone_count = 0
        for cluster in clusters:
            if len(cluster) >= self.min_cones:
                zone_name = f"zone_{zone_count + 1}"
                court_points = [pos['court_coords'] for pos in cluster]
                zone_polygon = self._create_zone_polygon(court_points)
                
                if zone_polygon:
                    # Assign different colors to different zones
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                    color = colors[zone_count % len(colors)]
                    
                    self.zones[zone_name] = {
                        'coordinates': zone_polygon,
                        'cone_count': len(cluster),
                        'cone_positions': cluster,
                        'area': self._calculate_polygon_area(zone_polygon),
                        'color': list(color)
                    }
                    zone_count += 1
        
        return self.zones
    
    def construct_rectangular_zone(self, zone_name: str = "rectangular_zone") -> Dict[str, Any]:
        """
        Construct a rectangular zone from 4 corner cones.
        
        Args:
            zone_name (str): Name for the constructed zone
            
        Returns:
            Dictionary with the constructed zone
        """
        if len(self.cone_positions) < 4:
            return {}
        
        # Get the 4 corner cones (most extreme positions)
        court_points = [pos['court_coords'] for pos in self.cone_positions]
        corner_points = self._find_rectangle_corners(court_points)
        
        if len(corner_points) >= 4:
            self.zones[zone_name] = {
                'coordinates': corner_points[:4],  # Use first 4 corners
                'cone_count': len(self.cone_positions),
                'cone_positions': self.cone_positions.copy(),
                'area': self._calculate_polygon_area(corner_points[:4]),
                'color': [0, 255, 0]  # Green for rectangular zones
            }
        
        return self.zones
    
    def _create_zone_polygon(self, court_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Create a polygon from cone positions.
        
        Args:
            court_points (list): List of court coordinate tuples
            
        Returns:
            List of polygon vertices
        """
        if len(court_points) < 3:
            return []
        
        if SHAPELY_AVAILABLE:
            try:
                # Create Shapely points
                points = [Point(x, y) for x, y in court_points]
                
                # Create convex hull if more than 3 points
                if len(points) > 3:
                    from shapely.geometry import MultiPoint
                    multipoint = MultiPoint(points)
                    hull = multipoint.convex_hull
                    if hasattr(hull, 'exterior'):
                        coords = list(hull.exterior.coords[:-1])  # Remove duplicate last point
                    else:
                        coords = court_points
                else:
                    coords = court_points
                
                return coords
                
            except Exception:
                # Fallback to simple polygon ordering
                return self._order_points_clockwise(court_points)
        else:
            # Use fallback method when Shapely not available
            return self._order_points_clockwise(court_points)
    
    def _order_points_clockwise(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Order points in clockwise direction."""
        # Find centroid
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        
        # Sort by angle from centroid
        def angle_from_center(point):
            return math.atan2(point[1] - cy, point[0] - cx)
        
        return sorted(points, key=angle_from_center)
    
    def _find_rectangle_corners(self, court_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Find the 4 corner points that form the best rectangle."""
        if len(court_points) < 4:
            return court_points
        
        # Find extreme points
        points_array = np.array(court_points)
        
        # Find min/max x and y coordinates
        min_x_idx = np.argmin(points_array[:, 0])
        max_x_idx = np.argmax(points_array[:, 0])
        min_y_idx = np.argmin(points_array[:, 1])
        max_y_idx = np.argmax(points_array[:, 1])
        
        # Get the 4 extreme points
        extreme_indices = [min_x_idx, max_x_idx, min_y_idx, max_y_idx]
        extreme_points = [court_points[i] for i in set(extreme_indices)]
        
        # If we have exactly 4 points, order them as rectangle corners
        if len(extreme_points) == 4:
            return self._order_rectangle_corners(extreme_points)
        else:
            # Fill in missing corners or reduce to 4 points
            return self._order_points_clockwise(extreme_points)
    
    def _order_rectangle_corners(self, corners: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Order 4 corner points as top-left, top-right, bottom-right, bottom-left."""
        corners_array = np.array(corners)
        
        # Find centroid
        cx, cy = np.mean(corners_array, axis=0)
        
        # Classify corners based on position relative to centroid
        top_left = None
        top_right = None
        bottom_left = None
        bottom_right = None
        
        for corner in corners:
            x, y = corner
            if x < cx and y < cy:
                top_left = corner
            elif x > cx and y < cy:
                top_right = corner
            elif x < cx and y > cy:
                bottom_left = corner
            elif x > cx and y > cy:
                bottom_right = corner
        
        # Return in order, filling None values with closest available corner
        ordered = []
        for corner in [top_left, top_right, bottom_right, bottom_left]:
            if corner is not None:
                ordered.append(corner)
        
        # If we don't have all 4, just return sorted by angle
        if len(ordered) < 4:
            return self._order_points_clockwise(corners)
        
        return ordered
    
    def _cluster_cones_by_distance(self, max_distance: float) -> List[List[Dict[str, Any]]]:
        """
        Cluster cones based on distance.
        
        Args:
            max_distance (float): Maximum distance to group cones
            
        Returns:
            List of cone clusters
        """
        if not self.cone_positions:
            return []
        
        clusters = []
        used_indices = set()
        
        for i, cone1 in enumerate(self.cone_positions):
            if i in used_indices:
                continue
                
            cluster = [cone1]
            used_indices.add(i)
            
            # Find nearby cones
            for j, cone2 in enumerate(self.cone_positions):
                if j in used_indices:
                    continue
                
                distance = self._calculate_distance(cone1['court_coords'], cone2['court_coords'])
                if distance <= max_distance:
                    cluster.append(cone2)
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def _calculate_polygon_area(self, polygon: List[Tuple[float, float]]) -> float:
        """Calculate area of polygon using shoelace formula."""
        if len(polygon) < 3:
            return 0.0
        
        area = 0.0
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        
        return abs(area) / 2.0
    
    def get_zone_at_point(self, court_point: Tuple[float, float]) -> Optional[str]:
        """
        Determine which constructed zone a point belongs to.
        
        Args:
            court_point (tuple): (x, y) in court coordinates
            
        Returns:
            Zone name or None if not in any zone
        """
        if SHAPELY_AVAILABLE:
            try:
                point = Point(court_point)
                
                for zone_name, zone_data in self.zones.items():
                    polygon = Polygon(zone_data['coordinates'])
                    if polygon.contains(point) or polygon.touches(point):
                        return zone_name
            except Exception:
                # Fallback to simple point-in-polygon test
                for zone_name, zone_data in self.zones.items():
                    if self._point_in_polygon_simple(court_point, zone_data['coordinates']):
                        return zone_name
        else:
            # Use fallback method when Shapely not available
            for zone_name, zone_data in self.zones.items():
                if self._point_in_polygon_simple(court_point, zone_data['coordinates']):
                    return zone_name
        
        return None
    
    def _point_in_polygon_simple(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """Simple point-in-polygon test using ray casting."""
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
    
    def clear_zones(self):
        """Clear all constructed zones and cone positions."""
        self.zones.clear()
        self.cone_positions.clear()
    
    def get_zones(self) -> Dict[str, Any]:
        """Get all constructed zones."""
        return self.zones.copy()
    
    def get_zone_info(self, zone_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific zone."""
        return self.zones.get(zone_name)
    
    def create_mock_zones(self) -> Dict[str, Any]:
        """
        Create mock zones for testing when no cones are detected.
        
        Returns:
            Dictionary of mock zones
        """
        # Create a sample zone in the target area
        mock_zone_coords = [
            (35, 5),   # Top-left
            (44, 5),   # Top-right  
            (44, 15),  # Bottom-right
            (35, 15)   # Bottom-left
        ]
        
        self.zones["mock_target_zone"] = {
            'coordinates': mock_zone_coords,
            'cone_count': 4,
            'cone_positions': [],
            'area': self._calculate_polygon_area(mock_zone_coords),
            'color': [0, 255, 255]  # Yellow for mock zones
        }
        
        return self.zones