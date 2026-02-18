"""
Point cloud interior filling utilities.

Converts surface-only point clouds (from depth images) to filled volume point clouds
(matching Phase 1 training data distribution).
"""

import numpy as np
from scipy.spatial import ConvexHull


def fill_point_cloud_convex_hull(surface_points, target_num_points=512):
    """
    Fill the interior of a point cloud using convex hull sampling.
    
    Args:
        surface_points: Nx3 numpy array of surface points
        target_num_points: Target number of points for filled cloud
    
    Returns:
        filled_points: target_num_points x 3 numpy array
    """
    if len(surface_points) < 4:
        # Not enough points for convex hull, just return resampled surface
        return sample_points(surface_points, target_num_points)
    
    try:
        # Compute bounding box
        min_coords = np.min(surface_points, axis=0)
        max_coords = np.max(surface_points, axis=0)
        
        # Generate random points in bounding box
        num_candidates = target_num_points * 10
        candidates = np.random.uniform(
            low=min_coords,
            high=max_coords,
            size=(num_candidates, 3)
        )
        
        # Compute convex hull
        hull = ConvexHull(surface_points)
        
        # Filter points inside convex hull
        inside_points = []
        for point in candidates:
            # Check if point is inside hull using equations
            is_inside = True
            for equation in hull.equations:
                if np.dot(equation[:3], point) + equation[3] > 1e-6:
                    is_inside = False
                    break
            if is_inside:
                inside_points.append(point)
                if len(inside_points) >= target_num_points:
                    break
        
        if len(inside_points) < target_num_points:
            # Not enough interior points, add some surface points
            num_surface = target_num_points - len(inside_points)
            surface_sample = sample_points(surface_points, num_surface)
            inside_points.extend(surface_sample)
        
        filled_points = np.array(inside_points[:target_num_points], dtype=np.float32)
        
    except Exception as e:
        # Fallback: just return resampled surface points
        print(f"Warning: Convex hull filling failed ({e}), using surface points only")
        filled_points = sample_points(surface_points, target_num_points)
    
    return filled_points


def fill_point_cloud_simple(surface_points, target_num_points=512, fill_ratio=0.7):
    """
    Fill the interior of a point cloud using simple random sampling.
    
    This is a faster but less accurate method than convex hull.
    
    Args:
        surface_points: Nx3 numpy array of surface points
        target_num_points: Target number of points for filled cloud
        fill_ratio: Ratio of interior points to total points
    
    Returns:
        filled_points: target_num_points x 3 numpy array
    """
    # Handle empty point cloud
    if len(surface_points) == 0:
        return np.zeros((target_num_points, 3), dtype=np.float32)
    
    # Compute bounding box
    min_coords = np.min(surface_points, axis=0)
    max_coords = np.max(surface_points, axis=0)
    center = (min_coords + max_coords) / 2
    
    # Compute approximate radius
    distances = np.linalg.norm(surface_points - center, axis=1)
    radius = np.mean(distances)
    
    # Generate interior points
    num_interior = int(target_num_points * fill_ratio)
    num_surface = target_num_points - num_interior
    
    # Sample interior points (uniform in sphere)
    interior_points = []
    while len(interior_points) < num_interior:
        # Random point in bounding box
        point = np.random.uniform(low=min_coords, high=max_coords)
        
        # Check if inside approximate sphere
        if np.linalg.norm(point - center) <= radius:
            interior_points.append(point)
    
    interior_points = np.array(interior_points, dtype=np.float32)
    
    # Sample surface points
    surface_sample = sample_points(surface_points, num_surface)
    
    # Combine
    filled_points = np.vstack([interior_points, surface_sample])
    
    return filled_points


def sample_points(points, num_points):
    """Sample fixed number of points from point cloud."""
    if len(points) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    
    if len(points) >= num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
    else:
        indices = np.random.choice(len(points), num_points, replace=True)
    
    return points[indices].astype(np.float32)
