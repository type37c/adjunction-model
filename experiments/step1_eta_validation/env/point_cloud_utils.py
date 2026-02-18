"""
Point cloud utility functions for PyBullet depth image to point cloud conversion.
"""

import numpy as np


def depth_image_to_point_cloud(depth_buffer, view_matrix, proj_matrix, width, height, far, near):
    """
    Convert PyBullet depth buffer to 3D point cloud.
    
    Args:
        depth_buffer: Depth buffer from PyBullet camera (HxW array)
        view_matrix: View matrix from PyBullet camera (4x4)
        proj_matrix: Projection matrix from PyBullet camera (4x4)
        width: Image width
        height: Image height
        far: Far clipping plane
        near: Near clipping plane
    
    Returns:
        points: Nx3 numpy array of 3D points
    """
    # Convert depth buffer to linear depth
    depth_buffer = np.array(depth_buffer).reshape(height, width)
    depth = far * near / (far - (far - near) * depth_buffer)
    
    # Create pixel coordinates
    x_indices = np.arange(width)
    y_indices = np.arange(height)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)
    
    # Convert to normalized device coordinates
    x_ndc = (2.0 * x_grid - width) / width
    y_ndc = -(2.0 * y_grid - height) / height  # Flip y
    
    # Get projection matrix components
    proj_matrix = np.array(proj_matrix).reshape(4, 4)
    fx = proj_matrix[0, 0]
    fy = proj_matrix[1, 1]
    
    # Convert to camera coordinates
    x_cam = x_ndc * depth / fx
    y_cam = y_ndc * depth / fy
    z_cam = -depth  # Camera looks along -z
    
    # Stack to get points in camera frame
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    points_cam = points_cam.reshape(-1, 3)
    
    # Transform to world frame
    view_matrix = np.array(view_matrix).reshape(4, 4)
    view_matrix_inv = np.linalg.inv(view_matrix)
    
    # Add homogeneous coordinate
    points_cam_homo = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)
    points_world_homo = points_cam_homo @ view_matrix_inv.T
    points_world = points_world_homo[:, :3]
    
    return points_world


def sample_point_cloud(points, num_points=1024):
    """
    Sample fixed number of points from point cloud.
    
    Args:
        points: Nx3 numpy array
        num_points: Target number of points
    
    Returns:
        sampled_points: num_points x 3 numpy array
    """
    if len(points) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    
    if len(points) >= num_points:
        # Random sampling
        indices = np.random.choice(len(points), num_points, replace=False)
    else:
        # Upsample with replacement
        indices = np.random.choice(len(points), num_points, replace=True)
    
    return points[indices].astype(np.float32)


def filter_point_cloud(points, z_min=0.0, z_max=2.0, distance_threshold=2.0):
    """
    Filter point cloud to remove outliers and background.
    
    Args:
        points: Nx3 numpy array
        z_min: Minimum z coordinate (table height)
        z_max: Maximum z coordinate
        distance_threshold: Maximum distance from origin
    
    Returns:
        filtered_points: Mx3 numpy array (M <= N)
    """
    # Filter by z coordinate
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    
    # Filter by distance from origin
    distances = np.linalg.norm(points, axis=1)
    mask &= (distances <= distance_threshold)
    
    return points[mask]


def normalize_point_cloud(points):
    """
    Normalize point cloud to unit sphere centered at origin.
    
    Args:
        points: Nx3 numpy array
    
    Returns:
        normalized_points: Nx3 numpy array
        centroid: 3D centroid
        scale: Scale factor
    """
    if len(points) == 0:
        return points, np.zeros(3), 1.0
    
    # Center at origin
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # Scale to unit sphere
    max_distance = np.max(np.linalg.norm(points_centered, axis=1))
    if max_distance > 0:
        scale = 1.0 / max_distance
        points_normalized = points_centered * scale
    else:
        scale = 1.0
        points_normalized = points_centered
    
    return points_normalized, centroid, scale
