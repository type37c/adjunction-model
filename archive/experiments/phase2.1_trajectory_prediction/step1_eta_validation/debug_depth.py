"""
Debug script to analyze depth image and conversion.
"""

import sys
import numpy as np
import pybullet as p
import pybullet_data

sys.path.append('/home/ubuntu/adjunction-model')
sys.path.append('/home/ubuntu/adjunction-model/experiments/phase2.1_trajectory_prediction/step1_eta_validation')

from env.point_cloud_utils import depth_image_to_point_cloud, filter_point_cloud


def main():
    """Main debug function."""
    print("="*60)
    print("Debug: Depth Image Conversion")
    print("="*60)
    
    # Create simple PyBullet scene
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # Load plane and box
    plane_id = p.loadURDF("plane.urdf")
    
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[0.8, 0.2, 0.2, 1.0])
    box_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=collision_shape, 
                                baseVisualShapeIndex=visual_shape, basePosition=[0, 0, 0.55])
    
    # Step simulation
    for _ in range(100):
        p.stepSimulation()
    
    # Get camera image
    width, height = 640, 480
    fov, near, far = 60, 0.1, 5.0
    camera_target = [0, 0, 0.5]
    camera_distance = 1.0
    camera_yaw = 0
    camera_pitch = -30
    
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=camera_distance,
        yaw=camera_yaw,
        pitch=camera_pitch,
        roll=0,
        upAxisIndex=2
    )
    
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=width / height,
        nearVal=near,
        farVal=far
    )
    
    img_arr = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER
    )
    
    depth = np.array(img_arr[3]).reshape(height, width)
    
    print(f"\nDepth buffer statistics:")
    print(f"  Shape: {depth.shape}")
    print(f"  Min: {depth.min():.6f}")
    print(f"  Max: {depth.max():.6f}")
    print(f"  Mean: {depth.mean():.6f}")
    print(f"  Unique values: {len(np.unique(depth))}")
    
    # Convert to point cloud
    points = depth_image_to_point_cloud(depth, view_matrix, proj_matrix, width, height, far, near)
    
    print(f"\nRaw point cloud (before filtering):")
    print(f"  Shape: {points.shape}")
    print(f"  Min: [{points[:, 0].min():.3f}, {points[:, 1].min():.3f}, {points[:, 2].min():.3f}]")
    print(f"  Max: [{points[:, 0].max():.3f}, {points[:, 1].max():.3f}, {points[:, 2].max():.3f}]")
    print(f"  Mean: [{points[:, 0].mean():.3f}, {points[:, 1].mean():.3f}, {points[:, 2].mean():.3f}]")
    
    # Check z range
    z_values = points[:, 2]
    print(f"\nZ coordinate analysis:")
    print(f"  Min Z: {z_values.min():.3f}")
    print(f"  Max Z: {z_values.max():.3f}")
    print(f"  Points with Z in [0.4, 1.0]: {((z_values >= 0.4) & (z_values <= 1.0)).sum()} / {len(z_values)}")
    print(f"  Points with Z in [0.0, 2.0]: {((z_values >= 0.0) & (z_values <= 2.0)).sum()} / {len(z_values)}")
    
    # Filter with different thresholds
    print(f"\nFiltering tests:")
    for z_min, z_max in [(0.4, 1.0), (0.0, 2.0), (-1.0, 2.0)]:
        filtered = filter_point_cloud(points, z_min=z_min, z_max=z_max)
        print(f"  z_min={z_min}, z_max={z_max}: {len(filtered)} points")
    
    p.disconnect(client)
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
