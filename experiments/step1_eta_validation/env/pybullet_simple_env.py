"""
Simple PyBullet environment for Step 1 eta validation experiment.
"""

import pybullet as p
import pybullet_data
import numpy as np
from .point_cloud_utils import (
    depth_image_to_point_cloud,
    filter_point_cloud,
    normalize_point_cloud,
    sample_point_cloud
)


class SimplePyBulletEnv:
    """Simple PyBullet environment with table and objects."""
    
    def __init__(self, object_type='cup', gui=False):
        """
        Initialize PyBullet environment.
        
        Args:
            object_type: Type of object ('cup', 'bowl', 'box')
            gui: Whether to use GUI mode
        """
        self.object_type = object_type
        self.gui = gui
        
        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Load plane (table)
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load object
        self.object_id = self._load_object(object_type)
        
        # Camera parameters
        self.camera_distance = 1.0
        self.camera_yaw = 0
        self.camera_pitch = -30
        self.camera_target = [0, 0, 0.5]
        self.width = 640
        self.height = 480
        self.fov = 60
        self.near = 0.1
        self.far = 5.0
        
    def _load_object(self, object_type):
        """Load object based on type."""
        if object_type == 'cup':
            # Create cylinder as cup approximation
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.04,
                height=0.1
            )
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.04,
                length=0.1,
                rgbaColor=[0.8, 0.2, 0.2, 1.0]
            )
            object_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0, 0, 0.55]
            )
        elif object_type == 'bowl':
            # Create sphere as bowl approximation
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=0.06
            )
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.06,
                rgbaColor=[0.2, 0.8, 0.2, 1.0]
            )
            object_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0, 0, 0.56]
            )
        elif object_type == 'box':
            # Create box
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.05, 0.05, 0.05]
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.05, 0.05, 0.05],
                rgbaColor=[0.2, 0.2, 0.8, 1.0]
            )
            object_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0, 0, 0.55]
            )
        else:
            raise ValueError(f"Unknown object type: {object_type}")
        
        return object_id
    
    def get_camera_image(self):
        """
        Get RGB-D image from camera.
        
        Returns:
            rgb: HxWx3 numpy array
            depth: HxW numpy array
            view_matrix: 4x4 view matrix
            proj_matrix: 4x4 projection matrix
        """
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far
        )
        
        img_arr = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        )
        
        rgb = np.array(img_arr[2]).reshape(self.height, self.width, 4)[:, :, :3]
        depth = np.array(img_arr[3]).reshape(self.height, self.width)
        
        return rgb, depth, view_matrix, proj_matrix
    
    def get_point_cloud(self, num_points=1024):
        """
        Get point cloud from current camera view.
        
        Args:
            num_points: Number of points to sample
        
        Returns:
            points: num_points x 3 numpy array (normalized)
        """
        rgb, depth, view_matrix, proj_matrix = self.get_camera_image()
        
        # Convert depth to point cloud
        points = depth_image_to_point_cloud(
            depth, view_matrix, proj_matrix,
            self.width, self.height, self.far, self.near
        )
        
        # Filter outliers
        points = filter_point_cloud(points, z_min=0.4, z_max=1.0)
        
        # Normalize
        points, _, _ = normalize_point_cloud(points)
        
        # Sample fixed number of points
        points = sample_point_cloud(points, num_points)
        
        return points
    
    def set_camera_view(self, yaw=None, pitch=None, distance=None):
        """Set camera viewpoint."""
        if yaw is not None:
            self.camera_yaw = yaw
        if pitch is not None:
            self.camera_pitch = pitch
        if distance is not None:
            self.camera_distance = distance
    
    def apply_force(self, force, position=None):
        """Apply force to object."""
        if position is None:
            position = [0, 0, 0]  # Center of mass
        p.applyExternalForce(
            self.object_id,
            -1,  # Link index (-1 for base)
            force,
            position,
            p.WORLD_FRAME
        )
    
    def reset_object(self):
        """Reset object to initial position."""
        if self.object_type == 'cup':
            p.resetBasePositionAndOrientation(
                self.object_id,
                [0, 0, 0.55],
                [0, 0, 0, 1]
            )
        elif self.object_type == 'bowl':
            p.resetBasePositionAndOrientation(
                self.object_id,
                [0, 0, 0.56],
                [0, 0, 0, 1]
            )
        elif self.object_type == 'box':
            p.resetBasePositionAndOrientation(
                self.object_id,
                [0, 0, 0.55],
                [0, 0, 0, 1]
            )
        
        # Reset velocity
        p.resetBaseVelocity(self.object_id, [0, 0, 0], [0, 0, 0])
    
    def get_object_state(self):
        """Get object position and orientation."""
        pos, orn = p.getBasePositionAndOrientation(self.object_id)
        vel, ang_vel = p.getBaseVelocity(self.object_id)
        return {
            'position': np.array(pos),
            'orientation': np.array(orn),
            'velocity': np.array(vel),
            'angular_velocity': np.array(ang_vel)
        }
    
    def step(self, num_steps=1):
        """Step simulation."""
        for _ in range(num_steps):
            p.stepSimulation()
    
    def close(self):
        """Close PyBullet connection."""
        p.disconnect(self.client)
