"""
PyBullet Reaching Environment

Simple reaching task: move robot arm end effector to target object.
"""

import numpy as np
import pybullet as p
import pybullet_data
from .robot_arm import SimpleRobotArm


class ReachingEnv:
    """
    Reaching task environment
    
    Goal: Move robot end effector close to target object
    Reward: -distance(end_effector, object)
    Success: distance < 0.05m
    """
    
    def __init__(self, gui=False, use_point_cloud=True):
        """
        Initialize environment
        
        Args:
            gui: Whether to use GUI
            use_point_cloud: Whether to provide point cloud observations
        """
        self.gui = gui
        self.use_point_cloud = use_point_cloud
        
        # Connect to PyBullet
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Create robot
        self.robot = SimpleRobotArm(base_position=[0, 0, 0.1])
        
        # Object types
        self.object_types = ['box', 'cylinder', 'sphere']
        self.object_id = None
        
        # Episode parameters
        self.max_steps = 200
        self.current_step = 0
        
        # Camera parameters (for point cloud)
        self.camera_distance = 1.5
        self.camera_yaw = 45
        self.camera_pitch = -30
        self.camera_target = [0, 0, 0.3]
        
        # Success threshold
        self.success_threshold = 0.05
    
    def reset(self):
        """
        Reset environment
        
        Returns:
            observation: Initial observation
        """
        # Reset robot
        self.robot.reset()
        
        # Remove old object if exists
        if self.object_id is not None:
            p.removeBody(self.object_id)
        
        # Spawn random object
        object_type = np.random.choice(self.object_types)
        object_pos = [
            np.random.uniform(0.3, 0.6),
            np.random.uniform(-0.3, 0.3),
            0.15
        ]
        
        if object_type == 'box':
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05],
                                              rgbaColor=[0.2, 0.6, 0.8, 1.0])
        elif object_type == 'cylinder':
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.04, height=0.1)
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.04, length=0.1,
                                              rgbaColor=[0.8, 0.6, 0.2, 1.0])
        else:  # sphere
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.05,
                                              rgbaColor=[0.6, 0.8, 0.2, 1.0])
        
        self.object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=object_pos
        )
        
        # Stabilize simulation
        for _ in range(50):
            p.stepSimulation()
        
        self.current_step = 0
        self.object_position = np.array(p.getBasePositionAndOrientation(self.object_id)[0])
        
        return self._get_observation()
    
    def step(self, action):
        """
        Execute action
        
        Args:
            action: np.array (num_joints,) - Joint torques
        
        Returns:
            observation: Next observation
            reward: Reward
            done: Whether episode is done
            info: Additional information
        """
        # Apply action
        self.robot.apply_action(action)
        
        # Step simulation
        p.stepSimulation()
        
        # Get new state
        robot_state = self.robot.get_state()
        end_effector_pos = robot_state['end_effector_pos']
        
        # Compute reward
        distance = np.linalg.norm(end_effector_pos - self.object_position)
        reward = -distance
        
        # Check success
        success = distance < self.success_threshold
        if success:
            reward += 10.0
        
        # Check done
        self.current_step += 1
        done = (self.current_step >= self.max_steps) or success
        
        info = {
            'success': success,
            'distance': distance,
            'end_effector_pos': end_effector_pos,
            'object_position': self.object_position
        }
        
        observation = self._get_observation()
        
        return observation, reward, done, info
    
    def _get_observation(self):
        """
        Get observation
        
        Returns:
            dict: Observation dictionary
        """
        robot_state = self.robot.get_state()
        
        obs = {
            'joint_angles': robot_state['joint_angles'],
            'joint_velocities': robot_state['joint_velocities'],
            'object_position': self.object_position
        }
        
        if self.use_point_cloud:
            obs['point_cloud'] = self._get_point_cloud()
        
        return obs
    
    def _get_point_cloud(self, num_points=512):
        """
        Get point cloud from camera
        
        Returns:
            np.array (num_points, 3): Point cloud
        """
        # Camera parameters
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.0
        )
        
        # Get depth image
        width, height = 128, 128
        _, _, _, depth_img, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        
        # Convert depth to point cloud
        depth_buffer = np.array(depth_img).reshape(height, width)
        
        # Get camera intrinsics
        fov = 60
        aspect = 1.0
        near = 0.1
        far = 3.0
        
        # Compute point cloud
        points = []
        for v in range(0, height, 2):  # Subsample
            for u in range(0, width, 2):
                depth = far * near / (far - (far - near) * depth_buffer[v, u])
                
                if depth < far * 0.9:  # Filter far points
                    # Pixel to camera coordinates
                    x = (u - width/2) / (width/2) * depth * np.tan(np.radians(fov/2)) * aspect
                    y = -(v - height/2) / (height/2) * depth * np.tan(np.radians(fov/2))
                    z = -depth
                    
                    points.append([x, y, z])
        
        points = np.array(points)
        
        if len(points) == 0:
            points = np.zeros((num_points, 3))
        else:
            # Subsample to num_points
            if len(points) > num_points:
                indices = np.random.choice(len(points), num_points, replace=False)
                points = points[indices]
            elif len(points) < num_points:
                # Pad with zeros
                padding = np.zeros((num_points - len(points), 3))
                points = np.vstack([points, padding])
        
        return points.astype(np.float32)
    
    def close(self):
        """Close environment"""
        p.disconnect()
    
    def get_observation_space_dims(self):
        """
        Get observation space dimensions
        
        Returns:
            dict: Dimension information
        """
        dims = {
            'joint_angles': self.robot.num_joints,
            'joint_velocities': self.robot.num_joints,
            'object_position': 3
        }
        
        if self.use_point_cloud:
            dims['point_cloud'] = (512, 3)
        
        return dims
    
    def get_action_space_dim(self):
        """
        Get action space dimension
        
        Returns:
            int: Action dimension
        """
        return self.robot.num_joints
