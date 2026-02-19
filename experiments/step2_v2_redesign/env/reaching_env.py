"""
PyBullet Reaching Environment v2

再設計のポイント:
1. 位置制御（トルク制御ではなく）
2. 改善された報酬関数（距離改善量 + 成功ボーナス + ステップペナルティ）
3. 豊かな状態表現（end-effector位置、相対ベクトル、正規化距離）
4. 複数ステップのシミュレーション（安定性向上）

哲学的根拠:
- 価値関数分析v3: 報酬 r(t) = α·Δη は「軌跡の瞬間的な傾き」
- ここでは外発的報酬として距離の改善量（Δd）を使用
- F/G統合版ではこれに内発的報酬（Δη）が加わる
"""

import numpy as np
import pybullet as p
import pybullet_data
from .robot_arm import SimpleRobotArm


class ReachingEnv:
    """
    Reaching task environment v2
    
    Goal: Move robot end effector close to target object
    
    State: [end_effector_pos(3), object_pos(3), relative_vec(3), 
            normalized_distance(1), joint_angles(3), joint_velocities(3)]
    Action: [Δθ_1, Δθ_2, Δθ_3] ∈ [-1, 1]^3
    Reward: distance_improvement + success_bonus - step_penalty
    """
    
    def __init__(self, gui=False, use_point_cloud=True, max_steps=100):
        """
        Initialize environment
        
        Args:
            gui: Whether to use GUI
            use_point_cloud: Whether to provide point cloud observations
            max_steps: Maximum steps per episode
        """
        self.gui = gui
        self.use_point_cloud = use_point_cloud
        self.max_steps = max_steps
        
        # Connect to PyBullet
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Create robot
        self.robot = SimpleRobotArm(base_position=[0, 0, 0.1])
        
        # Object types
        self.object_types = ['box', 'cylinder', 'sphere']
        self.object_id = None
        
        # Episode state
        self.current_step = 0
        self.prev_distance = None
        self.initial_distance = None
        self.object_position = None
        
        # Camera parameters (for point cloud)
        self.camera_distance = 1.5
        self.camera_yaw = 45
        self.camera_pitch = -30
        self.camera_target = [0, 0, 0.3]
        
        # Success threshold (relaxed for initial learning)
        self.success_threshold = 0.10
        
        # Simulation substeps per action
        self.sim_substeps = 10
        
        # Reward scaling
        self.distance_reward_scale = 20.0
        self.success_bonus = 100.0
        self.step_penalty = 0.05
        self.action_penalty_scale = 0.005
    
    def reset(self):
        """
        Reset environment
        
        Returns:
            observation: Initial observation dict
        """
        # Reset robot
        self.robot.reset()
        
        # Remove old object if exists
        if self.object_id is not None:
            p.removeBody(self.object_id)
        
        # Spawn random object at reachable position
        object_type = np.random.choice(self.object_types)
        
        # Object position: within robot's workspace
        # Robot is planar in XZ plane (Y≈0), total reach ≈ 0.75m, base at (0, 0, 0.1)
        # Place object in XZ plane at reachable positions
        x_sign = np.random.choice([-1, 1])
        object_pos = [
            x_sign * np.random.uniform(0.15, 0.45),  # X: reachable range
            0.0,                                       # Y: same plane as arm
            0.1 + np.random.uniform(0.2, 0.55),       # Z: within arm's vertical reach
        ]
        
        if object_type == 'box':
            size = np.random.uniform(0.03, 0.06)
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size]*3)
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[size]*3,
                                              rgbaColor=[0.2, 0.6, 0.8, 1.0])
        elif object_type == 'cylinder':
            radius_obj = np.random.uniform(0.02, 0.05)
            height = np.random.uniform(0.06, 0.12)
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius_obj, height=height)
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius_obj, length=height,
                                              rgbaColor=[0.8, 0.6, 0.2, 1.0])
        else:  # sphere
            radius_obj = np.random.uniform(0.03, 0.06)
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius_obj)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius_obj,
                                              rgbaColor=[0.6, 0.8, 0.2, 1.0])
        
        self.object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=object_pos
        )
        
        # Stabilize simulation
        for _ in range(100):
            p.stepSimulation()
        
        self.current_step = 0
        self.object_position = np.array(p.getBasePositionAndOrientation(self.object_id)[0])
        
        # Compute initial distance
        robot_state = self.robot.get_state()
        ee_pos = robot_state['end_effector_pos']
        self.initial_distance = np.linalg.norm(ee_pos - self.object_position)
        self.prev_distance = self.initial_distance
        
        return self._get_observation()
    
    def step(self, action):
        """
        Execute action (position control)
        
        Args:
            action: np.array (3,) - Joint angle deltas in [-1, 1]
        
        Returns:
            observation, reward, done, info
        """
        # Clip action
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action
        self.robot.apply_action(action)
        
        # Step simulation multiple times for stability
        for _ in range(self.sim_substeps):
            p.stepSimulation()
        
        # Get new state
        robot_state = self.robot.get_state()
        ee_pos = robot_state['end_effector_pos']
        
        # Compute distance
        distance = np.linalg.norm(ee_pos - self.object_position)
        
        # === Reward Design ===
        # 1. Distance improvement reward (Δd scaled)
        distance_improvement = self.prev_distance - distance
        reward_distance = distance_improvement * self.distance_reward_scale
        
        # 2. Success bonus
        success = distance < self.success_threshold
        reward_success = self.success_bonus if success else 0.0
        
        # 3. Step penalty (encourage efficiency)
        reward_step = -self.step_penalty
        
        # 4. Action smoothness penalty
        reward_action = -self.action_penalty_scale * np.sum(action ** 2)
        
        # Total extrinsic reward
        reward = reward_distance + reward_success + reward_step + reward_action
        
        # Update state
        self.prev_distance = distance
        self.current_step += 1
        done = (self.current_step >= self.max_steps) or success
        
        info = {
            'success': success,
            'distance': distance,
            'initial_distance': self.initial_distance,
            'distance_improvement_total': self.initial_distance - distance,
            'end_effector_pos': ee_pos.copy(),
            'object_position': self.object_position.copy(),
            'reward_components': {
                'distance': reward_distance,
                'success': reward_success,
                'step': reward_step,
                'action': reward_action,
            }
        }
        
        observation = self._get_observation()
        
        return observation, reward, done, info
    
    def _get_observation(self):
        """
        Get rich observation
        
        Returns:
            dict with:
                'state_vector': np.array (16,) - [ee_pos(3), obj_pos(3), relative(3), 
                                                   norm_dist(1), joint_angles(3), joint_vel(3)]
                'point_cloud': np.array (512, 3) - if use_point_cloud
        """
        robot_state = self.robot.get_state()
        ee_pos = robot_state['end_effector_pos']
        joint_angles = robot_state['joint_angles']
        joint_velocities = robot_state['joint_velocities']
        
        # Relative vector (object - end_effector)
        relative_vec = self.object_position - ee_pos
        
        # Normalized distance
        distance = np.linalg.norm(relative_vec)
        norm_distance = np.array([distance / max(self.initial_distance, 0.01)])
        
        # State vector: rich representation
        state_vector = np.concatenate([
            ee_pos,                    # 3: end-effector position
            self.object_position,      # 3: object position
            relative_vec,              # 3: relative vector (direction to target)
            norm_distance,             # 1: normalized distance
            joint_angles,              # 3: joint angles
            joint_velocities * 0.1,    # 3: joint velocities (scaled down)
        ]).astype(np.float32)
        
        obs = {
            'state_vector': state_vector,
            'joint_angles': joint_angles,
            'joint_velocities': joint_velocities,
            'end_effector_pos': ee_pos,
            'object_position': self.object_position,
        }
        
        if self.use_point_cloud:
            obs['point_cloud'] = self._get_point_cloud()
        
        return obs
    
    def _get_point_cloud(self, num_points=128):
        """
        Get point cloud from camera
        
        Returns:
            np.array (num_points, 3): Point cloud
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
            fov=60, aspect=1.0, nearVal=0.1, farVal=3.0
        )
        
        width, height = 128, 128
        _, _, _, depth_img, _ = p.getCameraImage(
            width=width, height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        
        depth_buffer = np.array(depth_img).reshape(height, width)
        
        fov = 60
        near = 0.1
        far = 3.0
        
        points = []
        for v in range(0, height, 2):
            for u in range(0, width, 2):
                depth = far * near / (far - (far - near) * depth_buffer[v, u])
                
                if depth < far * 0.9:
                    x = (u - width/2) / (width/2) * depth * np.tan(np.radians(fov/2))
                    y = -(v - height/2) / (height/2) * depth * np.tan(np.radians(fov/2))
                    z = -depth
                    points.append([x, y, z])
        
        points = np.array(points) if len(points) > 0 else np.zeros((1, 3))
        
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        elif len(points) < num_points:
            padding = np.zeros((num_points - len(points), 3))
            points = np.vstack([points, padding])
        
        return points.astype(np.float32)
    
    def close(self):
        """Close environment"""
        p.disconnect()
    
    @property
    def state_dim(self):
        """State vector dimension"""
        return 16  # 3+3+3+1+3+3
    
    @property
    def action_dim(self):
        """Action dimension"""
        return 3  # 3 joint angle deltas
