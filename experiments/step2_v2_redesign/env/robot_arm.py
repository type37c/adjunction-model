"""
Simple Robot Arm for Reaching Task (v2)

Position-controlled 3-DOF planar robot arm in PyBullet.

修正点:
- 関節軸をY軸 [0, 1, 0] に変更（XZ平面で動くプレーナーアーム）
- リンクの向きを正しく設定
- 位置制御（トルク制御ではなく）
"""

import pybullet as p
import numpy as np


class SimpleRobotArm:
    """
    3-DOF planar robot arm with POSITION CONTROL
    
    Operates in XZ plane (Y-axis rotation joints).
    Total reach ≈ 0.75m (0.3 + 0.25 + 0.2)
    """
    
    def __init__(self, base_position=[0, 0, 0.1]):
        """
        Initialize robot arm
        
        Args:
            base_position: Base position of the robot
        """
        self.base_position = base_position
        self.num_joints = 3
        self.link_lengths = [0.3, 0.25, 0.2]
        
        # Create robot
        self.robot_id = self._create_robot()
        
        # Joint limits (radians)
        self.joint_lower = np.array([-np.pi * 0.8, -np.pi * 0.8, -np.pi * 0.8])
        self.joint_upper = np.array([np.pi * 0.8, np.pi * 0.8, np.pi * 0.8])
        
        # Position control parameters
        self.max_delta = 0.15  # Maximum joint angle change per step
        self.position_gain = 0.3
        self.velocity_gain = 1.0
        self.max_force = 100.0
        
        # Current target angles
        self.target_angles = np.zeros(self.num_joints)
    
    def _create_robot(self):
        """Create robot programmatically with Y-axis rotation joints"""
        # Base
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.05])
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.05],
                                         rgbaColor=[0.5, 0.5, 0.5, 1.0])
        
        link_masses = [0.5, 0.4, 0.3]
        link_collision_shapes = []
        link_visual_shapes = []
        link_positions = []
        link_orientations = []
        link_inertial_frame_positions = []
        link_inertial_frame_orientations = []
        link_parent_indices = []
        link_joint_types = []
        link_joint_axes = []
        
        for i in range(self.num_joints):
            # Collision shape (box for link)
            half_length = self.link_lengths[i] / 2
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.02, 0.02, half_length]
            )
            link_collision_shapes.append(collision)
            
            # Visual shape
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.02, 0.02, half_length],
                rgbaColor=[0.8, 0.3, 0.3, 1.0] if i == 0 else 
                          [0.3, 0.8, 0.3, 1.0] if i == 1 else
                          [0.3, 0.3, 0.8, 1.0]
            )
            link_visual_shapes.append(visual)
            
            # Link position relative to parent joint
            if i == 0:
                # First link: from base top
                link_positions.append([0, 0, self.link_lengths[i] / 2 + 0.05])
            else:
                # Subsequent links: from end of previous link
                link_positions.append([0, 0, self.link_lengths[i-1] / 2 + self.link_lengths[i] / 2])
            
            link_orientations.append([0, 0, 0, 1])
            
            # Inertial frame at center of link
            link_inertial_frame_positions.append([0, 0, 0])
            link_inertial_frame_orientations.append([0, 0, 0, 1])
            
            link_parent_indices.append(i)  # 0→base, 1→link0, 2→link1
            link_joint_types.append(p.JOINT_REVOLUTE)
            
            # KEY FIX: Rotate around Y axis for XZ plane motion
            link_joint_axes.append([0, 1, 0])
        
        robot_id = p.createMultiBody(
            baseMass=5.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=self.base_position,
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shapes,
            linkVisualShapeIndices=link_visual_shapes,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_frame_positions,
            linkInertialFrameOrientations=link_inertial_frame_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes
        )
        
        # Disable default velocity motors
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                robot_id, i, p.VELOCITY_CONTROL, force=0
            )
        
        return robot_id
    
    def reset(self, joint_angles=None):
        """
        Reset robot to initial configuration
        
        Args:
            joint_angles: Initial joint angles (default: slightly random)
        """
        if joint_angles is None:
            # Start with arm mostly upright, slight random perturbation
            joint_angles = np.random.uniform(-0.3, 0.3, size=self.num_joints)
        
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, joint_angles[i], 0.0)
        
        self.target_angles = np.array(joint_angles, dtype=np.float64)
        
        # Apply position control to hold position
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL,
                targetPosition=float(self.target_angles[i]),
                positionGain=self.position_gain,
                velocityGain=self.velocity_gain,
                force=self.max_force
            )
    
    def get_state(self):
        """
        Get current robot state
        
        Returns:
            dict: {
                'joint_angles': np.array (num_joints,),
                'joint_velocities': np.array (num_joints,),
                'end_effector_pos': np.array (3,)
            }
        """
        joint_states = [p.getJointState(self.robot_id, i) for i in range(self.num_joints)]
        
        joint_angles = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        
        # End effector position (last link)
        link_state = p.getLinkState(self.robot_id, self.num_joints - 1)
        end_effector_pos = np.array(link_state[0])
        
        return {
            'joint_angles': joint_angles,
            'joint_velocities': joint_velocities,
            'end_effector_pos': end_effector_pos
        }
    
    def apply_action(self, delta_angles):
        """
        Apply position control action (joint angle deltas)
        
        Args:
            delta_angles: np.array (num_joints,) - Desired change in joint angles
                         Values should be in [-1, 1], will be scaled by max_delta
        """
        delta_angles = np.clip(delta_angles, -1.0, 1.0) * self.max_delta
        
        self.target_angles = self.target_angles + delta_angles
        self.target_angles = np.clip(self.target_angles, self.joint_lower, self.joint_upper)
        
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL,
                targetPosition=float(self.target_angles[i]),
                positionGain=self.position_gain,
                velocityGain=self.velocity_gain,
                force=self.max_force
            )
