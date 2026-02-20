"""
Simple Robot Arm for Reaching Task

3-DOF planar robot arm in PyBullet.
"""

import pybullet as p
import numpy as np


class SimpleRobotArm:
    """
    3-DOF planar robot arm
    
    - 3 revolute joints
    - Operates in XY plane (Z is fixed)
    - Simple reaching task
    """
    
    def __init__(self, base_position=[0, 0, 0.5]):
        """
        Initialize robot arm
        
        Args:
            base_position: Base position of the robot
        """
        self.base_position = base_position
        self.num_joints = 3
        self.link_lengths = [0.3, 0.25, 0.2]  # Link lengths
        
        # Create robot using URDF (we'll create a simple one programmatically)
        self.robot_id = self._create_robot()
        
        # Joint limits
        self.joint_limits = [
            (-np.pi, np.pi),
            (-np.pi/2, np.pi/2),
            (-np.pi/2, np.pi/2)
        ]
        
        # Max torques
        self.max_torques = [10.0, 8.0, 5.0]
    
    def _create_robot(self):
        """Create robot programmatically"""
        # Base
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05], 
                                         rgbaColor=[0.5, 0.5, 0.5, 1.0])
        
        # Links
        link_masses = [1.0, 0.8, 0.5]
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
            # Collision shape (cylinder)
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.02,
                height=self.link_lengths[i]
            )
            link_collision_shapes.append(collision)
            
            # Visual shape
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.02,
                length=self.link_lengths[i],
                rgbaColor=[0.8, 0.3, 0.3, 1.0]
            )
            link_visual_shapes.append(visual)
            
            # Link position (relative to parent)
            if i == 0:
                link_positions.append([0, 0, self.link_lengths[i]/2])
            else:
                link_positions.append([0, 0, self.link_lengths[i-1]/2 + self.link_lengths[i]/2])
            
            link_orientations.append([0, 0, 0, 1])
            link_inertial_frame_positions.append([0, 0, 0])
            link_inertial_frame_orientations.append([0, 0, 0, 1])
            link_parent_indices.append(i)
            link_joint_types.append(p.JOINT_REVOLUTE)
            link_joint_axes.append([0, 0, 1])  # Rotate around Z axis
        
        # Create multi-body
        robot_id = p.createMultiBody(
            baseMass=2.0,
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
        
        return robot_id
    
    def reset(self, joint_angles=None):
        """
        Reset robot to initial configuration
        
        Args:
            joint_angles: Initial joint angles (default: all zeros)
        """
        if joint_angles is None:
            joint_angles = [0.0] * self.num_joints
        
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, joint_angles[i])
    
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
    
    def apply_action(self, torques):
        """
        Apply torques to joints
        
        Args:
            torques: np.array (num_joints,) - Joint torques
        """
        torques = np.clip(torques, -np.array(self.max_torques), np.array(self.max_torques))
        
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.TORQUE_CONTROL,
                force=float(torques[i])
            )
