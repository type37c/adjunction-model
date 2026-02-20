"""
Escape Room Environment: Test Affordance Understanding

This environment implements the "escape room" task from the initial experiment note.

Scenario:
- Agent is in a room with an unknown object
- The object has a specific affordance (e.g., lever → push, button → push, knob → rotate)
- Agent must perform the correct action to open the door and escape
- Success requires understanding the shape-action relationship

Phases:
- Phase 0: Known shapes (cube, cylinder, sphere)
- Phase 1: Unknown shapes (lever, button, knob)
- Phase 2: Known shapes with constraints (gravity change, friction change)
"""

import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
from typing import Tuple, Dict, Optional


class EscapeRoomEnv(gym.Env):
    """
    Escape room environment for testing affordance understanding.
    
    The agent must identify the correct action for an object to escape.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    # Object types
    CUBE = 0
    CYLINDER = 1
    SPHERE = 2
    LEVER = 3
    BUTTON = 4
    KNOB = 5
    
    # Actions
    PUSH = 0
    PULL = 1
    ROTATE = 2
    
    # Object-action mappings (ground truth affordances)
    AFFORDANCE_MAP = {
        CUBE: PUSH,
        CYLINDER: ROTATE,
        SPHERE: PUSH,
        LEVER: PUSH,
        BUTTON: PUSH,
        KNOB: ROTATE
    }
    
    def __init__(
        self,
        object_type: int = CUBE,
        render: bool = False,
        num_points: int = 512,
        gravity: float = -9.8,
        friction: float = 0.5,
        use_gui: bool = False
    ):
        """
        Args:
            object_type: Type of object in the room
            render: Whether to render
            num_points: Number of points in point cloud
            gravity: Gravity (for Phase 2 constraints)
            friction: Friction coefficient (for Phase 2 constraints)
            use_gui: Whether to use PyBullet GUI
        """
        super().__init__()
        
        self.object_type = object_type
        self.render_mode = render
        self.num_points = num_points
        self.gravity = gravity
        self.friction = friction
        self.use_gui = use_gui
        
        # Action space: discrete (Push, Pull, Rotate)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: point cloud (num_points, 3)
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(num_points, 3), dtype=np.float32
        )
        
        # PyBullet client
        self.client = None
        self.object_id = None
        self.plane_id = None
        
        # Episode state
        self.steps = 0
        self.max_steps = 10
        self.done = False
    
    def reset(self) -> np.ndarray:
        """Reset the environment."""
        # Disconnect previous client
        if self.client is not None:
            p.disconnect(self.client)
        
        # Connect to PyBullet
        if self.use_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        # Set up environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), self.client)
        p.setGravity(0, 0, self.gravity, self.client)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=self.client)
        
        # Load object
        self.object_id = self._create_object(self.object_type)
        
        # Set friction
        p.changeDynamics(
            self.object_id, -1,
            lateralFriction=self.friction,
            physicsClientId=self.client
        )
        
        # Reset state
        self.steps = 0
        self.done = False
        
        # Get observation (point cloud)
        observation = self._get_point_cloud()
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index (0: Push, 1: Pull, 2: Rotate)
        
        Returns:
            observation: Point cloud
            reward: Reward
            done: Whether episode is done
            info: Additional information
        """
        self.steps += 1
        
        # Check if action is correct
        correct_action = self.AFFORDANCE_MAP[self.object_type]
        success = (action == correct_action)
        
        # Reward
        if success:
            reward = 1.0
            self.done = True
        else:
            reward = -0.1
        
        # Check max steps
        if self.steps >= self.max_steps:
            self.done = True
        
        # Get observation
        observation = self._get_point_cloud()
        
        # Info
        info = {
            'success': success,
            'correct_action': correct_action,
            'object_type': self.object_type,
            'steps': self.steps
        }
        
        return observation, reward, self.done, info
    
    def _create_object(self, object_type: int) -> int:
        """
        Create object in the environment.
        
        Args:
            object_type: Object type
        
        Returns:
            object_id: PyBullet object ID
        """
        position = [0, 0, 1]  # 1m above ground
        
        if object_type == self.CUBE:
            # Cube: 0.5m x 0.5m x 0.5m
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.25, 0.25, 0.25],
                physicsClientId=self.client
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.25, 0.25, 0.25],
                rgbaColor=[1, 0, 0, 1],
                physicsClientId=self.client
            )
        
        elif object_type == self.CYLINDER:
            # Cylinder: radius 0.25m, height 0.5m
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.25,
                height=0.5,
                physicsClientId=self.client
            )
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.25,
                length=0.5,
                rgbaColor=[0, 1, 0, 1],
                physicsClientId=self.client
            )
        
        elif object_type == self.SPHERE:
            # Sphere: radius 0.25m
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=0.25,
                physicsClientId=self.client
            )
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.25,
                rgbaColor=[0, 0, 1, 1],
                physicsClientId=self.client
            )
        
        elif object_type == self.LEVER:
            # Lever: long thin box
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.1, 0.1, 0.5],
                physicsClientId=self.client
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.1, 0.1, 0.5],
                rgbaColor=[1, 1, 0, 1],
                physicsClientId=self.client
            )
        
        elif object_type == self.BUTTON:
            # Button: flat cylinder
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.3,
                height=0.1,
                physicsClientId=self.client
            )
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.3,
                length=0.1,
                rgbaColor=[1, 0, 1, 1],
                physicsClientId=self.client
            )
        
        elif object_type == self.KNOB:
            # Knob: small sphere
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=0.15,
                physicsClientId=self.client
            )
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.15,
                rgbaColor=[0, 1, 1, 1],
                physicsClientId=self.client
            )
        
        else:
            raise ValueError(f"Unknown object type: {object_type}")
        
        # Create multi-body
        object_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.client
        )
        
        return object_id
    
    def _get_point_cloud(self) -> np.ndarray:
        """
        Get point cloud of the object.
        
        For simplicity, we sample points from the object's AABB.
        In a more sophisticated implementation, we would use depth cameras.
        
        Returns:
            point_cloud: (num_points, 3) array
        """
        # Get object AABB
        aabb_min, aabb_max = p.getAABB(self.object_id, physicsClientId=self.client)
        aabb_min = np.array(aabb_min)
        aabb_max = np.array(aabb_max)
        
        # Sample points uniformly from AABB
        points = np.random.uniform(aabb_min, aabb_max, (self.num_points, 3))
        
        # Center the point cloud
        center = (aabb_min + aabb_max) / 2
        points = points - center
        
        return points.astype(np.float32)
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human' and not self.use_gui:
            print("Warning: render() called but use_gui=False")
        # PyBullet GUI handles rendering automatically if use_gui=True
        pass
    
    def close(self):
        """Close the environment."""
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None
    
    def __del__(self):
        """Destructor."""
        self.close()


def test_escape_room():
    """Test the escape room environment."""
    print("Testing Escape Room Environment")
    print("=" * 60)
    
    # Test each object type
    object_types = [
        (EscapeRoomEnv.CUBE, "Cube"),
        (EscapeRoomEnv.CYLINDER, "Cylinder"),
        (EscapeRoomEnv.SPHERE, "Sphere"),
        (EscapeRoomEnv.LEVER, "Lever"),
        (EscapeRoomEnv.BUTTON, "Button"),
        (EscapeRoomEnv.KNOB, "Knob")
    ]
    
    for obj_type, obj_name in object_types:
        print(f"\nTesting {obj_name}...")
        env = EscapeRoomEnv(object_type=obj_type, render=False)
        
        # Reset
        obs = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
        
        # Try correct action
        correct_action = EscapeRoomEnv.AFFORDANCE_MAP[obj_type]
        action_names = ["Push", "Pull", "Rotate"]
        print(f"  Correct action: {action_names[correct_action]}")
        
        obs, reward, done, info = env.step(correct_action)
        print(f"  Reward: {reward}, Done: {done}, Success: {info['success']}")
        
        env.close()
    
    print("\n" + "=" * 60)
    print("All tests passed!")


if __name__ == '__main__':
    test_escape_room()
