"""
Simplified Composite Shape Dataset for Phase 2

This dataset generates composite objects using direct point cloud generation
instead of complex trimesh operations.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict


class CompositeShapeDataset(torch.utils.data.Dataset):
    """
    Dataset of composite 3D shapes for testing Agent C's ability to decompose
    unknown objects into known components using attention selection.
    
    Uses direct point cloud generation for simplicity and reliability.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        num_points: int = 1024,
        category: str = "all",
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.num_points = num_points
        self.category = category
        np.random.seed(seed)
        
        self.data = self._generate_dataset()
    
    def _generate_dataset(self) -> List[Dict]:
        """Generate composite shape dataset."""
        data = []
        
        categories = {
            "container": self._generate_containers,
            "tool": self._generate_tools,
            "structure": self._generate_structures,
            "composite": self._generate_composite_objects
        }
        
        if self.category == "all":
            selected_categories = list(categories.keys())
        else:
            selected_categories = [self.category]
        
        samples_per_category = self.num_samples // len(selected_categories)
        
        for cat in selected_categories:
            generator = categories[cat]
            for _ in range(samples_per_category):
                data.append(generator())
        
        return data
    
    def _ensure_exact_points(self, points: np.ndarray, segments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure points and segments have exactly num_points elements."""
        if len(points) > self.num_points:
            points = points[:self.num_points]
            segments = segments[:self.num_points]
        elif len(points) < self.num_points:
            extra_needed = self.num_points - len(points)
            extra_points = points[-extra_needed:]
            extra_segments = segments[-extra_needed:]
            points = np.vstack([points, extra_points])
            segments = np.concatenate([segments, extra_segments])
        return points, segments
    
    def _sample_cylinder(self, radius: float, height: float, num_points: int, center: np.ndarray = None) -> np.ndarray:
        """Generate points on a cylinder surface."""
        if center is None:
            center = np.zeros(3)
        
        # Sample points on cylinder surface
        theta = np.random.rand(num_points) * 2 * np.pi
        z = np.random.rand(num_points) * height - height / 2
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        points = np.column_stack([x, y, z]) + center
        return points
    
    def _sample_box(self, size: float, num_points: int, center: np.ndarray = None) -> np.ndarray:
        """Generate points on a box surface."""
        if center is None:
            center = np.zeros(3)
        
        #        # Sample points on 6 faces
        points_per_face = num_points // 6
        remainder = num_points % 6
        points = []
        
        for i in range(6):
            # Add extra point to first faces if there's a remainder
            n_points = points_per_face + (1 if i < remainder else 0)
            face_points = np.random.rand(n_points, 3) * size - size / 2
            if i == 0:  # +X face
                face_points[:, 0] = size / 2
            elif i == 1:  # -X face
                face_points[:, 0] = -size / 2
            elif i == 2:  # +Y face
                face_points[:, 1] = size / 2
            elif i == 3:  # -Y face
                face_points[:, 1] = -size / 2
            elif i == 4:  # +Z face
                face_points[:, 2] = size / 2
            else:  # -Z face
                face_points[:, 2] = -size / 2
            points.append(face_points)
        
        points = np.vstack(points) + center
        return points
    
    def _sample_sphere(self, radius: float, num_points: int, center: np.ndarray = None) -> np.ndarray:
        """Generate points on a sphere surface."""
        if center is None:
            center = np.zeros(3)
        
        # Sample points on sphere using spherical coordinates
        theta = np.random.rand(num_points) * 2 * np.pi
        phi = np.arccos(2 * np.random.rand(num_points) - 1)
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        points = np.column_stack([x, y, z]) + center
        return points
    
    def _generate_containers(self) -> Dict:
        """Generate container objects (cup, bowl, box)."""
        container_type = np.random.choice(["cup", "bowl", "box"])
        
        if container_type == "cup":
            # Cup: cylinder (body) + handle (small cylinder)
            body_radius = 0.3 + np.random.rand() * 0.2
            body_height = 0.5 + np.random.rand() * 0.3
            
            points_body = self._sample_cylinder(body_radius, body_height, self.num_points // 2)
            seg_body = np.zeros(self.num_points // 2, dtype=np.int64)
            
            # Handle (small cylinder on the side)
            handle_radius = 0.05
            handle_height = body_height * 0.4
            handle_center = np.array([body_radius + handle_radius, 0, body_height * 0.1])
            points_handle = self._sample_cylinder(handle_radius, handle_height, self.num_points // 2, handle_center)
            seg_handle = np.ones(self.num_points // 2, dtype=np.int64)
            
            points = np.vstack([points_body, points_handle])
            segments = np.concatenate([seg_body, seg_handle])
            points, segments = self._ensure_exact_points(points, segments)
            
            affordances = {
                0: ["graspable", "stackable", "containable"],
                1: ["graspable"]
            }
            
        elif container_type == "bowl":
            # Bowl: hemisphere (body) + disk (base)
            radius = 0.4 + np.random.rand() * 0.2
            
            # Hemisphere (upper half of sphere)
            sphere_points = self._sample_sphere(radius, self.num_points // 2)
            points_body = sphere_points[sphere_points[:, 2] >= 0]
            # Pad if needed
            if len(points_body) < self.num_points // 2:
                extra = self._sample_sphere(radius, self.num_points // 2 - len(points_body))
                extra = extra[extra[:, 2] >= 0]
                points_body = np.vstack([points_body, extra])
            points_body = points_body[:self.num_points // 2]
            seg_body = np.zeros(len(points_body), dtype=np.int64)
            
            # Disk base (thin cylinder)
            points_base = self._sample_cylinder(radius * 0.8, 0.05, self.num_points // 2, np.array([0, 0, -0.025]))
            seg_base = np.ones(self.num_points // 2, dtype=np.int64)
            
            points = np.vstack([points_body, points_base])
            segments = np.concatenate([seg_body, seg_base])
            points, segments = self._ensure_exact_points(points, segments)
            
            affordances = {
                0: ["containable", "stackable"],
                1: ["stackable"]
            }
            
        else:  # box
            # Box: hollow cube
            size = 0.5 + np.random.rand() * 0.3
            points = self._sample_box(size, self.num_points)
            segments = np.zeros(len(points), dtype=np.int64)
            points, segments = self._ensure_exact_points(points, segments)
            
            affordances = {
                0: ["graspable", "stackable", "containable"]
            }
        
        return {
            "points": torch.FloatTensor(points),
            "segments": torch.LongTensor(segments),
            "affordances": affordances,
            "category": "container",
            "type": container_type
        }
    
    def _generate_tools(self) -> Dict:
        """Generate tool objects (spoon, hook)."""
        tool_type = np.random.choice(["spoon", "hook"])
        
        if tool_type == "spoon":
            # Spoon: cylinder (handle) + hemisphere (scoop)
            handle_length = 0.6 + np.random.rand() * 0.2
            handle_radius = 0.05
            
            points_handle = self._sample_cylinder(handle_radius, handle_length, self.num_points // 2)
            seg_handle = np.zeros(self.num_points // 2, dtype=np.int64)
            
            # Scoop (small hemisphere at the end)
            scoop_radius = 0.15
            scoop_center = np.array([0, 0, handle_length / 2 + scoop_radius * 0.5])
            sphere_points = self._sample_sphere(scoop_radius, self.num_points // 2, scoop_center)
            points_scoop = sphere_points[sphere_points[:, 2] >= handle_length / 2]
            # Pad if needed
            if len(points_scoop) < self.num_points // 2:
                extra = self._sample_sphere(scoop_radius, self.num_points // 2 - len(points_scoop), scoop_center)
                points_scoop = np.vstack([points_scoop, extra])
            points_scoop = points_scoop[:self.num_points // 2]
            seg_scoop = np.ones(len(points_scoop), dtype=np.int64)
            
            points = np.vstack([points_handle, points_scoop])
            segments = np.concatenate([seg_handle, seg_scoop])
            points, segments = self._ensure_exact_points(points, segments)
            
            affordances = {
                0: ["graspable"],
                1: ["scoopable"]
            }
            
        else:  # hook
            # Hook: curved cylinder (simplified as bent cylinder)
            radius = 0.05
            
            # Create hook as two connected cylinders
            points_vertical = self._sample_cylinder(radius, 0.4, self.num_points // 2)
            points_horizontal = self._sample_cylinder(radius, 0.3, self.num_points // 2, np.array([0.15, 0, 0.2]))
            
            points = np.vstack([points_vertical, points_horizontal])
            segments = np.zeros(len(points), dtype=np.int64)
            points, segments = self._ensure_exact_points(points, segments)
            
            affordances = {
                0: ["graspable", "hookable"]
            }
        
        return {
            "points": torch.FloatTensor(points),
            "segments": torch.LongTensor(segments),
            "affordances": affordances,
            "category": "tool",
            "type": tool_type
        }
    
    def _generate_structures(self) -> Dict:
        """Generate structure objects (ring, shelf)."""
        structure_type = np.random.choice(["ring", "shelf"])
        
        if structure_type == "ring":
            # Ring: torus (simplified as thick cylinder)
            major_radius = 0.4 + np.random.rand() * 0.2
            minor_radius = 0.05 + np.random.rand() * 0.05
            
            # Sample points on torus
            theta = np.random.rand(self.num_points) * 2 * np.pi
            phi = np.random.rand(self.num_points) * 2 * np.pi
            
            x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
            y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
            z = minor_radius * np.sin(phi)
            
            points = np.column_stack([x, y, z])
            segments = np.zeros(len(points), dtype=np.int64)
            points, segments = self._ensure_exact_points(points, segments)
            
            affordances = {
                0: ["passable_through"]
            }
            
        else:  # shelf
            # Shelf: flat plane (thin box)
            width = 0.8 + np.random.rand() * 0.4
            depth = 0.4 + np.random.rand() * 0.2
            thickness = 0.05
            
            points = self._sample_box(width, self.num_points)
            # Flatten in Z direction
            points[:, 2] *= thickness / width
            segments = np.zeros(len(points), dtype=np.int64)
            points, segments = self._ensure_exact_points(points, segments)
            
            affordances = {
                0: ["placeable_on"]
            }
        
        return {
            "points": torch.FloatTensor(points),
            "segments": torch.LongTensor(segments),
            "affordances": affordances,
            "category": "structure",
            "type": structure_type
        }
    
    def _generate_composite_objects(self) -> Dict:
        """Generate composite objects (cube with handle)."""
        # Cube with handle
        cube_size = 0.5 + np.random.rand() * 0.2
        
        points_cube = self._sample_box(cube_size, self.num_points // 2)
        seg_cube = np.zeros(self.num_points // 2, dtype=np.int64)
        
        # Handle (cylinder on the side)
        handle_radius = 0.05
        handle_length = cube_size * 0.6
        handle_center = np.array([cube_size / 2 + handle_radius, 0, cube_size * 0.3])
        points_handle = self._sample_cylinder(handle_radius, handle_length, self.num_points // 2, handle_center)
        seg_handle = np.ones(self.num_points // 2, dtype=np.int64)
        
        points = np.vstack([points_cube, points_handle])
        segments = np.concatenate([seg_cube, seg_handle])
        points, segments = self._ensure_exact_points(points, segments)
        
        affordances = {
            0: ["stackable"],
            1: ["graspable"]
        }
        
        return {
            "points": torch.FloatTensor(points),
            "segments": torch.LongTensor(segments),
            "affordances": affordances,
            "category": "composite",
            "type": "cube_with_handle"
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


if __name__ == "__main__":
    # Test dataset generation
    dataset = CompositeShapeDataset(num_samples=10, category="all")
    print(f"Generated {len(dataset)} composite objects")
    
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Category: {sample['category']}")
        print(f"  Type: {sample['type']}")
        print(f"  Points shape: {sample['points'].shape}")
        print(f"  Segments shape: {sample['segments'].shape}")
        print(f"  Affordances: {sample['affordances']}")
