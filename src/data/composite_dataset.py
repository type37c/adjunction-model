"""
Composite Shape Dataset for Phase 2

This dataset generates composite objects (containers, tools, structures, complex objects)
with semantic segmentation labels for attention selection experiments.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
import trimesh


class CompositeShapeDataset(torch.utils.data.Dataset):
    """
    Dataset of composite 3D shapes for testing Agent C's ability to decompose
    unknown objects into known components using attention selection.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        num_points: int = 1024,
        category: str = "all",  # "all", "container", "tool", "structure", "composite"
        seed: int = 42
    ):
        """
        Args:
            num_samples: Number of composite objects to generate
            num_points: Number of points to sample from each object
            category: Category of composite objects to generate
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.category = category
        np.random.seed(seed)
        
        # Generate dataset
        self.data = self._generate_dataset()
    
    def _generate_dataset(self) -> List[Dict]:
        """Generate composite shape dataset with segmentation labels."""
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
    
    def _generate_containers(self) -> Dict:
        """Generate container objects (cup, bowl, box)."""
        container_type = np.random.choice(["cup", "bowl", "box"])
        
        if container_type == "cup":
            # Cup: cylinder (body) + torus (handle)
            body_radius = 0.3 + np.random.rand() * 0.2
            body_height = 0.5 + np.random.rand() * 0.3
            
            # Cylinder body
            cylinder = trimesh.creation.cylinder(
                radius=body_radius,
                height=body_height,
                sections=32
            )
            
            # Torus handle
            handle = trimesh.creation.annulus(
                r_min=body_radius * 0.8,
                r_max=body_radius * 1.0,
                height=body_height * 0.3,
                sections=16
            )
            handle.apply_translation([body_radius * 1.2, 0, body_height * 0.2])
            
            # Combine and sample points
            points_body, seg_body = self._sample_points_from_mesh(cylinder, self.num_points // 2, segment_id=0)
            points_handle, seg_handle = self._sample_points_from_mesh(handle, self.num_points // 2, segment_id=1)
            
            points = np.vstack([points_body, points_handle])
            segments = np.concatenate([seg_body, seg_handle])
            
            affordances = {
                0: ["graspable", "stackable", "containable"],  # body
                1: ["graspable"]  # handle
            }
            
        elif container_type == "bowl":
            # Bowl: hemisphere (body) + disk (base)
            radius = 0.4 + np.random.rand() * 0.2
            
            # Hemisphere
            sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
            # Keep only upper half
            sphere.vertices = sphere.vertices[sphere.vertices[:, 2] >= 0]
            
            # Disk base
            disk = trimesh.creation.cylinder(radius=radius * 0.8, height=0.05, sections=32)
            disk.apply_translation([0, 0, -0.025])
            
            points_body, seg_body = self._sample_points_from_mesh(sphere, self.num_points // 2, segment_id=0)
            points_base, seg_base = self._sample_points_from_mesh(disk, self.num_points // 2, segment_id=1)
            
            points = np.vstack([points_body, points_base])
            segments = np.concatenate([seg_body, seg_base])
            
            affordances = {
                0: ["containable", "stackable"],  # body
                1: ["stackable"]  # base
            }
            
        else:  # box
            # Box: hollow cube
            size = 0.5 + np.random.rand() * 0.3
            box = trimesh.creation.box(extents=[size, size, size])
            
            points, segments = self._sample_points_from_mesh(box, self.num_points, segment_id=0)
            
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
            # Spoon: cylinder (handle) + shallow hemisphere (scoop)
            handle_length = 0.6 + np.random.rand() * 0.2
            handle_radius = 0.05
            
            # Handle
            handle = trimesh.creation.cylinder(
                radius=handle_radius,
                height=handle_length,
                sections=16
            )
            handle.apply_translation([0, 0, handle_length / 2])
            
            # Scoop (shallow hemisphere)
            scoop_radius = 0.15
            scoop = trimesh.creation.icosphere(subdivisions=2, radius=scoop_radius)
            scoop.vertices = scoop.vertices[scoop.vertices[:, 2] >= -scoop_radius * 0.3]
            scoop.apply_translation([0, 0, handle_length + scoop_radius * 0.5])
            
            points_handle, seg_handle = self._sample_points_from_mesh(handle, self.num_points // 2, segment_id=0)
            points_scoop, seg_scoop = self._sample_points_from_mesh(scoop, self.num_points // 2, segment_id=1)
            
            points = np.vstack([points_handle, points_scoop])
            segments = np.concatenate([seg_handle, seg_scoop])
            
            affordances = {
                0: ["graspable"],  # handle
                1: ["scoopable"]  # scoop
            }
            
        else:  # hook
            # Hook: curved cylinder
            radius = 0.05
            curve_radius = 0.2
            
            # Create curved path
            theta = np.linspace(0, np.pi, 20)
            path = np.column_stack([
                curve_radius * np.cos(theta),
                np.zeros_like(theta),
                curve_radius * np.sin(theta)
            ])
            
            # Create hook mesh along path
            hook = trimesh.creation.cylinder(radius=radius, height=1.0, sections=16)
            
            points, segments = self._sample_points_from_mesh(hook, self.num_points, segment_id=0)
            
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
            # Ring: torus
            major_radius = 0.4 + np.random.rand() * 0.2
            minor_radius = 0.05 + np.random.rand() * 0.05
            
            ring = trimesh.creation.annulus(
                r_min=major_radius - minor_radius,
                r_max=major_radius + minor_radius,
                height=minor_radius * 2,
                sections=32
            )
            
            points, segments = self._sample_points_from_mesh(ring, self.num_points, segment_id=0)
            
            affordances = {
                0: ["passable_through"]
            }
            
        else:  # shelf
            # Shelf: flat plane
            width = 0.8 + np.random.rand() * 0.4
            depth = 0.4 + np.random.rand() * 0.2
            thickness = 0.05
            
            shelf = trimesh.creation.box(extents=[width, depth, thickness])
            
            points, segments = self._sample_points_from_mesh(shelf, self.num_points, segment_id=0)
            
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
        # Cube with handle: cube (body) + curved cylinder (handle)
        cube_size = 0.5 + np.random.rand() * 0.2
        
        # Cube body
        cube = trimesh.creation.box(extents=[cube_size, cube_size, cube_size])
        
        # Handle (curved cylinder)
        handle_radius = 0.05
        handle = trimesh.creation.cylinder(radius=handle_radius, height=cube_size * 0.6, sections=16)
        handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
        handle.apply_translation([cube_size * 0.6, 0, cube_size * 0.3])
        
        points_cube, seg_cube = self._sample_points_from_mesh(cube, self.num_points // 2, segment_id=0)
        points_handle, seg_handle = self._sample_points_from_mesh(handle, self.num_points // 2, segment_id=1)
        
        points = np.vstack([points_cube, points_handle])
        segments = np.concatenate([seg_cube, seg_handle])
        
        affordances = {
            0: ["stackable"],  # cube body
            1: ["graspable"]  # handle
        }
        
        return {
            "points": torch.FloatTensor(points),
            "segments": torch.LongTensor(segments),
            "affordances": affordances,
            "category": "composite",
            "type": "cube_with_handle"
        }
    
    def _sample_points_from_mesh(
        self,
        mesh: trimesh.Trimesh,
        num_points: int,
        segment_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points from mesh surface."""
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        segments = np.full(num_points, segment_id, dtype=np.int64)
        return points, segments
    
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
