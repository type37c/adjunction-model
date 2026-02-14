import unittest
import torch
from torch.utils.data import DataLoader
import sys
import os

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.purposeless_dataset import PurposelessAssemblyDataset, collate_purposeless_batch
from src.models.adjunction_model import AdjunctionModel
from experiments.purpose_emergent_experiment import DisplacementHead, PurposeEmergentTrainer

class TestPurposeEmergent(unittest.TestCase):
    def setUp(self):
        self.num_points = 256
        self.dataset = PurposelessAssemblyDataset(num_samples=4, num_points=self.num_points)
        self.dataloader = DataLoader(self.dataset, batch_size=2, collate_fn=collate_purposeless_batch)
        
    def test_dataset_output(self):
        batch = next(iter(self.dataloader))
        self.assertIn('initial_points', batch)
        self.assertEqual(batch['initial_points'].shape[0], 2 * self.num_points)
        
    def test_model_initialization(self):
        model = AdjunctionModel(num_points=self.num_points)
        self.assertIsNotNone(model)
        
    def test_displacement_head(self):
        head = DisplacementHead(context_dim=128, num_affordances=5)
        affordances = torch.randn(2 * self.num_points, 5)
        batch_idx = torch.cat([torch.zeros(self.num_points), torch.ones(self.num_points)]).long()
        context_batch = torch.randn(2, 128)
        out = head(context_batch, affordances, batch_idx)
        self.assertEqual(out.shape, (2 * self.num_points, 3))

if __name__ == '__main__':
    unittest.main()
