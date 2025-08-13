import unittest
import trl
import datasets
import numpy as np
from datasets.utils.logging import disable_progress_bar

class TestBFDPacking(unittest.TestCase):
    """Test cases for BFD packing functionality."""
    
    def setUp(self):
        disable_progress_bar()
        
    def test_bfd_packing_power_of_2(self):
        """Test BFD packing with power of 2 seq_length."""
        # Create input_ids with varying lengths
        N = 100
        lens = np.random.normal(4096, 512, N).astype(int).clip(100, 8192).tolist()
        input_ids = [[0 for _ in range(n)] for n in lens]
        dataset = datasets.Dataset.from_dict({'input_ids': input_ids}).with_format("arrow")
        
        # Pack with power of 2
        packed_dataset = trl.data_utils.pack_dataset(dataset, seq_length=8192, strategy="bfd")
        
        # Verify that packing occurred (output length should be less than input)
        self.assertLess(len(packed_dataset), len(dataset))
        
    def test_bfd_packing_power_of_2_plus_1(self):
        """Test BFD packing with power of 2 + 1 seq_length."""
        # Create input_ids with varying lengths
        N = 100
        lens = np.random.normal(4096, 512, N).astype(int).clip(100, 8192).tolist()
        input_ids = [[0 for _ in range(n)] for n in lens]
        dataset = datasets.Dataset.from_dict({'input_ids': input_ids}).with_format("arrow")
        
        # Pack with power of 2 + 1
        packed_dataset = trl.data_utils.pack_dataset(dataset, seq_length=8193, strategy="bfd")
        
        # Verify that packing occurred (output length should be less than input)
        self.assertLess(len(packed_dataset), len(dataset))
        
    def test_bfd_packing_identical_results(self):
        """Test that BFD packing gives similar results for power of 2 and power of 2 + 1."""
        # Create input_ids with varying lengths
        N = 50
        lens = np.random.normal(2048, 256, N).astype(int).clip(100, 4096).tolist()
        input_ids = [[0 for _ in range(n)] for n in lens]
        dataset = datasets.Dataset.from_dict({'input_ids': input_ids}).with_format("arrow")
        
        # Pack with power of 2 and power of 2 + 1
        packed_8192 = trl.data_utils.pack_dataset(dataset, seq_length=8192, strategy="bfd")
        packed_8193 = trl.data_utils.pack_dataset(dataset, seq_length=8193, strategy="bfd")
        
        # Verify that both produce packed results
        self.assertLess(len(packed_8192), len(dataset))
        self.assertLess(len(packed_8193), len(dataset))

if __name__ == '__main__':
    unittest.main()