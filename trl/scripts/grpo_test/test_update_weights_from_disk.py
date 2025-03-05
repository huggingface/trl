import os
import tempfile
import unittest
import torch
import sglang as sgl


def init_sglang_engine(
    rank: int, base_gpu_id: int, mem_fraction: float, random_seed: int, model_path: str
):
    """
    Initialize the SGLang offline engine on rank 0.

    Args:
        rank (int): Process rank.
        base_gpu_id (int): GPU ID for SGLang engine.
        mem_fraction (float): Memory fraction for static allocation.
        random_seed (int): Random seed for reproducibility.
        model_path (str): Path to the model.

    Returns:
        engine instance if successful, None otherwise.
    """
    if rank != 0:
        print(
            f"[Rank {rank}] Skipping SGLang engine initialization (only rank 0 initializes)"
        )
        return None

    print(
        f"[Rank {rank}] Initializing SGLang engine on GPU {base_gpu_id} with memory fraction {mem_fraction}"
    )
    try:
        engine = sgl.Engine(
            model_path=model_path,
            base_gpu_id=base_gpu_id,
            random_seed=random_seed,
            mem_fraction_static=mem_fraction,
        )
        print(f"[Rank {rank}] SGLang engine initialized successfully")
        return engine
    except Exception as e:
        print(f"[Rank {rank}] Failed to initialize SGLang engine: {e}")
        import traceback

        traceback.print_exc()
        return None


class TestOfflineEngineUpdateWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use a test model path (adjust as needed)
        cls.model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        # Initialize the offline engine on rank 0.
        cls.engine = init_sglang_engine(
            rank=0,
            base_gpu_id=4,
            mem_fraction=0.5,
            random_seed=42,
            model_path=cls.model_path,
        )
        assert cls.engine is not None, "Failed to initialize the SGLang offline engine"

    def test_update_weights_valid_checkpoint(self):
        # Create a temporary file to simulate a valid checkpoint.
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            # Write a dummy state dictionary to the checkpoint file.
            dummy_state = {"dummy": 123}
            torch.save(dummy_state, tmp_path)

        try:
            # Call update_weights_from_disk with the valid checkpoint.
            result = self.engine.update_weights_from_disk(tmp_path)
            self.assertTrue(
                result,
                "update_weights_from_disk should return True for a valid checkpoint",
            )
        finally:
            os.remove(tmp_path)

    def test_update_weights_invalid_checkpoint(self):
        # Provide a non-existent checkpoint path.
        fake_path = "non_existent_checkpoint.bin"
        result = self.engine.update_weights_from_disk(fake_path)
        self.assertFalse(
            result,
            "update_weights_from_disk should return False for an invalid checkpoint",
        )


if __name__ == "__main__":
    # __main__ guard is required when using spawn mode for subprocesses.
    unittest.main()
