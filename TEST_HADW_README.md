# Testing GRPO with HA-DW

This directory contains a test script to verify the HA-DW implementation on your local machine.

## Requirements

Make sure you have the necessary dependencies installed:

```bash
pip install torch transformers datasets accelerate
```

## Running the Test

### Test with HA-DW enabled (default)

```bash
python test_hadw_grpo.py
```

This will:
- Load a small Qwen 2.5 0.5B model
- Create a synthetic math dataset (32 simple addition problems)
- Train with GRPO + HA-DW for 1 epoch
- Display HA-DW metrics during training

### Test baseline GRPO (for comparison)

```bash
python test_hadw_grpo.py --no-hadw
```

This runs the same test but with HA-DW disabled, allowing you to compare the training dynamics.

## What to Expect

The script will:

1. ✅ Check MPS availability
2. ✅ Load the model (Qwen2.5-0.5B-Instruct)
3. ✅ Create a small synthetic dataset
4. ✅ Initialize the GRPO trainer with HA-DW
5. ✅ Run training for 1 epoch
6. ✅ Display HA-DW metrics:
   - `hadw/capability_prior`: Model's evolving capability estimate
   - `hadw/capability_posterior`: Updated capability after each batch
   - `hadw/batch_accuracy`: Accuracy on current batch
   - `hadw/eta_t`: Adaptive forgetting factor
   - `hadw/reweighting_mean`: Average reweighting factor
   - `hadw/reweighting_std`: Std dev of reweighting factors

## Expected Output

You should see output like:

```
================================================================================
Testing GRPO with HA-DW on MPS
================================================================================
✓ MPS is available

📦 Loading model: Qwen/Qwen2.5-0.5B-Instruct
   Device: mps
   Using dtype: torch.float16

📊 Creating synthetic dataset...
   Dataset size: 32 samples
   Example prompt: What is 0 + 3? Answer with just the number.
   Example answer: 3

⚙️  Configuring GRPO with HA-DW...
   ✓ HA-DW enabled: True
   ✓ Eta: 0.1
   ✓ Lambda scale: 1.0
   ✓ History window: 5
   ✓ Num generations: 4

🚀 Initializing GRPO Trainer...
   ✓ Trainer initialized successfully

🏋️  Starting training...
[Training logs...]

📈 HA-DW Metrics:
   hadw/capability_prior:
     - First: 0.2500
     - Last:  0.3125
     - Mean:  0.2812
   hadw/capability_posterior:
     - First: 0.2625
     - Last:  0.3250
     - Mean:  0.2937
   [...]

================================================================================
✅ Test completed successfully!
================================================================================
```

## Troubleshooting

### MPS not available
If you see "MPS not available", the script will fall back to CPU. This is normal on non-Apple Silicon machines.

### Out of memory
If you run out of memory, try:
- Reducing `per_device_train_batch_size` in the script (currently 2)
- Reducing `num_generations` (currently 4)
- Using an even smaller dataset

### Model download issues
The first run will download the model (~500MB). Ensure you have:
- Internet connection
- Sufficient disk space
- HuggingFace access (no token needed for this public model)

## Comparing with and without HA-DW

To see the effect of HA-DW, run both versions and compare the metrics:

```bash
# With HA-DW
python test_hadw_grpo.py > with_hadw.log 2>&1

# Without HA-DW
python test_hadw_grpo.py --no-hadw > without_hadw.log 2>&1

# Compare
diff with_hadw.log without_hadw.log
```

You should observe that HA-DW:
- Adjusts advantages based on prompt difficulty
- Tracks model capability evolution across batches
- Applies adaptive reweighting to correct bias
