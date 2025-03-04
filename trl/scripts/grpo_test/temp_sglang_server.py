
import os
import sys
import time
import signal

# Handle termination signals gracefully
def signal_handler(sig, frame):
    print("Received termination signal, shutting down...")
    if 'engine' in globals():
        engine.shutdown()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

try:
    import sglang as sgl
    
    # Initialize engine with model
    print("Initializing SGLang Engine with model: Qwen/Qwen2.5-0.5B-Instruct")
    engine = sgl.Engine(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        base_gpu_id=0,
        random_seed=42,
        mem_fraction_static=0.5,
    )
    
    # Start server
    print("Starting server on port 12355")
    engine.start_server(port=12355)
    
    # Keep the script running
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    print("Keyboard interrupt received, shutting down...")
    if 'engine' in globals():
        engine.shutdown()
        
except Exception as e:
    print(f"Error in server: {e}")
    import traceback
    traceback.print_exc()
    if 'engine' in globals():
        engine.shutdown()
    sys.exit(1)
