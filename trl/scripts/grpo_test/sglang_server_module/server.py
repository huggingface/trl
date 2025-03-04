import os
import sys
import time
import signal
from typing import Optional

class SGLangServer:
    """
    SGLang engine server implementation.
    """
    
    def __init__(self, model_path: str, port: int = 12355):
        """
        Initialize the SGLang engine server.
        
        Args:
            model_path: Path to the model to use
            port: Port number to listen on
        """
        self.model_path = model_path
        self.port = port
        self.engine = None
    
    def start(self) -> None:
        """
        Start the SGLang engine and server.
        """
        import sglang as sgl
        
        # Initialize engine with model
        print(f"Initializing SGLang Engine with model: {self.model_path}")
        self.engine = sgl.Engine(
            model_path=self.model_path,
            base_gpu_id=0,
            random_seed=42,
            mem_fraction_static=0.5,
        )
        
        # Start server
        print(f"Starting server on port {self.port}")
        self.engine.start_server(port=self.port)
    
    def run_forever(self) -> None:
        """
        Keep the server running until interrupted.
        """
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()
    
    def shutdown(self) -> None:
        """
        Shutdown the engine properly.
        """
        if self.engine is not None:
            print("Shutting down SGLang engine...")
            self.engine.shutdown()
            self.engine = None


def setup_signal_handlers(server: SGLangServer) -> None:
    """
    Set up signal handlers for graceful shutdown.
    
    Args:
        server: SGLangServer instance to shut down
    """
    def signal_handler(sig, frame):
        print(f"Received signal {sig}, shutting down...")
        server.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
