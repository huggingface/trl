from sglang_server_module.server import SGLangServer, setup_signal_handlers

def main():
    # Create and start server
    server = SGLangServer(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        port=12355
    )
    
    # Set up signal handlers
    setup_signal_handlers(server)
    
    # Start the engine and server
    server.start()
    
    # Keep running
    server.run_forever()

if __name__ == "__main__":
    main()
