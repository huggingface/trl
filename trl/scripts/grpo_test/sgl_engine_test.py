import sglang as sgl


def main():
    llm = sgl.Engine(model_path="Qwen/Qwen2.5-0.5B-Instruct")
    result = llm.generate("What is the capital of France?")
    print("Generated output:", result)
    llm.shutdown()


if __name__ == "__main__":
    main()
